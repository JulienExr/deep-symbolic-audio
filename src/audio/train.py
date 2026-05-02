from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

try:
    from .dataset import EncodecTokenDataset, interleave_codebooks
    from .model import TransformerConfig, AudioTokenTransformer
except ImportError:
    from dataset import EncodecTokenDataset, interleave_codebooks
    from model import TransformerConfig, AudioTokenTransformer


AUDIO_CHECKPOINT_DIR = Path("checkpoints/audio")
CHECKPOINT_EVERY_N_EPOCHS = 5
GENERATIONS_PER_SAVE = 5
GENERATION_PREFIX_FRAMES = 8
GENERATION_MAX_NEW_TOKENS = 1000
GENERATION_TEMPERATURE = 0.95
GENERATION_TOP_K = 50
DEFAULT_ENCODEC_MODEL = "facebook/encodec_32khz"
DEFAULT_ENCODEC_BANDWIDTH = 6.0
LOSS_PLOT_PATH = AUDIO_CHECKPOINT_DIR / "losses.png"
MPLCONFIGDIR = AUDIO_CHECKPOINT_DIR / ".matplotlib"
MODEL_CONFIG_FIELDS = ("vocab_size", "max_seq_len", "d_model", "n_heads", "n_layers", "d_ff", "dropout")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an audio-token Transformer.")
    
    parser.add_argument("--tokens_dir", type=str, required=True, help="Dossier contenant les .pt tokenisés.")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint .pt à recharger pour reprendre l'entraînement.")
    parser.add_argument("--seq_len", type=int, default=1024, help="Longueur de séquence.")
    parser.add_argument("--stride", type=int, default=512, help="Stride pour les fenêtres.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay AdamW.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--num_workers", type=int, default=0, help="Nombre de workers DataLoader.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion validation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparamètres du modèle
    parser.add_argument("--vocab_size", type=int, required=True, help="Taille du vocabulaire.")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Options dataset
    parser.add_argument("--preload", action="store_true", help="Précharge les séquences en RAM.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(tokens_dir, seq_len, stride, batch_size, num_workers, val_ratio, preload, seed):
    dataset = EncodecTokenDataset(
        tokens_dir=tokens_dir,
        seq_length=seq_len,
        stride=stride,
        preload=preload,
    )

    total_size = len(dataset)
    if total_size == 0:
        token_files = sorted(Path(tokens_dir).glob("*.pt"))
        if not token_files:
            raise ValueError(f"No token files found in {tokens_dir}.")

        flat_lengths = []
        sample_descriptions = []
        for path in token_files:
            payload = torch.load(path, map_location="cpu")
            codes = payload["codes"]
            if not isinstance(codes, torch.Tensor) or codes.dim() != 2:
                continue

            flat_len = int(codes.size(0) * codes.size(1))
            flat_lengths.append(flat_len)

            if len(sample_descriptions) < 3:
                sample_descriptions.append(f"{path.name}: codes{tuple(codes.shape)} -> {flat_len} tokens")

        if not flat_lengths:
            raise ValueError(
                f"No usable token tensors found in {tokens_dir}. "
                "Expected payloads containing a 2D 'codes' tensor."
            )

        max_flat_len = max(flat_lengths)
        recommended_seq_len = max_flat_len - 1
        raise ValueError(
            "No training windows could be created. "
            f"Found {len(token_files)} token files with flat lengths in "
            f"[{min(flat_lengths)}, {max_flat_len}] tokens. "
            f"Current seq_len={seq_len}, but next-token training requires seq_len <= {recommended_seq_len}. "
            f"Example files: {' | '.join(sample_descriptions)}"
        )

    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Invalid split: total={total_size}, train={train_size}, val={val_size}. "
            f"Reduce val_ratio or increase the dataset."
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(args, device):
    config = build_model_config_from_args(args)

    model = AudioTokenTransformer(config).to(device)
    return model


def build_model_config_from_args(args):
    return TransformerConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )


def normalize_checkpoint_model_config(config_dict):
    config_dict = dict(config_dict)
    if "max_seq_length" in config_dict and "max_seq_len" not in config_dict:
        config_dict["max_seq_len"] = config_dict.pop("max_seq_length")
    return config_dict


def warn_if_resume_config_differs(args, checkpoint_config):
    current_config = build_model_config_from_args(args).__dict__
    mismatches = []

    for field in MODEL_CONFIG_FIELDS:
        current_value = current_config[field]
        checkpoint_value = checkpoint_config[field]
        if current_value != checkpoint_value:
            mismatches.append(f"{field}: cli={current_value}, checkpoint={checkpoint_value}")

    if mismatches:
        print(
            "[WARN] Model hyperparameters passed on the CLI differ from the checkpoint. "
            "The checkpoint configuration will be used. "
            + " | ".join(mismatches)
        )


def load_training_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {"epoch", "model_state_dict", "model_config"}
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing keys: {', '.join(missing_keys)}")

    checkpoint["model_config"] = normalize_checkpoint_model_config(checkpoint["model_config"])
    return checkpoint_path, checkpoint


def create_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, scaler, best_val_loss, train_losses, val_losses, args, filename):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "model_config": model.config.__dict__,
        "train_args": vars(args),
    }

    torch.save(payload, checkpoint_dir / filename)


def plot_losses(checkpoint_dir, train_losses, val_losses):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not save loss plot: {exc}")
        return

    epochs = list(range(1, len(train_losses) + 1))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, train_losses, label="Train loss", linewidth=2)
    axis.plot(epochs, val_losses, label="Val loss", linewidth=2)
    axis.set_title("Audio Transformer Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(checkpoint_dir / LOSS_PLOT_PATH.name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_generation_sources(tokens_dir, sample_count, seed):
    token_files = sorted(Path(tokens_dir).glob("*.pt"))
    if not token_files:
        return []

    sample_count = min(sample_count, len(token_files))
    rng = random.Random(seed)
    selected = rng.sample(token_files, k=sample_count)
    return sorted(selected)


def load_generation_prefix(token_path):
    payload = torch.load(token_path, map_location="cpu")
    if "codes" not in payload:
        raise KeyError(f"File {token_path} does not contain the key 'codes'")

    codes = payload["codes"]
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"'codes' in {token_path} is not a Tensor")
    if codes.dim() != 2:
        raise ValueError(f"'codes' in {token_path} must have shape [K, T], received {tuple(codes.shape)}")

    codes = codes.to(torch.long)
    num_codebooks = int(payload.get("num_codebooks", codes.size(0)))
    available_frames = codes.size(1)
    prefix_frames = min(GENERATION_PREFIX_FRAMES, available_frames)
    if prefix_frames <= 0:
        raise ValueError(f"Token file {token_path} does not contain enough frames for generation.")

    prefix_codes = codes[:, :prefix_frames]
    prefix_tokens = interleave_codebooks(prefix_codes)

    return {
        "prefix_tokens": prefix_tokens,
        "num_codebooks": num_codebooks,
        "model_name": payload.get("model_name", DEFAULT_ENCODEC_MODEL),
        "bandwidth": payload.get("bandwidth", DEFAULT_ENCODEC_BANDWIDTH),
        "source_name": token_path.stem,
    }


def save_generation_samples(model, checkpoint_dir, epoch, generation_sources, device, seed):
    if not generation_sources:
        return

    try:
        try:
            from .generate import decode_codes_with_encodec, deinterleave_tokens, load_encodec, save_waveform
        except ImportError:
            from generate import decode_codes_with_encodec, deinterleave_tokens, load_encodec, save_waveform
    except Exception as exc:
        print(f"[WARN] Could not import audio generation utilities: {exc}")
        return

    output_dir = checkpoint_dir / "generations" / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)

    encodec_runtimes = {}
    decode_device = torch.device("cpu")

    for sample_index, token_path in enumerate(generation_sources, start=1):
        try:
            prefix = load_generation_prefix(token_path)
            runtime_key = (prefix["model_name"], prefix["bandwidth"])

            if runtime_key not in encodec_runtimes:
                _, encodec_model, sample_rate = load_encodec(
                    model_name=prefix["model_name"],
                    device=decode_device,
                    bandwidth=prefix["bandwidth"],
                )
                encodec_runtimes[runtime_key] = (encodec_model, sample_rate)

            encodec_model, sample_rate = encodec_runtimes[runtime_key]
            start_tokens = prefix["prefix_tokens"].unsqueeze(0).to(device)
            max_new_tokens = GENERATION_MAX_NEW_TOKENS

            fork_devices = [device.index] if device.type == "cuda" and device.index is not None else []
            if device.type == "cuda" and device.index is None:
                fork_devices = list(range(torch.cuda.device_count()))

            with torch.random.fork_rng(devices=fork_devices):
                generation_seed = seed + (epoch * 1000) + sample_index
                torch.manual_seed(generation_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(generation_seed)

                generated = model.generate(
                    start_tokens=start_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=GENERATION_TEMPERATURE,
                    top_k=GENERATION_TOP_K,
                )

            generated_flat = generated[0].detach().cpu()
            usable_len = (generated_flat.numel() // prefix["num_codebooks"]) * prefix["num_codebooks"]
            if usable_len == 0:
                raise RuntimeError("Not enough tokens generated to reconstruct codes.")
            generated_flat = generated_flat[:usable_len]
            codes = deinterleave_tokens(generated_flat, num_codebooks=prefix["num_codebooks"])
            waveform = decode_codes_with_encodec(
                codes=codes,
                encodec_model=encodec_model,
                device=decode_device,
            )

            output_path = output_dir / f"sample_{sample_index:02d}_{prefix['source_name']}.wav"
            save_waveform(output_path, waveform, sample_rate)
            print(f"[INFO] Generation saved: {output_path}")
        except Exception as exc:
            print(f"[WARN] Could not save generation from {token_path.name}: {exc}")


def train_one_epoch(model, train_loader, optimizer, scaler, device, grad_clip):
    model.train()

    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(train_loader, desc="Train", leave=False)

    for step, batch in enumerate(progress_bar, start=1):
        input_ids, target_ids = batch
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            _, loss = model.compute_loss(input_ids, target_ids)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{total_loss / total_batches:.4f}",
        )

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            _, loss = model.compute_loss(input_ids, target_ids)

        total_loss += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    resume_path = Path(args.resume_from) if args.resume_from is not None else None
    checkpoint_dir = AUDIO_CHECKPOINT_DIR
    generation_sources = select_generation_sources(
        tokens_dir=args.tokens_dir,
        sample_count=GENERATIONS_PER_SAVE,
        seed=args.seed,
    )

    print("[INFO] Device:", device)
    print(f"[INFO] Checkpoints dir: {checkpoint_dir}")
    print(f"[INFO] Generation sources selected: {len(generation_sources)}")
    if resume_path is not None:
        print(f"[INFO] Resume checkpoint: {resume_path}")
    print("[INFO] DataLoaders creation...")

    train_loader, val_loader = create_dataloaders(
        tokens_dir=args.tokens_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        preload=args.preload,
        seed=args.seed,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches  : {len(val_loader)}")

    resume_checkpoint = None
    if resume_path is not None:
        resume_path, resume_checkpoint = load_training_checkpoint(resume_path, device)
        warn_if_resume_config_differs(args, resume_checkpoint["model_config"])

    print("[INFO] Creating model...")
    if resume_checkpoint is not None:
        model = AudioTokenTransformer(
            TransformerConfig(**resume_checkpoint["model_config"])
        ).to(device)
    else:
        model = create_model(args, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Number of parameters: {n_params:,}")

    optimizer = create_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    start_epoch = 1

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])

        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        else:
            print("[WARN] optimizer_state_dict missing in checkpoint, optimizer reinitialized.")

        if "scaler_state_dict" in resume_checkpoint and scaler.is_enabled():
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        elif scaler.is_enabled():
            print("[WARN] scaler_state_dict missing in checkpoint, GradScaler reinitialized.")

        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        train_losses = list(resume_checkpoint.get("train_losses", []))
        val_losses = list(resume_checkpoint.get("val_losses", []))
        completed_epoch = int(resume_checkpoint["epoch"])
        start_epoch = completed_epoch + 1

        if len(train_losses) > completed_epoch:
            train_losses = train_losses[:completed_epoch]
        if len(val_losses) > completed_epoch:
            val_losses = val_losses[:completed_epoch]
        if train_losses and len(train_losses) != completed_epoch:
            print(
                f"[WARN] train_losses contains {len(train_losses)} points for a checkpoint at epoch {completed_epoch}."
            )
        if val_losses and len(val_losses) != completed_epoch:
            print(
                f"[WARN] val_losses contains {len(val_losses)} points for a checkpoint at epoch {completed_epoch}."
            )

        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {completed_epoch}. "
                f"Current --epochs={args.epochs} leaves nothing to run."
            )

        print(
            f"[INFO] Resumed from epoch {completed_epoch} | "
            f"next epoch={start_epoch} | best_val_loss={best_val_loss:.4f}"
        )

        if train_losses and val_losses:
            plot_losses(
                checkpoint_dir=checkpoint_dir,
                train_losses=train_losses,
                val_losses=val_losses,
            )

    print("[INFO] Beginning training...")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss, train_ppl = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
        )

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} | train_ppl={train_ppl:.2f} | "
            f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(
            checkpoint_dir=checkpoint_dir,
            train_losses=train_losses,
            val_losses=val_losses,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[INFO] New best model: val_loss={best_val_loss:.4f}")

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                train_losses=train_losses,
                val_losses=val_losses,
                args=args,
                filename="best.pt",
            )

        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                train_losses=train_losses,
                val_losses=val_losses,
                args=args,
                filename=f"epoch_{epoch}.pt",
            )
            save_generation_samples(
                model=model,
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                generation_sources=generation_sources,
                device=device,
                seed=args.seed,
            )

    print("\n[FIN] Training completed.")
    print(f"[INFO] Best val_loss: {best_val_loss:.4f}")
    print(f"[INFO] Loss plot saved in: {checkpoint_dir / LOSS_PLOT_PATH.name}")


if __name__ == "__main__":
    main()
