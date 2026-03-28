import argparse
import random
import re
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in this project, this is defensive.
    np = None


EMOPIA_EMOTIONS = ("HAPPY", "SAD", "ANGRY", "RELAXED")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train, preprocess, fine-tune or generate symbolic music with LSTM and Transformer models."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Build dataset tensors and vocab files from a MIDI directory.",
    )
    preprocess_parser.add_argument("--tokenizer-mode", choices=["mono", "poly", "emopia"], default="mono")
    preprocess_parser.add_argument("--input-dir", default=None, help="Directory containing source MIDI files.")
    preprocess_parser.add_argument("--dataset-output", default=None, help="Output dataset path (.pt).")
    preprocess_parser.add_argument("--vocab-output", default=None, help="Output vocabulary path prefix.")
    preprocess_parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of MIDI files.")
    preprocess_parser.add_argument("--seq-length", type=int, default=None, help="Sequence length used for training examples.")
    preprocess_parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride used for training examples.")
    preprocess_parser.add_argument("--seed", type=int, default=42, help="Seed used for file ordering and dataset splits.")

    train_parser = subparsers.add_parser("train", help="Train an LSTM or Transformer model.")
    add_model_arguments(train_parser, tokenizer_modes=["mono", "poly", "emopia"])
    add_runtime_arguments(train_parser)
    add_training_arguments(train_parser)
    train_parser.add_argument("--dataset", default="data/processed/dataset.pt", help="Path to the training dataset.")
    train_parser.add_argument("--val-dataset", default=None, help="Optional path to the validation dataset.")

    fine_tune_parser = subparsers.add_parser("fine-tune", help="Fine-tune a checkpoint on a new vocabulary.")
    add_model_arguments(fine_tune_parser, tokenizer_modes=["mono", "poly", "emopia"])
    add_runtime_arguments(fine_tune_parser)
    add_training_arguments(fine_tune_parser)
    fine_tune_parser.add_argument("--dataset", default="data/processed/dataset.pt", help="Path to the training dataset.")
    fine_tune_parser.add_argument("--val-dataset", default=None, help="Optional path to the validation dataset.")
    fine_tune_parser.add_argument("--checkpoint", default=None, help="Checkpoint to load before fine-tuning.")
    fine_tune_parser.add_argument("--old-vocab", default=None, help="Original token_to_id JSON used by the checkpoint.")
    fine_tune_parser.add_argument("--new-vocab", default=None, help="New token_to_id JSON used for fine-tuning.")
    fine_tune_parser.add_argument("--fine-tune-tag", default="fine_tune", help="Suffix used when saving fine-tuned checkpoints.")

    generate_parser = subparsers.add_parser("generate", help="Generate a MIDI file from a trained checkpoint.")
    add_model_arguments(generate_parser, tokenizer_modes=["mono", "poly", "emopia"])
    add_runtime_arguments(generate_parser)
    generate_parser.add_argument("--checkpoint", default=None, help="Checkpoint to load for generation.")
    generate_parser.add_argument("--output-midi", default="outputs/generated_music.mid", help="Output MIDI path.")
    generate_parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate.")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation.")
    generate_parser.add_argument("--top-k", type=int, default=10, help="Top-k sampling used for Transformer generation.")
    generate_parser.add_argument("--emotion", choices=EMOPIA_EMOTIONS, default=None, help="Emotion label used for emopia generation.")
    generate_parser.add_argument("--start-token", default=None, help="Explicit start token for generation, e.g. START or START_HAPPY.")
    generate_parser.add_argument("--metrics-output", default=None, help="Optional JSON path for generation metrics.")

    return parser


def add_model_arguments(parser, tokenizer_modes):
    parser.add_argument("--model", choices=["lstm", "transformer", "transformer_giantmidi"], default="lstm", help="Model architecture to use.")
    parser.add_argument("--tokenizer-mode", choices=tokenizer_modes, default=tokenizer_modes[0], help="Vocabulary/tokenization mode to use.")


def add_runtime_arguments(parser):
    parser.add_argument("--device", default=None, help="Torch device to use, e.g. cpu or cuda. Defaults to auto-detect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")


def add_training_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total optimizer steps used for linear warmup in Transformer training.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.0,
        help="Final learning-rate ratio relative to --lr after cosine decay in Transformer training.",
    )


def main(root_dir):
    args = build_parser().parse_args()
    set_seed(args.seed)

    if args.action == "preprocess":
        run_preprocess(args, root_dir)
        return

    device = resolve_device(args.device)

    if args.action == "train":
        run_training(args, root_dir, device)
        return

    if args.action == "fine-tune":
        run_fine_tuning(args, root_dir, device)
        return

    run_generation(args, root_dir, device)


def set_seed(seed):
    if seed is None:
        return

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)


def resolve_device(device_name):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch est requis pour les commandes train, fine-tune et generate.") from exc

    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(root_dir, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root_dir / path


def resolve_validation_dataset_path(root_dir, dataset_path, val_dataset_path=None):
    if val_dataset_path is not None:
        return resolve_path(root_dir, val_dataset_path)

    dataset_path = Path(dataset_path)
    candidates = []
    if dataset_path.name.endswith("_train.pt"):
        candidates.append(dataset_path.with_name(dataset_path.name.replace("_train.pt", "_val.pt")))

    if dataset_path.suffix == ".pt":
        candidates.append(dataset_path.with_name(dataset_path.stem + "_val.pt"))

    for candidate in candidates:
        candidate_path = candidate if candidate.is_absolute() else root_dir / candidate
        if candidate_path.exists():
            return candidate_path

    return None


def build_model(model_name, vocab_size, tokenizer_mode="mono"):
    from models import build_music_lstm, build_music_transformer

    if model_name == "lstm":
        return build_music_lstm(vocab_size)
    return build_music_transformer(vocab_size, emotion_mode=(tokenizer_mode == "emopia"))


def build_vocab_for_mode(tokenizer_mode):
    from tokenizer import build_vocab, build_vocab_emopia, build_vocab_polyphonic

    if tokenizer_mode == "poly":
        return build_vocab_polyphonic()
    if tokenizer_mode == "emopia":
        return build_vocab_emopia()
    return build_vocab()


def default_checkpoint_path(root_dir, model_name, tokenizer_mode):
    def checkpoint_sort_key(path):
        match = re.search(r"_epoch_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    model_dir = root_dir / "models" / model_name
    final_path = model_dir / f"{model_name}_{tokenizer_mode}_final.pt"
    if final_path.exists():
        return final_path

    epoch_checkpoints = sorted(model_dir.glob(f"{model_name}_{tokenizer_mode}_epoch_*.pt"), key=checkpoint_sort_key)
    if epoch_checkpoints:
        return epoch_checkpoints[-1]

    legacy_final_path = root_dir / "models" / f"{model_name}_{tokenizer_mode}_final.pt"
    if legacy_final_path.exists():
        return legacy_final_path

    legacy_checkpoint_dir = root_dir / "checkpoints" / model_name / tokenizer_mode
    if legacy_checkpoint_dir.exists():
        legacy_checkpoints = sorted(
            legacy_checkpoint_dir.glob(f"{model_name}_{tokenizer_mode}_epoch_*.pt"),
            key=checkpoint_sort_key,
        )
        if legacy_checkpoints:
            return legacy_checkpoints[-1]

    return final_path


def get_model_dir(root_dir, model_name):
    model_dir = root_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_metrics_path(root_dir, model_name, tokenizer_mode):
    return get_model_dir(root_dir, model_name) / f"{model_name}_{tokenizer_mode}_metrics.json"


def get_fine_tune_report_path(root_dir, model_name, tokenizer_mode):
    return get_model_dir(root_dir, model_name) / f"{model_name}_{tokenizer_mode}_fine_tune_report.json"


def build_training_metadata(args, root_dir, device, lr, train_dataset_path, val_dataset_path=None):
    metadata = {
        "action": args.action,
        "model_name": args.model,
        "tokenizer_mode": args.tokenizer_mode,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": lr,
        "train_dataset_path": str(train_dataset_path),
        "val_dataset_path": str(val_dataset_path) if val_dataset_path is not None else None,
        "root_dir": str(root_dir),
    }
    if args.model != "lstm":
        metadata["warmup_ratio"] = args.warmup_ratio
        metadata["min_lr_ratio"] = args.min_lr_ratio
    if getattr(args, "checkpoint", None) is not None:
        metadata["checkpoint_path"] = str(resolve_path(root_dir, args.checkpoint))
    return metadata


def resolve_generation_start_token(args):
    if args.start_token and args.emotion:
        raise ValueError("Utilise soit --start-token soit --emotion, pas les deux en meme temps.")

    if args.tokenizer_mode != "emopia":
        if args.start_token or args.emotion:
            raise ValueError("--start-token et --emotion sont reserves au mode emopia.")
        return None

    if args.start_token:
        return args.start_token
    if args.emotion:
        return f"START_{args.emotion}"
    return None


def run_preprocess(args, root_dir):
    from tokenizer import create_vocab_and_dataset_for_mode, get_preprocess_defaults

    defaults = get_preprocess_defaults(args.tokenizer_mode)
    input_dir = resolve_path(root_dir, args.input_dir or defaults["input_dir"])
    dataset_output = resolve_path(root_dir, args.dataset_output or defaults["dataset_output_path"])
    vocab_output = resolve_path(root_dir, args.vocab_output or defaults["vocab_output_path"])

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Repertoire MIDI introuvable: {input_dir}. "
            f"Utilise --input-dir ou place les fichiers au chemin attendu."
        )

    create_vocab_and_dataset_for_mode(
        tokenizer_mode=args.tokenizer_mode,
        input_dir=str(input_dir),
        dataset_output_path=str(dataset_output),
        vocab_output_path=str(vocab_output),
        max_files=args.max_files,
        seq_length=args.seq_length,
        stride=args.stride,
        seed=args.seed,
    )

    print(f"Preprocessing termine pour le mode {args.tokenizer_mode}.")
    print(f"Dataset(s) ecrits a partir de: {dataset_output}")
    print(f"Vocabulaire ecrit avec le prefixe: {vocab_output}")


def run_training(args, root_dir, device):
    from dataset import load_dataloaders
    from train import train_lstm, train_transformer

    train_dataset_path = resolve_path(root_dir, args.dataset)
    default_lr = 8e-4 if args.model == "lstm" else 3e-4
    lr = args.lr if args.lr is not None else default_lr
    require_emotions = args.model == "transformer" and args.tokenizer_mode == "emopia"

    dataloader = load_dataloaders(
        str(train_dataset_path),
        batch_size=args.batch_size,
        require_emotions=require_emotions,
    )
    val_dataset_path = resolve_validation_dataset_path(root_dir, args.dataset, args.val_dataset)
    val_dataloader = None
    if val_dataset_path is not None:
        val_dataloader = load_dataloaders(
            str(val_dataset_path),
            batch_size=args.batch_size,
            shuffle=False,
            require_emotions=require_emotions,
        )

    metrics_metadata = build_training_metadata(
        args,
        root_dir=root_dir,
        device=device,
        lr=lr,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
    )

    vocab, _, _ = build_vocab_for_mode(args.tokenizer_mode)
    model = build_model(args.model, len(vocab), tokenizer_mode=args.tokenizer_mode)

    if args.model == "lstm":
        train_lstm(
            model,
            dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.epochs,
            lr=lr,
            device=device,
            tokenizer_mode=args.tokenizer_mode,
            metrics_metadata=metrics_metadata,
        )
        print(f"Metrics saved to {get_metrics_path(root_dir, args.model, args.tokenizer_mode)}")
        return

    if val_dataset_path is None:
        raise FileNotFoundError(
            "Validation dataset introuvable. Utilise --val-dataset ou un dataset train nomme *_train.pt avec son *_val.pt correspondant."
        )

    train_transformer(
        model,
        dataloader,
        val_dataloader,
        num_epochs=args.epochs,
        lr=lr,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        device=device,
        tokenizer_mode=args.tokenizer_mode,
        metrics_metadata=metrics_metadata,
    )
    print(f"Metrics saved to {get_metrics_path(root_dir, args.model, args.tokenizer_mode)}")


def run_fine_tuning(args, root_dir, device):
    from fine_tune import fine_tune_model
    from metrics import save_json

    checkpoint_path = resolve_path(root_dir, args.checkpoint) if args.checkpoint else default_checkpoint_path(root_dir, args.model, args.tokenizer_mode)
    if args.old_vocab is None or args.new_vocab is None:
        raise ValueError("Le fine-tuning requiert --old-vocab et --new-vocab.")

    old_vocab_path = resolve_path(root_dir, args.old_vocab)
    new_vocab_path = resolve_path(root_dir, args.new_vocab)
    train_dataset_path = resolve_path(root_dir, args.dataset)
    val_dataset_path = resolve_validation_dataset_path(root_dir, args.dataset, args.val_dataset)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    effective_lr = args.lr if args.lr is not None else (8e-4 if args.model == "lstm" else 3e-4)
    metrics_metadata = build_training_metadata(
        args,
        root_dir=root_dir,
        device=device,
        lr=effective_lr,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
    )
    metrics_metadata.update(
        {
            "old_vocab_path": str(old_vocab_path),
            "new_vocab_path": str(new_vocab_path),
            "fine_tune_tag": args.fine_tune_tag,
        }
    )

    _, transfer_report, training_report = fine_tune_model(
        checkpoint_path=str(checkpoint_path),
        old_vocab_path=str(old_vocab_path),
        new_vocab_path=str(new_vocab_path),
        train_dataset_path=str(train_dataset_path),
        model_name=args.model,
        val_dataset_path=str(val_dataset_path) if val_dataset_path is not None else None,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        tokenizer_mode=args.fine_tune_tag,
        device=device,
        metrics_metadata=metrics_metadata,
    )
    report_path = get_fine_tune_report_path(root_dir, args.model, args.fine_tune_tag)
    save_json(
        report_path,
        {
            "transfer_report": transfer_report,
            "training_report": training_report,
        },
    )
    print(f"Fine-tuning report saved to {report_path}")


def run_generation(args, root_dir, device):
    from generate import generate_tokens, tokens_to_midi
    from metrics import build_generation_report, save_json

    checkpoint_path = resolve_path(root_dir, args.checkpoint) if args.checkpoint else default_checkpoint_path(root_dir, args.model, args.tokenizer_mode)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    start_token = resolve_generation_start_token(args)
    generated_tokens = generate_tokens(
        model_name=args.model,
        checkpoint_path=str(checkpoint_path),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        tokenizer_mode=args.tokenizer_mode,
        start_token=start_token,
        device=device,
    )

    output_path = resolve_path(root_dir, args.output_midi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(generated_tokens)
    tokens_to_midi(generated_tokens, str(output_path), tokenizer_mode=args.tokenizer_mode)

    metrics_path = resolve_path(root_dir, args.metrics_output) if args.metrics_output else output_path.with_name(f"{output_path.stem}_metrics.json")
    generation_report = build_generation_report(
        generated_tokens,
        tokenizer_mode=args.tokenizer_mode,
        metadata={
            "action": args.action,
            "model_name": args.model,
            "checkpoint_path": str(checkpoint_path),
            "output_midi": str(output_path),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": str(device),
            "start_token": start_token or ("START_HAPPY" if args.tokenizer_mode == "emopia" else "START"),
        },
    )
    save_json(metrics_path, generation_report)
    print(f"Generation metrics saved to {metrics_path}")
