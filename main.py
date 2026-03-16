import argparse
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import load_dataloaders
from fine_tune import fine_tune_model
from generate import generate_tokens, tokens_to_midi
from models import build_music_lstm, build_music_transformer
from tokenizer import build_vocab, build_vocab_polyphonic
from train import train_lstm, train_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train or generate symbolic music with an LSTM or Transformer model.")
    parser.add_argument("action", choices=["train", "generate", "fine-tune"], help="Run training, fine-tuning or MIDI generation.")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm", help="Model architecture to use.")
    parser.add_argument("--tokenizer-mode", choices=["mono", "poly"], default="mono", help="Vocabulary/tokenization mode to use.")
    parser.add_argument("--device", default="cuda", help="Torch device to use, e.g. cpu or cuda.")

    parser.add_argument("--dataset", default="data/processed/dataset.pt", help="Path to the training dataset.")
    parser.add_argument("--val-dataset", default=None, help="Optional path to the validation dataset.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")

    parser.add_argument("--checkpoint", default=None, help="Checkpoint to load for generation.")
    parser.add_argument("--old-vocab", default=None, help="Original token_to_id JSON used by the checkpoint.")
    parser.add_argument("--new-vocab", default=None, help="New token_to_id JSON used for fine-tuning.")
    parser.add_argument("--fine-tune-tag", default="fine_tune", help="Suffix used when saving fine-tuned checkpoints.")
    parser.add_argument("--output-midi", default="outputs/generated_music.mid", help="Output MIDI path for generation.")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for generation.")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, vocab_size):
    if model_name == "lstm":
        return build_music_lstm(vocab_size)

    return build_music_transformer(vocab_size)


def build_vocab_for_mode(tokenizer_mode):
    if tokenizer_mode == "poly":
        return build_vocab_polyphonic()
    return build_vocab()


def default_checkpoint_path(model_name, tokenizer_mode):
    model_dir = ROOT_DIR / "models" / model_name
    final_path = model_dir / f"{model_name}_{tokenizer_mode}_final.pt"
    if final_path.exists():
        return final_path

    epoch_checkpoints = sorted(model_dir.glob(f"{model_name}_{tokenizer_mode}_epoch_*.pt"))
    if epoch_checkpoints:
        return epoch_checkpoints[-1]

    legacy_final_path = ROOT_DIR / "models" / f"{model_name}_{tokenizer_mode}_final.pt"
    if legacy_final_path.exists():
        return legacy_final_path

    legacy_checkpoint_dir = ROOT_DIR / "checkpoints" / model_name / tokenizer_mode
    if legacy_checkpoint_dir.exists():
        legacy_checkpoints = sorted(legacy_checkpoint_dir.glob(f"{model_name}_{tokenizer_mode}_epoch_*.pt"))
        if legacy_checkpoints:
            return legacy_checkpoints[-1]

    return final_path


def resolve_validation_dataset_path(dataset_path, val_dataset_path=None):
    if val_dataset_path is not None:
        return ROOT_DIR / val_dataset_path

    dataset_path = Path(dataset_path)

    candidates = []
    if dataset_path.name.endswith("_train.pt"):
        candidates.append(dataset_path.with_name(dataset_path.name.replace("_train.pt", "_val.pt")))

    if dataset_path.suffix == ".pt":
        candidates.append(dataset_path.with_name(dataset_path.stem + "_val.pt"))

    for candidate in candidates:
        candidate_path = candidate if candidate.is_absolute() else ROOT_DIR / candidate
        if candidate_path.exists():
            return candidate_path

    return None


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path

def run_training(args, model, device):
    dataloader = load_dataloaders(str(ROOT_DIR / args.dataset), batch_size=args.batch_size)
    default_lr = 8e-4 if args.model == "lstm" else 3e-4
    lr = args.lr if args.lr is not None else default_lr

    if args.model == "lstm":
        train_lstm(model, dataloader, num_epochs=args.epochs, lr=lr, device=device, tokenizer_mode=args.tokenizer_mode)
    else:
        val_dataset_path = resolve_validation_dataset_path(args.dataset, args.val_dataset)
        if val_dataset_path is None:
            raise FileNotFoundError(
                "Validation dataset introuvable. Utilise --val-dataset ou un dataset train nommé *_train.pt avec son *_val.pt correspondant."
            )

        val_dataloader = load_dataloaders(str(val_dataset_path), batch_size=args.batch_size, shuffle=False)
        train_transformer(
            model,
            dataloader,
            val_dataloader,
            num_epochs=args.epochs,
            lr=lr,
            device=device,
            tokenizer_mode=args.tokenizer_mode,
        )


def run_fine_tuning(args, device):
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.model, args.tokenizer_mode)
    if args.old_vocab is None or args.new_vocab is None:
        raise ValueError("Le fine-tuning requiert --old-vocab et --new-vocab.")

    old_vocab_path = resolve_path(args.old_vocab)
    new_vocab_path = resolve_path(args.new_vocab)
    train_dataset_path = resolve_path(args.dataset)
    val_dataset_path = resolve_validation_dataset_path(args.dataset, args.val_dataset)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    fine_tune_model(
        checkpoint_path=str(checkpoint_path),
        old_vocab_path=str(old_vocab_path),
        new_vocab_path=str(new_vocab_path),
        train_dataset_path=str(train_dataset_path),
        model_name=args.model,
        val_dataset_path=str(val_dataset_path) if val_dataset_path is not None else None,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        tokenizer_mode=args.fine_tune_tag,
        device=device,
    )


def run_generation(args, device):
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.model, args.tokenizer_mode)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT_DIR / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    generated_tokens = generate_tokens(
        model_name=args.model,
        checkpoint_path=str(checkpoint_path),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tokenizer_mode=args.tokenizer_mode,
        device=device,
    )
    output_path = ROOT_DIR / args.output_midi
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(generated_tokens)
    tokens_to_midi(generated_tokens, str(output_path), tokenizer_mode=args.tokenizer_mode)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.action == "train":
        vocab, _, _ = build_vocab_for_mode(args.tokenizer_mode)
        model = build_model(args.model, len(vocab))
        run_training(args, model, device)
        return

    if args.action == "fine-tune":
        run_fine_tuning(args, device)
        return
    
    run_generation(args, device)


if __name__ == "__main__":
    main()
