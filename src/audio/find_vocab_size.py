import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Inspecte la plage des tokens EnCodec sauvegardés.")
    parser.add_argument(
        "--tokens-dir",
        default="data/audio/tokenized/tokens",
        help="Dossier contenant les fichiers .pt tokenisés.",
    )
    return parser.parse_args()


def inspect_vocab_size(tokens_dir):
    token_paths = sorted(Path(tokens_dir).glob("*.pt"))
    if not token_paths:
        raise FileNotFoundError(f"Aucun fichier .pt trouvé dans {tokens_dir}.")

    max_token = -1
    min_token = 10**18

    for path in token_paths:
        payload = torch.load(path, map_location="cpu")
        codes = payload["codes"]
        max_token = max(max_token, int(codes.max().item()))
        min_token = min(min_token, int(codes.min().item()))

    return {
        "min_token": min_token,
        "max_token": max_token,
        "vocab_size": max_token + 1,
        "files": len(token_paths),
    }


def main():
    args = parse_args()
    stats = inspect_vocab_size(args.tokens_dir)
    print("files =", stats["files"])
    print("min_token =", stats["min_token"])
    print("max_token =", stats["max_token"])
    print("vocab_size =", stats["vocab_size"])


if __name__ == "__main__":
    main()
