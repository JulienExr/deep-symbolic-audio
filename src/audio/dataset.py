from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def interleave_codebooks(codes):
    if codes.dim() != 2:
        raise ValueError(f"Expected codes to be a 2D tensor of shape (num_codebooks, seq_length), got {tuple(codes.shape)}")
    
    return codes.transpose(0, 1).reshape(-1).to(torch.long)


def build_training_windows(flat_tokens, seq_length, stride):

    if flat_tokens.dim() != 1:
        raise ValueError(f"Expected flat_tokens to be a 1D tensor, got {tuple(flat_tokens.shape)}")

    if seq_length <= 0:
        raise ValueError(f"seq_length must be > 0, got {seq_length}")

    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    total_length = flat_tokens.size(0)
    windows = []

    # A next-token target needs one extra token beyond the input window.
    for start in range(0, total_length - seq_length, stride):
        end = start + seq_length
        windows.append((start, end))

    return windows

class EncodecTokenDataset(Dataset):
    def __init__(self, tokens_dir, seq_length, stride, preload=True):
        super().__init__()

        self.tokens_dir = Path(tokens_dir)
        self.seq_length = seq_length
        self.stride = stride
        self.preload = preload
        self.preloaded_sequences = []

        if not self.tokens_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.tokens_dir}")

        self.token_files = sorted(self.tokens_dir.glob("*.pt"))
        if not self.token_files:
            raise RuntimeError(f"No .pt files found in {self.tokens_dir}")
        
        self.examples = []
        if self.preload:
            self._build_index_preload()
        else:
            self._build_index_lazy()
    
    def _load_codes_from_file(self, path):
        payload = torch.load(path, map_location="cpu")

        if "codes" not in payload:
            raise KeyError(f"File {path} does not contain 'codes' key")

        codes = payload["codes"]

        if not isinstance(codes, torch.Tensor):
            raise TypeError(f"'codes' in {path} is not a Tensor")

        if codes.dim() != 2:
            raise ValueError(f"'codes' in {path} must have shape [K, T], received {tuple(codes.shape)}")

        return codes.to(torch.long)

    def _build_index_preload(self):
        for file_idx, path in enumerate(self.token_files):
            codes = self._load_codes_from_file(path)
            flat_tokens = interleave_codebooks(codes)

            self.preloaded_sequences.append(flat_tokens)

            windows = build_training_windows(
                flat_tokens=flat_tokens,
                seq_length=self.seq_length,
                stride=self.stride,
            )

            for start, end in windows:
                self.examples.append(
                    {
                        "file_idx": file_idx,
                        "start": start,
                        "end": end,
                        "path": str(path),
                    }
                )

    def _build_index_lazy(self):
        for file_idx, path in enumerate(self.token_files):
            codes = self._load_codes_from_file(path)
            flat_tokens = interleave_codebooks(codes)

            windows = build_training_windows(
                flat_tokens=flat_tokens,
                seq_length=self.seq_length,
                stride=self.stride,
            )

            for start, end in windows:
                self.examples.append(
                    {
                        "file_idx": file_idx,
                        "start": start,
                        "end": end,
                        "path": str(path),
                    }
                )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        start = ex["start"]
        end = ex["end"]

        if self.preload:
            flat_tokens = self.preloaded_sequences[ex["file_idx"]]
        else:
            path = Path(ex["path"])
            codes = self._load_codes_from_file(path)
            flat_tokens = interleave_codebooks(codes)

        input_ids = flat_tokens[start:end]
        target_ids = flat_tokens[start + 1 : end + 1]

        if input_ids.size(0) != self.seq_length:
            raise RuntimeError(
                f"input_ids has the wrong size: {input_ids.size(0)} instead of {self.seq_length}"
            )

        if target_ids.size(0) != self.seq_length:
            raise RuntimeError(
                f"target_ids has the wrong size: {target_ids.size(0)} instead of {self.seq_length}"
            )

        return input_ids, target_ids





