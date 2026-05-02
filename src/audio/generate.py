from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import torch
from transformers import AutoProcessor, EncodecModel

try:
    from .model import TransformerConfig, AudioTokenTransformer
except ImportError:
    from model import TransformerConfig, AudioTokenTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with a trained audio-token Transformer.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt du modèle entraîné.")
    parser.add_argument("--output_wav", type=str, required=True, help="Chemin du wav de sortie.")
    parser.add_argument("--model_name", type=str, default="facebook/encodec_32khz", help="Nom du modèle EnCodec.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--start_tokens_file", type=str, default=None,
                        help="Fichier .pt de tokens servant de préfixe. Si absent, génération depuis un token 0.")
    parser.add_argument("--start_tokens_len", type=int, default=32,
                        help="Nombre de tokens initiaux à prendre depuis start_tokens_file.")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
                        help="Nombre de nouveaux tokens à générer.")
    parser.add_argument("--temperature", type=float, default=0.95, help="Température sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--bandwidth", type=float, default=6.0, help="Bande passante EnCodec si supportée.")

    parser.add_argument("--seed", type=int, default=42, help="Seed pour la génération.")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_config" not in checkpoint:
        raise KeyError("checkpoint does not contain 'model_config' key")

    config_dict = dict(checkpoint["model_config"])
    if "max_seq_length" in config_dict and "max_seq_len" not in config_dict:
        config_dict["max_seq_len"] = config_dict.pop("max_seq_length")

    config = TransformerConfig(**config_dict)
    model = AudioTokenTransformer(config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def load_encodec(model_name, device, bandwidth):
    processor = AutoProcessor.from_pretrained(model_name)
    encodec_model = EncodecModel.from_pretrained(model_name).to(device)
    encodec_model.eval()

    if bandwidth is not None and hasattr(encodec_model, "set_target_bandwidth"):
        encodec_model.set_target_bandwidth(bandwidth)

    expected_sr = None
    if hasattr(processor, "sampling_rate") and processor.sampling_rate is not None:
        expected_sr = int(processor.sampling_rate)
    elif getattr(encodec_model.config, "sampling_rate", None) is not None:
        expected_sr = int(encodec_model.config.sampling_rate)

    if expected_sr is None:
        raise RuntimeError("Impossible to determine expected sample rate from the EnCodec model or processor.")

    return processor, encodec_model, expected_sr


def interleave_codebooks(codes):
    if codes.dim() != 2:
        raise ValueError(f"codes must have shape [K, T], received {tuple(codes.shape)}")

    return codes.transpose(0, 1).reshape(-1).to(torch.long)


def deinterleave_tokens(flat_tokens, num_codebooks):
    if flat_tokens.dim() != 1:
        raise ValueError(f"flat_tokens must be 1D, received {tuple(flat_tokens.shape)}")

    if num_codebooks <= 0:
        raise ValueError("num_codebooks must be > 0")

    n = flat_tokens.size(0)
    if n % num_codebooks != 0:
        raise ValueError(
            f"The total number of tokens ({n}) is not divisible by num_codebooks ({num_codebooks})"
        )

    num_frames = n // num_codebooks

    codes = flat_tokens.view(num_frames, num_codebooks).transpose(0, 1).contiguous()

    return codes.to(torch.long)


def load_start_tokens(start_tokens_file, start_tokens_len):
    if start_tokens_file is None:
        return torch.tensor([0], dtype=torch.long), 4

    payload = torch.load(start_tokens_file, map_location="cpu")
    if "codes" not in payload:
        raise KeyError(f"File {start_tokens_file} does not contain the key 'codes'")

    codes = payload["codes"]
    if codes.dim() != 2:
        raise ValueError(f"'codes' must have shape [K, T], received {tuple(codes.shape)}")

    num_codebooks = int(codes.size(0))
    flat = interleave_codebooks(codes)

    if start_tokens_len <= 0:
        raise ValueError("start_tokens_len must be > 0")

    flat = flat[:start_tokens_len]
    if flat.numel() == 0:
        raise ValueError("Prefix is empty after truncation start_tokens_len")

    return flat.to(torch.long), num_codebooks


@torch.no_grad()
def decode_codes_with_encodec(codes, encodec_model, device):
    if codes.dim() != 2:
        raise ValueError(f"codes must have shape [K, T], received {tuple(codes.shape)}")

    if encodec_model.config.chunk_length is not None:
        raise NotImplementedError(
            "This script currently supports only EnCodec models with chunk_length=None "
            "(for example facebook/encodec_32khz)."
        )

    audio_codes = codes.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, K, T]
    audio_scales = [torch.ones((1, 1), device=device)]

    decoded = encodec_model.decode(audio_codes=audio_codes, audio_scales=audio_scales)

    if not hasattr(decoded, "audio_values"):
        raise RuntimeError("The output of the EnCodec decode does not contain 'audio_values'")

    waveform = decoded.audio_values[0].detach().cpu()  # [channels, samples]

    return waveform


def save_waveform(path, waveform, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() != 2:
        raise ValueError(f"waveform must have shape [channels, samples], received {tuple(waveform.shape)}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH.")

    num_channels = int(waveform.size(0))
    pcm = (
        waveform.detach()
        .cpu()
        .to(torch.float32)
        .clamp(-1.0, 1.0)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .tobytes()
    )

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(num_channels),
        "-i",
        "pipe:0",
    ]

    if path.suffix.lower() == ".wav":
        command += ["-c:a", "pcm_s16le"]

    command.append(str(path))

    try:
        subprocess.run(
            command,
            input=pcm,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed to write audio to {path}: {stderr}") from exc


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_wav = Path(args.output_wav)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model, checkpoint = load_checkpoint(checkpoint_path, device)

    print("[INFO] Loading EnCodec...")
    processor, encodec_model, sample_rate = load_encodec(
        model_name=args.model_name,
        device=device,
        bandwidth=args.bandwidth,
    )

    print(f"[INFO] Sample rate EnCodec: {sample_rate}")

    start_tokens_flat, num_codebooks = load_start_tokens(
        start_tokens_file=args.start_tokens_file,
        start_tokens_len=args.start_tokens_len,
    )

    print(f"[INFO] Prefix: {start_tokens_flat.numel()} tokens")
    print(f"[INFO] Number of codebooks: {num_codebooks}")

    start_tokens = start_tokens_flat.unsqueeze(0).to(device)  # [1, T_start]

    print(f"[INFO] Generating {args.max_new_tokens} new tokens...")
    generated = model.generate(
        start_tokens=start_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_flat = generated[0].detach().cpu()

    print(f"[INFO] Total length generated: {generated_flat.numel()} tokens")

    usable_len = (generated_flat.numel() // num_codebooks) * num_codebooks
    if usable_len == 0:
        raise RuntimeError("Not enough tokens generated to reconstruct codes.")

    generated_flat = generated_flat[:usable_len]

    codes = deinterleave_tokens(generated_flat, num_codebooks=num_codebooks)
    print(f"[INFO] Codes reconstructed: shape={tuple(codes.shape)}")

    print("[INFO] Decoding audio with EnCodec...")
    waveform = decode_codes_with_encodec(
        codes=codes,
        encodec_model=encodec_model,
        device=device,
    )

    print(f"[INFO] Waveform reconstructed: shape={tuple(waveform.shape)}")

    save_waveform(output_wav, waveform, sample_rate)
    print(f"[FIN] Audio saved in: {output_wav}")


if __name__ == "__main__":
    main()
