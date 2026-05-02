from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    duration_seconds: float | None = None
    codec_name: str | None = None


def _run_command(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        binary = Path(args[0]).name
        raise RuntimeError(f"Required binary '{binary}' was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"Command failed: {' '.join(args)}") from exc


def probe_audio(path: Path) -> AudioMetadata:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    stream = next((item for item in payload.get("streams", []) if item.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio stream found in {path}.")

    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])

    duration_seconds = None
    raw_duration = stream.get("duration") or payload.get("format", {}).get("duration")
    if raw_duration is not None:
        duration_seconds = float(raw_duration)

    return AudioMetadata(
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration_seconds,
        codec_name=stream.get("codec_name"),
    )


def load_audio(path: Path, *, target_sr: int | None = None, mono: bool = False) -> tuple[torch.Tensor, int]:
    metadata = probe_audio(path)
    sample_rate = target_sr or metadata.sample_rate
    channels = 1 if mono else metadata.channels

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)]
    if mono:
        cmd.extend(["-ac", "1"])
    if target_sr is not None:
        cmd.extend(["-ar", str(target_sr)])
    cmd.extend(["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])

    result = _run_command(cmd)
    if not result.stdout:
        raise ValueError(f"Audio file {path} is empty.")

    wave_form = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()
    if wave_form.numel() % channels != 0:
        raise ValueError(
            f"Decoded sample count {wave_form.numel()} is not divisible by channel count {channels} for {path}."
        )

    wave_form = wave_form.view(-1, channels).transpose(0, 1).contiguous()
    return wave_form, sample_rate


def save_waveform(path: Path, wave_form: torch.Tensor, sample_rate: int) -> None:
    if wave_form.dim() != 2:
        raise ValueError("wave_form must have shape (channels, num_samples).")
    if wave_form.numel() == 0:
        raise ValueError(f"Refusing to save empty audio file to {path}.")

    pcm = wave_form.detach().cpu().clamp(-1.0, 1.0)
    pcm16 = (pcm * 32767.0).round().to(torch.int16).transpose(0, 1).contiguous()

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(wave_form.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.numpy().tobytes())
