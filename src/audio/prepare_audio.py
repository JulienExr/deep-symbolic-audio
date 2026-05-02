import hashlib
import csv
import os
from pathlib import Path

try:
    from .audio_io import load_audio, save_waveform
except ImportError:
    from audio_io import load_audio, save_waveform


def find_audio_files(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def safe_stem_from_path(path: Path):
    raw = str(path).replace("\\", "/")
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    stem = path.stem.replace(" ", "_")
    return f"{stem}_{digest}"


def peak_normalize(wave_form, target_peak=0.95):
    peak = wave_form.abs().max()
    if peak > 0:
        wave_form = wave_form * (target_peak / peak)
    return wave_form


def split_into_segments(wave_form, sample_rate, segment_seconds):
    if wave_form.dim() != 2 or wave_form.shape[0] != 1:
        raise ValueError("Input wave_form must be a mono audio tensor of shape (1, num_samples).")

    total_samples = wave_form.shape[1]
    segment_samples = int(segment_seconds * sample_rate)
    segments = []
    num_segments = total_samples // segment_samples

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        seg = wave_form[:, start:end]
        segments.append((seg, start, end))

    return segments


def seconds_from_samples(num_samples, sample_rate):
    return num_samples / sample_rate


def main():
    input_dir = Path("data/audio/2017")
    output_dir = Path("data/audio/prepared")
    segment_dir = output_dir / "segments"
    metadata_file = output_dir / "metadata.csv"
    sample_rate = 32000
    segment_seconds = 5

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir}.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    total_segments = 0
    total_files_ok = 0
    total_files_failed = 0

    print(f"sample_rate: {sample_rate}, segment_seconds: {segment_seconds}")
    print(f"Found {len(audio_files)} audio files. Processing...")

    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"Processing file {idx}/{len(audio_files)}: {audio_path}")
        try:
            wave_form, _ = load_audio(Path(audio_path), target_sr=sample_rate, mono=True)
            wave_form = peak_normalize(wave_form)

            segments = split_into_segments(wave_form, sample_rate, segment_seconds)

            if not segments:
                print(f"Warning: No segments created for {audio_path} (file too short).")
                total_files_failed += 1
                continue

            for seg_idx, (seg_waveform, start_sample, end_sample) in enumerate(segments):
                segment_filename = f"{safe_stem_from_path(Path(audio_path))}_segment{seg_idx+1}.wav"
                segment_path = segment_dir / segment_filename
                save_waveform(segment_path, seg_waveform, sample_rate)

                metadata_rows.append({
                    "original_file": str(audio_path),
                    "segment_file": str(segment_path),
                    "start_time": seconds_from_samples(start_sample, sample_rate),
                    "end_time": seconds_from_samples(end_sample, sample_rate),
                })

            total_segments += len(segments)
            total_files_ok += 1

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            total_files_failed += 1

    fieldnames = ["original_file", "segment_file", "start_time", "end_time"]
    with metadata_file.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n[FIN]")
    print(f"Fichiers OK      : {total_files_ok}")
    print(f"Fichiers erreurs : {total_files_failed}")
    print(f"Segments sauvés  : {total_segments}")
    print(f"Metadata         : {metadata_file}")


if __name__ == "__main__":
    main()
