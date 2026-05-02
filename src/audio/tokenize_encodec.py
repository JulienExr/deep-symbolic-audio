import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoProcessor, EncodecModel

try:
    from .audio_io import load_audio, probe_audio
except ImportError:
    from audio_io import load_audio, probe_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize WAV segments with pretrained EnCodec.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Dossier de sortie de prepare_audio.py contenant segments/ et metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Dossier où sauvegarder les tokens .pt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/encodec_32khz",
        help="Nom du modèle EnCodec Hugging Face.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="Bande passante cible EnCodec si supportée par le modèle.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de fichiers à traiter.",
    )
    return parser.parse_args()


def load_segment_paths(input_dir):
    segment_dir = Path(input_dir) / "segments"
    if not segment_dir.exists():
        raise FileNotFoundError(f"Le dossier {segment_dir} n'existe pas. Assurez-vous d'avoir exécuté prepare_audio.py.")
    
    wavs = sorted(segment_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Aucun fichier .wav trouvé dans {segment_dir}.")
    return wavs


def load_audio_mono(path, target_sr=32000):
    metadata = probe_audio(path)

    if metadata.sample_rate != target_sr:
        raise ValueError(f"Le taux d'échantillonnage de {path} est {metadata.sample_rate}, attendu {target_sr}. Assurez-vous que prepare_audio.py a correctement resamplé les fichiers.")
    
    if metadata.channels != 1:
        raise ValueError(f"Le fichier {path} n'est pas mono. Assurez-vous que prepare_audio.py a correctement converti les fichiers en mono.")
    
    wave_form, _ = load_audio(path)
    return wave_form


def infer_expected_sample_rate(processor, model):
    if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
        return processor.feature_extractor.sampling_rate
    elif hasattr(model.config, "sampling_rate"):
        return model.config.sampling_rate
    else:
        raise ValueError("Impossible de déterminer le taux d'échantillonnage attendu du modèle. Veuillez vérifier la documentation du modèle ou spécifier manuellement.")
    

def set_model_bandwidth(model, bandwidth):
    if bandwidth is None:
        return
    if hasattr(model, "set_target_bandwidth"):
        model.set_target_bandwidth(bandwidth)
    else:
        print(f"Attention : le modèle {model.__class__.__name__} ne supporte pas la configuration de la bande passante. Ignoring --bandwidth argument.")
    

def detach_to_cpu(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


@torch.no_grad()
def encode_audio(wave_form, processor, model, sample_rate):
    inputs = processor(
        raw_audio=wave_form.squeeze(0).cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
                       )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    encoded = model.encode(**inputs)

    audio_codes = encoded.audio_codes
    audio_scales = getattr(encoded, "audio_scales", None)

    if audio_codes.dim() == 4:
        # souvent [frames, batch, codebooks, seq]
        if audio_codes.size(0) != 1:
            raise ValueError(
                f"Nombre de frames inattendu ({audio_codes.size(0)}). "
                "Pour ce script on attend 1 frame logique côté sortie encode."
            )
        codes = audio_codes[0, 0]  
        # [num_codebooks, seq]
    elif audio_codes.dim() == 3:
        # souvent [batch, codebooks, seq]
        codes = audio_codes[0]
    elif audio_codes.dim() == 2:
        # déjà [codebooks, seq] ou [1, seq]
        codes = audio_codes
    else:
        raise ValueError(f"Shape audio_codes non gérée: {tuple(audio_codes.shape)}")

    codes = codes.to(torch.long).cpu()
    audio_scales = detach_to_cpu(audio_scales)

    return codes, audio_scales

def save_token_file(out_path, audio_path, segment_id, codes, audio_scales, sample_rate, bandwidth, model_name):

    payload = {
        "audio_path": str(audio_path),
        "segment_id": segment_id,
        "codes": codes,
        "audio_scales": audio_scales,
        "sample_rate": sample_rate,
        "num_codebooks": codes.shape[0] if codes.dim() > 1 else 1,
        "num_frames": codes.shape[1] if codes.dim() > 1 else codes.shape[0],
        "bandwidth": bandwidth,
        "model_name": model_name,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    token_dir = output_dir / "tokens"
    metadata_csv = output_dir / "metadata.csv"

    segment_paths = load_segment_paths(input_dir)
    if args.limit is not None:
        segment_paths = segment_paths[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model and processor for {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = EncodecModel.from_pretrained(args.model_name).to(device)

    model.eval()

    set_model_bandwidth(model, args.bandwidth)
    expected_sr = infer_expected_sample_rate(processor, model)
    print(f"Expected sample rate for model: {expected_sr} Hz")
    print(f"Processing {len(segment_paths)} segments...")


    rows = []
    ok_count = 0
    fail_count = 0


    for idx, segment_path in enumerate(segment_paths, start=1):
        segment_id = segment_path.stem
        out_path = token_dir / f"{segment_id}.pt"
        print(f"Processing segment {idx}/{len(segment_paths)}: {segment_path.name} -> {out_path.name}")

        if out_path.exists():
            print(f"Token file {out_path} already exists, skipping.")
            continue

        try:
            wave_form = load_audio_mono(segment_path, target_sr=expected_sr)
            codes, audio_scales = encode_audio(wave_form, processor, model, expected_sr)
            if codes.dim() != 2:
                raise ValueError(f"Codes shape inattendue pour {segment_path}: {tuple(codes.shape)}. Attendu [num_codebooks, seq].")    
            
            save_token_file(out_path, segment_path, segment_id, codes, audio_scales, expected_sr, args.bandwidth, args.model_name)

            rows.append({
                "segment_id": segment_id,
                "audio_path": str(segment_path),
                "token_path": str(out_path),
                "sample_rate": expected_sr,
                "num_codebooks": codes.shape[0] if codes.dim() > 1 else 1,
                "num_frames": codes.shape[1] if codes.dim() > 1 else codes.shape[0],
                "bandwidth": args.bandwidth,
                "model_name": args.model_name,
            })
            ok_count += 1
            print(f"Successfully processed {segment_path.name}, saved tokens to {out_path.name}.")
        except Exception as e:
            print(f"Error processing {segment_path.name}: {e}")
            fail_count += 1

    fieldnames = ["segment_id", "audio_path", "token_path", "sample_rate", "num_codebooks", "num_frames", "bandwidth", "model_name"]

    with metadata_csv.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Finished processing. {ok_count} segments successfully tokenized, {fail_count} failed. Metadata saved to {metadata_csv}.")

if __name__ == "__main__":
    main()
