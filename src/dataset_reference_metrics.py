import argparse
import random
import statistics
from datetime import datetime
from pathlib import Path

import pretty_midi

from metrics import save_csv, save_json
from music_analysis import compute_music_metrics_from_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
METRIC_KEYS = [
    "note_count",
    "total_duration_sec",
    "note_density_per_sec",
    "silence_ratio",
    "mean_polyphony",
    "pitch_class_entropy",
    "tonal_center_strength",
    "consonance_ratio",
    "harmonic_motif_repetition_ratio_4",
    "rhythmic_diversity_ratio",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calcule des metriques de reference sur un dataset MIDI brut (ex: GiantMIDI)."
    )
    parser.add_argument("--input-dir", required=True, help="Dossier racine contenant les fichiers .mid/.midi")
    parser.add_argument("--max-files", type=int, default=None, help="Nombre max de fichiers a analyser")
    parser.add_argument("--seed", type=int, default=0, help="Seed utilisee pour melanger les fichiers")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Par defaut: outputs/dataset_reference/<timestamp>",
    )
    return parser


def list_midi_files(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Dossier introuvable: {input_path}")

    midi_files = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    ]
    return sorted(midi_files)


def build_output_paths(output_dir=None):
    if output_dir:
        run_dir = Path(output_dir)
    else:
        run_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT_DIR / "outputs" / "dataset_reference" / run_name

    return {
        "run_dir": run_dir,
        "json_path": run_dir / "report.json",
        "summary_csv_path": run_dir / "summary.csv",
        "per_file_csv_path": run_dir / "per_file.csv",
    }


def summarize_per_file(rows):
    summary = {}
    for key in METRIC_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def build_summary_csv_row(summary):
    row = {}
    for key in METRIC_KEYS:
        stats = summary.get(key)
        row[f"{key}.mean"] = stats["mean"] if stats else None
        row[f"{key}.std"] = stats["std"] if stats else None
        row[f"{key}.min"] = stats["min"] if stats else None
        row[f"{key}.max"] = stats["max"] if stats else None
    return row


def main():
    args = build_parser().parse_args()

    midi_files = list_midi_files(args.input_dir)
    random.Random(args.seed).shuffle(midi_files)
    if args.max_files is not None:
        midi_files = midi_files[: args.max_files]

    if not midi_files:
        raise ValueError("Aucun fichier MIDI trouve dans le dossier fourni.")

    output_paths = build_output_paths(args.output_dir)

    per_file_rows = []
    errors = []

    for index, midi_path in enumerate(midi_files, start=1):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            metrics = compute_music_metrics_from_midi(midi)
            row = {
                "file_index": index,
                "file_path": str(midi_path),
            }
            row.update({key: metrics.get(key) for key in METRIC_KEYS})
            per_file_rows.append(row)
        except Exception as exc:
            errors.append({"file_path": str(midi_path), "error": str(exc)})

    if not per_file_rows:
        raise RuntimeError("Aucune metrique n'a pu etre calculee (erreurs sur tous les fichiers).")

    summary = summarize_per_file(per_file_rows)

    payload = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "files_found": len(list_midi_files(args.input_dir)),
        "files_selected": len(midi_files),
        "files_processed": len(per_file_rows),
        "files_failed": len(errors),
        "seed": args.seed,
        "max_files": args.max_files,
        "metrics_summary": summary,
        "errors": errors,
    }

    save_json(output_paths["json_path"], payload)

    per_file_fieldnames = ["file_index", "file_path", *METRIC_KEYS]
    save_csv(output_paths["per_file_csv_path"], per_file_rows, per_file_fieldnames)

    summary_row = build_summary_csv_row(summary)
    summary_fieldnames = list(summary_row.keys())
    save_csv(output_paths["summary_csv_path"], [summary_row], summary_fieldnames)

    print(f"Run directory: {output_paths['run_dir']}")
    print(f"JSON report saved to {output_paths['json_path']}")
    print(f"Summary CSV saved to {output_paths['summary_csv_path']}")
    print(f"Per-file CSV saved to {output_paths['per_file_csv_path']}")
    if errors:
        print(f"Warnings: {len(errors)} fichier(s) ignores suite a des erreurs de lecture.")


if __name__ == "__main__":
    main()
