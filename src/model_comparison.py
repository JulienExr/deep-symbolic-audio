import argparse
import json
import random
import re
import statistics
from itertools import product
from datetime import datetime
from pathlib import Path

import torch

from generate import (
    generate_lstm,
    generate_transformer,
    get_emopia_emotion_id,
    load_generation_model,
    tokens_to_pretty_midi_dispatch,
)
from metrics import build_generation_report, count_model_parameters, save_csv, save_json
from model_analysis_plots import PLOT_METRICS, generate_plots
from music_analysis import compute_music_metrics


ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_METRICS = [
    ("generation.token_count", "Token count"),
    ("generation.unique_token_ratio", "Unique token ratio"),
    ("generation.invalid_token_ratio", "Invalid token ratio"),
    *PLOT_METRICS,
]
SUMMARY_METRIC_NAMES = {metric_name for metric_name, _ in SUMMARY_METRICS}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyse un ou plusieurs checkpoints et genere un rapport horodate automatiquement."
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint(s) .pt a analyser.")
    parser.add_argument("--model-name", choices=["lstm", "transformer", "transformer_giantmidi"], default=None)
    parser.add_argument("--tokenizer-mode", choices=["mono", "poly", "emopia"], default=None)
    parser.add_argument("--device", default=None, help="cpu ou cuda")
    parser.add_argument("--seed", type=int, default=0, help="Seed de base pour la generation")
    parser.add_argument("--num-samples", type=int, default=4, help="Nombre d'echantillons generes par modele")
    parser.add_argument("--max-tokens", type=int, default=512, help="Longueur maximale d'un echantillon")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature de generation")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k pour le Transformer")
    parser.add_argument(
        "--temperature-values",
        nargs="+",
        default=None,
        help="Liste de temperatures a comparer (ex: --temperature-values 0.7 0.9 1.1 ou 0.7,0.9,1.1)",
    )
    parser.add_argument(
        "--top-k-values",
        nargs="+",
        default=None,
        help="Liste de top-k a comparer (ex: --top-k-values 10 20 50 ou 10,20,50)",
    )
    parser.add_argument(
        "--save-generated-midis",
        action="store_true",
        help="Sauvegarde aussi les MIDIs generes dans le dossier du run",
    )
    return parser


def resolve_device(device_name):
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def resolve_checkpoints(raw_paths):
    checkpoints = []
    seen = set()

    for raw_path in raw_paths:
        checkpoint_path = resolve_path(raw_path)
        resolved_path = checkpoint_path.resolve()
        if resolved_path in seen:
            continue
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
        seen.add(resolved_path)
        checkpoints.append(checkpoint_path)

    return checkpoints


def infer_tokenizer_mode(checkpoint_path, explicit_mode=None):
    if explicit_mode is not None:
        return explicit_mode

    checkpoint_name = checkpoint_path.name.lower()
    if "emopia" in checkpoint_name or "emotion" in checkpoint_name:
        return "emopia"
    if "_poly_" in checkpoint_name or checkpoint_name.endswith("_poly.pt"):
        return "poly"
    if "_mono_" in checkpoint_name or checkpoint_name.endswith("_mono.pt"):
        return "mono"

    raise ValueError(
        f"Impossible d'inferer le tokenizer pour {checkpoint_path.name}. Passe --tokenizer-mode explicitement."
    )


def infer_model_name(checkpoint_path, explicit_name=None):
    if explicit_name is not None:
        return explicit_name

    checkpoint_name = checkpoint_path.name.lower()
    if "transformer_giantmidi" in checkpoint_name:
        return "transformer_giantmidi"
    if "lstm" in checkpoint_name:
        return "lstm"

    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if "head.weight" in checkpoint_state:
        inferred_name = "transformer"
    elif "fc.weight" in checkpoint_state:
        inferred_name = "lstm"
    else:
        raise ValueError("Impossible de deduire l'architecture a partir du checkpoint.")

    if inferred_name == "transformer" and "giantmidi" in checkpoint_name:
        return "transformer_giantmidi"
    return inferred_name


def resolve_start_token(tokenizer_mode, token_to_id):
    if tokenizer_mode == "emopia":
        for candidate in ("START_HAPPY", "START_SAD", "START_ANGRY", "START_RELAXED"):
            if candidate in token_to_id:
                return candidate
        raise ValueError("Aucun token START_* EMOPIA trouve dans le vocabulaire.")

    if "START" not in token_to_id:
        raise ValueError("Le vocabulaire ne contient pas le token START.")
    return "START"


def _parse_list_values(raw_values, cast_type, option_name):
    if raw_values is None:
        return None

    parsed = []
    for raw_value in raw_values:
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed.append(cast_type(item))
            except ValueError as exc:
                raise ValueError(f"Valeur invalide pour {option_name}: {item}") from exc

    if not parsed:
        raise ValueError(f"Aucune valeur valide fournie pour {option_name}.")

    return parsed


def build_sampling_configs(args):
    temperatures = _parse_list_values(args.temperature_values, float, "--temperature-values")
    top_ks = _parse_list_values(args.top_k_values, int, "--top-k-values")

    if temperatures is None:
        temperatures = [args.temperature]
    if top_ks is None:
        top_ks = [args.top_k]

    configs = []
    for temperature, top_k in product(temperatures, top_ks):
        config = {
            "temperature": float(temperature),
            "top_k": int(top_k),
        }
        config["label"] = f"temp={config['temperature']:.3g} | top_k={config['top_k']}"
        config["path_label"] = f"temp_{config['temperature']:.3g}_topk_{config['top_k']}"
        configs.append(config)

    return configs


def generate_tokens(model, model_name, tokenizer_mode, token_to_id, id_to_token, args, device, sampling_config):
    start_token = resolve_start_token(tokenizer_mode, token_to_id)
    start_token_id = token_to_id[start_token]
    emotion_id = None

    if tokenizer_mode == "emopia" and model_name in {"transformer", "transformer_giantmidi"}:
        emotion_id = get_emopia_emotion_id(start_token, token_to_id)

    if model_name == "lstm":
        generated_ids = generate_lstm(
            model,
            start_token_id,
            id_to_token,
            max_tokens=args.max_tokens,
            temperature=sampling_config["temperature"],
            device=device,
        )
    else:
        generated_ids = generate_transformer(
            model,
            start_token_id,
            id_to_token,
            max_tokens=args.max_tokens,
            temperature=sampling_config["temperature"],
            top_k=sampling_config["top_k"],
            emotion_id=emotion_id,
            device=device,
        )

    return start_token, [id_to_token[token_id] for token_id in generated_ids]


def extract_generation_metrics(tokens, tokenizer_mode):
    report = build_generation_report(tokens, tokenizer_mode=tokenizer_mode)
    return {
        "token_count": report["token_count"],
        "unique_token_ratio": report["unique_token_ratio"],
        "invalid_token_ratio": report["invalid_token_ratio"],
        "ended_with_end": report["ended_with_end"],
    }


def flatten_numeric(prefix, payload, out):
    for key, value in payload.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten_numeric(metric_name, value, out)
        elif isinstance(value, bool):
            out[metric_name] = float(value)
        elif isinstance(value, (int, float)):
            out[metric_name] = float(value)


def summarize_metrics(sample_reports):
    metric_values = {metric_name: [] for metric_name in SUMMARY_METRIC_NAMES}

    for sample_report in sample_reports:
        flattened = {}
        flatten_numeric("generation", sample_report["generation_metrics"], flattened)
        flatten_numeric("music", sample_report["music_metrics"], flattened)
        for metric_name in SUMMARY_METRIC_NAMES:
            value = flattened.get(metric_name)
            if value is not None:
                metric_values[metric_name].append(value)

    summary = {}
    for metric_name, metric_label in SUMMARY_METRICS:
        values = metric_values[metric_name]
        if not values:
            continue
        summary[metric_name] = {
            "label": metric_label,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def parse_checkpoint_epoch(checkpoint_path):
    match = re.search(r"_epoch_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else None


def resolve_metrics_report_path(checkpoint_path):
    stem = re.sub(r"_final$|_epoch_\d+$", "", checkpoint_path.stem)
    report_path = checkpoint_path.parent / f"{stem}_metrics.json"
    return report_path if report_path.exists() else None


def load_training_context(checkpoint_path):
    report_path = resolve_metrics_report_path(checkpoint_path)
    context = {
        "checkpoint_epoch": parse_checkpoint_epoch(checkpoint_path),
        "training_report_path": str(report_path) if report_path is not None else None,
        "best_epoch": None,
    }

    if report_path is None:
        return context

    training_report = json.loads(report_path.read_text())
    context["best_epoch"] = training_report.get("best_epoch")
    return context


def build_run_paths():
    run_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT_DIR / "outputs" / "model_analysis" / run_name
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "json_path": run_dir / "report.json",
        "csv_path": run_dir / "summary.csv",
        "plot_dir": run_dir / "plots",
        "midi_dir": run_dir / "generated_midis",
    }


def save_generated_midi(output_dir, checkpoint_path, sample_index, tokens, tokenizer_mode, sampling_config):
    model_output_dir = output_dir / checkpoint_path.stem / sampling_config["path_label"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / f"sample_{sample_index:02d}.mid"
    midi = tokens_to_pretty_midi_dispatch(tokens, tokenizer_mode=tokenizer_mode)
    midi.write(str(output_path))
    return output_path


def analyze_checkpoint(checkpoint_path, model_name, tokenizer_mode, args, device, sampling_config, midi_output_dir=None):
    model, token_to_id, id_to_token = load_generation_model(
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        tokenizer_mode=tokenizer_mode,
        device=device,
    )
    parameter_stats = count_model_parameters(model)

    sample_reports = []
    start_token = None

    for sample_index in range(args.num_samples):
        sample_seed = args.seed + sample_index
        set_seed(sample_seed)
        start_token, tokens = generate_tokens(
            model=model,
            model_name=model_name,
            tokenizer_mode=tokenizer_mode,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            args=args,
            device=device,
            sampling_config=sampling_config,
        )

        sample_report = {
            "sample_index": sample_index,
            "seed": sample_seed,
            "generation_metrics": extract_generation_metrics(tokens, tokenizer_mode),
            "music_metrics": compute_music_metrics(tokens, tokenizer_mode=tokenizer_mode),
        }
        if midi_output_dir is not None:
            sample_report["generated_midi_path"] = str(
                save_generated_midi(
                    midi_output_dir,
                    checkpoint_path,
                    sample_index,
                    tokens,
                    tokenizer_mode,
                    sampling_config,
                )
            )
        sample_reports.append(sample_report)

    return {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_display_name": f"{checkpoint_path.stem} [{sampling_config['label']}]",
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "tokenizer_mode": tokenizer_mode,
        "device": str(device),
        "sampling": {
            "temperature": sampling_config["temperature"],
            "top_k": sampling_config["top_k"],
            "label": sampling_config["label"],
        },
        "start_token": start_token,
        "parameter_stats": parameter_stats,
        "training_context": load_training_context(checkpoint_path),
        "aggregate_metrics": summarize_metrics(sample_reports),
        "samples": sample_reports,
    }


def build_summary_rows(checkpoint_reports):
    rows = []

    for report in checkpoint_reports:
        row = {
            "checkpoint_name": report["checkpoint_name"],
            "checkpoint_display_name": report.get("checkpoint_display_name"),
            "checkpoint_path": report["checkpoint_path"],
            "model_name": report["model_name"],
            "tokenizer_mode": report["tokenizer_mode"],
            "sampling.temperature": report.get("sampling", {}).get("temperature"),
            "sampling.top_k": report.get("sampling", {}).get("top_k"),
            "sampling.label": report.get("sampling", {}).get("label"),
            "start_token": report["start_token"],
            "total_parameters": report["parameter_stats"].get("total_parameters"),
            "trainable_parameters": report["parameter_stats"].get("trainable_parameters"),
            "training.checkpoint_epoch": report["training_context"].get("checkpoint_epoch"),
            "training.best_epoch": report["training_context"].get("best_epoch"),
            "training.training_report_path": report["training_context"].get("training_report_path"),
        }

        for metric_name, _ in SUMMARY_METRICS:
            stats = report["aggregate_metrics"].get(metric_name)
            row[f"{metric_name}.mean"] = stats["mean"] if stats else None
            row[f"{metric_name}.std"] = stats["std"] if stats else None

        rows.append(row)

    return rows


def build_summary_fieldnames():
    fieldnames = [
        "checkpoint_name",
        "checkpoint_display_name",
        "checkpoint_path",
        "model_name",
        "tokenizer_mode",
        "sampling.temperature",
        "sampling.top_k",
        "sampling.label",
        "start_token",
        "total_parameters",
        "trainable_parameters",
        "training.checkpoint_epoch",
        "training.best_epoch",
        "training.training_report_path",
    ]
    for metric_name, _ in SUMMARY_METRICS:
        fieldnames.append(f"{metric_name}.mean")
        fieldnames.append(f"{metric_name}.std")
    return fieldnames


def write_reports(checkpoint_reports, plot_paths, run_paths, args):
    sampling_configs = build_sampling_configs(args)
    payload = {
        "run_name": run_paths["run_name"],
        "run_directory": str(run_paths["run_dir"]),
        "config": {
            "checkpoints": [str(resolve_path(path)) for path in args.checkpoints],
            "model_name": args.model_name,
            "tokenizer_mode": args.tokenizer_mode,
            "device": args.device,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "temperature_values": args.temperature_values,
            "top_k_values": args.top_k_values,
            "sampling_grid": sampling_configs,
            "save_generated_midis": args.save_generated_midis,
        },
        "plots": [str(path) for path in plot_paths],
        "checkpoints": checkpoint_reports,
    }
    save_json(run_paths["json_path"], payload)
    save_csv(run_paths["csv_path"], build_summary_rows(checkpoint_reports), build_summary_fieldnames())


def main():
    args = build_parser().parse_args()
    checkpoints = resolve_checkpoints(args.checkpoints)
    sampling_configs = build_sampling_configs(args)
    device = resolve_device(args.device)
    run_paths = build_run_paths()

    midi_output_dir = None
    if args.save_generated_midis:
        midi_output_dir = run_paths["midi_dir"]
        midi_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_reports = []
    for checkpoint_path in checkpoints:
        model_name = infer_model_name(checkpoint_path, explicit_name=args.model_name)
        tokenizer_mode = infer_tokenizer_mode(checkpoint_path, explicit_mode=args.tokenizer_mode)
        for sampling_config in sampling_configs:
            checkpoint_reports.append(
                analyze_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    args=args,
                    device=device,
                    sampling_config=sampling_config,
                    midi_output_dir=midi_output_dir,
                )
            )

    plot_paths = generate_plots(checkpoint_reports, run_paths["plot_dir"])
    write_reports(checkpoint_reports, plot_paths, run_paths, args)

    print(f"Run directory: {run_paths['run_dir']}")
    print(f"JSON report saved to {run_paths['json_path']}")
    print(f"CSV summary saved to {run_paths['csv_path']}")
    print(f"Plots saved to {run_paths['plot_dir']}")


if __name__ == "__main__":
    main()
