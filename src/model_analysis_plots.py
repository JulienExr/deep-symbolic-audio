import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


PLOT_METRICS = [
    ("music.note_density_per_sec", "Densite de notes"),
    ("music.silence_ratio", "Ratio de silence"),
    ("music.mean_polyphony", "Polyphonie moyenne"),
    ("music.pitch_class_entropy", "Entropie harmonique"),
    ("music.tonal_center_strength", "Force tonale"),
    ("music.consonance_ratio", "Consonance"),
    ("music.harmonic_motif_repetition_ratio_4", "Rep. motif harm."),
    ("music.rhythmic_diversity_ratio", "Diversite rythmique"),
]


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "plot"


def checkpoint_label(report):
    custom_label = report.get("checkpoint_display_name")
    if custom_label:
        return custom_label

    checkpoint_epoch = report["training_context"].get("checkpoint_epoch")
    stem = Path(report["checkpoint_name"]).stem
    return f"{stem}\n(epoch {checkpoint_epoch})" if checkpoint_epoch is not None else stem


def humanize_identifier(value):
    cleaned = re.sub(r"_+", " ", value).strip()
    return cleaned if cleaned else value


def build_plot_title(metric_label, checkpoint_reports):
    prefix = "Analyse du modele" if len(checkpoint_reports) == 1 else "Comparaison des modeles"
    return f"{prefix} - {metric_label}"


def get_aggregate_metric(report, metric_name):
    metric = report["aggregate_metrics"].get(metric_name)
    if metric is None:
        return None, None
    return metric["mean"], metric["std"]


def format_metric_value(value):
    if value is None:
        return "-"
    absolute = abs(value)
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    if absolute >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def compute_mean_pitch_class_profile(sample_reports):
    profiles = [
        sample_report["music_metrics"]["pitch_class_profile"]
        for sample_report in sample_reports
        if sample_report["music_metrics"].get("pitch_class_profile") is not None
    ]
    if not profiles:
        return [0.0] * 12
    return [statistics.mean(profile[index] for profile in profiles) for index in range(12)]


def plot_metric_bars(checkpoint_reports, output_dir, metric_name, metric_label):
    labels = [checkpoint_label(report) for report in checkpoint_reports]
    means = []
    stds = []

    for report in checkpoint_reports:
        mean_value, std_value = get_aggregate_metric(report, metric_name)
        means.append(mean_value if mean_value is not None else 0.0)
        stds.append(std_value if std_value is not None else 0.0)

    fig_width = max(8, len(checkpoint_reports) * 2.2)
    fig, axis = plt.subplots(figsize=(fig_width, 5.5))
    bars = axis.bar(range(len(checkpoint_reports)), means, yerr=stds, color="#4472C4", alpha=0.88, capsize=4)

    axis.set_title(build_plot_title(metric_label, checkpoint_reports))
    axis.set_ylabel(metric_label)
    axis.set_xticks(range(len(checkpoint_reports)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            format_metric_value(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = output_dir / f"metric_{sanitize_filename(metric_name)}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_series(checkpoint_reports, output_dir):
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_metric_bars(checkpoint_reports, metric_dir, metric_name, metric_label)
        for metric_name, metric_label in PLOT_METRICS
    ]


def plot_pitch_class_profiles(checkpoint_reports, output_dir):
    fig, axis = plt.subplots(figsize=(14, 7))
    pitch_class_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    for report in checkpoint_reports:
        axis.plot(
            pitch_class_labels,
            compute_mean_pitch_class_profile(report["samples"]),
            marker="o",
            linewidth=2,
            label=checkpoint_label(report).replace("\n", " "),
        )

    axis.set_title(build_plot_title("Profil moyen des classes de pitch", checkpoint_reports))
    axis.set_ylabel("Poids relatif")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    fig.tight_layout()
    output_path = output_dir / "pitch_class_profiles.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def resolve_checkpoint_epoch_for_plot(report, training_report):
    checkpoint_epoch = report["training_context"].get("checkpoint_epoch")
    if checkpoint_epoch is not None:
        return checkpoint_epoch
    if report["checkpoint_name"].endswith("_final.pt"):
        return training_report.get("epochs")
    return None


def plot_training_histories(checkpoint_reports, output_dir):
    plot_paths = []
    grouped_reports = defaultdict(list)

    for report in checkpoint_reports:
        training_report_path = report["training_context"].get("training_report_path")
        if training_report_path:
            grouped_reports[training_report_path].append(report)

    for training_report_path, reports in grouped_reports.items():
        report_path = Path(training_report_path)
        if not report_path.exists():
            continue

        training_report = json.loads(report_path.read_text())
        history = training_report.get("history", [])
        if not history:
            continue

        epochs = [row["epoch"] for row in history]
        train_losses = [row.get("train_loss") for row in history]
        val_losses = [row.get("val_loss") for row in history]

        fig, axis = plt.subplots(figsize=(12, 6))
        axis.plot(epochs, train_losses, label="Train loss", linewidth=2)

        if any(value is not None for value in val_losses):
            filtered_epochs = [epoch for epoch, value in zip(epochs, val_losses) if value is not None]
            filtered_val_losses = [value for value in val_losses if value is not None]
            axis.plot(filtered_epochs, filtered_val_losses, label="Val loss", linewidth=2)

        for report in reports:
            checkpoint_epoch = resolve_checkpoint_epoch_for_plot(report, training_report)
            if checkpoint_epoch is None:
                continue
            row = next((item for item in history if item["epoch"] == checkpoint_epoch), None)
            if row is None:
                continue
            y_value = row.get("val_loss") if row.get("val_loss") is not None else row.get("train_loss")
            axis.scatter([checkpoint_epoch], [y_value], s=70, label=checkpoint_label(report).replace("\n", " "))

        title_stem = re.sub(r"_metrics$", "", report_path.stem)
        axis.set_title(f"Historique d'entrainement - {humanize_identifier(title_stem)}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

        fig.tight_layout()
        output_path = output_dir / f"training_history_{sanitize_filename(report_path.stem)}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(output_path)

    return plot_paths


def generate_plots(checkpoint_reports, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    plot_paths.extend(plot_metric_series(checkpoint_reports, output_dir))
    plot_paths.append(plot_pitch_class_profiles(checkpoint_reports, output_dir))
    plot_paths.extend(plot_training_histories(checkpoint_reports, output_dir))
    return plot_paths
