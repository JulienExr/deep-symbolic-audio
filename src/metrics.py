import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)


def save_csv(output_path, rows, fieldnames):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_perplexity(loss):
    if loss is None:
        return None
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def count_model_parameters(model):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
    }


def build_training_history(train_losses, val_losses=None):
    history = []
    val_losses = val_losses or []

    for epoch_index, train_loss in enumerate(train_losses, start=1):
        val_loss = val_losses[epoch_index - 1] if epoch_index <= len(val_losses) else None
        history.append(
            {
                "epoch": epoch_index,
                "train_loss": float(train_loss),
                "train_perplexity": safe_perplexity(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "val_perplexity": safe_perplexity(val_loss),
            }
        )

    return history


def build_training_report(
    model_name,
    tokenizer_mode,
    train_losses,
    val_losses=None,
    metadata=None,
    checkpoint_paths=None,
    final_model_path=None,
    parameter_stats=None,
):
    history = build_training_history(train_losses, val_losses=val_losses)
    train_losses = list(train_losses)
    val_losses = list(val_losses or [])

    reference_losses = val_losses if val_losses else train_losses
    best_epoch = None
    if reference_losses:
        best_epoch = min(range(len(reference_losses)), key=lambda index: reference_losses[index]) + 1

    best_val_loss = min(val_losses) if val_losses else None
    best_train_loss = min(train_losses) if train_losses else None
    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None

    return {
        "timestamp_utc": utc_timestamp(),
        "model_name": model_name,
        "tokenizer_mode": tokenizer_mode,
        "epochs": len(train_losses),
        "best_epoch": best_epoch,
        "best_metric": "val_loss" if val_losses else "train_loss",
        "best_train_loss": float(best_train_loss) if best_train_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "final_train_perplexity": safe_perplexity(final_train_loss),
        "final_val_perplexity": safe_perplexity(final_val_loss),
        "parameter_stats": parameter_stats or {},
        "checkpoint_paths": checkpoint_paths or [],
        "final_model_path": str(final_model_path) if final_model_path is not None else None,
        "metadata": metadata or {},
        "history": history,
    }


def write_training_report(json_path, csv_path, report):
    save_json(json_path, report)
    history = report.get("history", [])
    save_csv(
        csv_path,
        history,
        fieldnames=["epoch", "train_loss", "train_perplexity", "val_loss", "val_perplexity"],
    )


def is_note_token(token):
    return token.startswith("NOTE_")


def is_duration_token(token):
    return token.startswith("DUR_")


def is_rest_token(token):
    return token.startswith("REST_")


def is_shift_token(token):
    return token.startswith("SHIFT_")


def is_note_on_token(token):
    return token.startswith("NOTE_ON_")


def is_note_off_token(token):
    return token.startswith("NOTE_OFF_")


def parse_pitch(token):
    try:
        return int(token.split("_")[-1])
    except Exception:
        return None


def analyze_mono_tokens(tokens):
    invalid_token_count = 0
    note_count = 0
    rest_count = 0
    dangling_note_count = 0
    end_index = None

    if not tokens:
        return {
            "invalid_token_count": 0,
            "note_count": 0,
            "rest_count": 0,
            "dangling_note_count": 0,
            "trailing_token_count": 0,
        }

    if tokens[0] != "START":
        invalid_token_count += 1

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "END":
            end_index = index
            break
        if is_rest_token(token):
            rest_count += 1
            index += 1
            continue
        if is_note_token(token):
            note_count += 1
            if index + 1 < len(tokens) and is_duration_token(tokens[index + 1]):
                index += 2
                continue
            dangling_note_count += 1
            invalid_token_count += 1
            index += 1
            continue

        invalid_token_count += 1
        index += 1

    trailing_token_count = 0
    if end_index is not None and end_index < len(tokens) - 1:
        trailing_token_count = len(tokens) - end_index - 1
        invalid_token_count += trailing_token_count

    return {
        "invalid_token_count": invalid_token_count,
        "note_count": note_count,
        "rest_count": rest_count,
        "dangling_note_count": dangling_note_count,
        "trailing_token_count": trailing_token_count,
    }


def analyze_polyphonic_tokens(tokens):
    invalid_token_count = 0
    note_on_count = 0
    note_off_count = 0
    shift_count = 0
    duplicate_note_on_count = 0
    unmatched_note_off_count = 0
    end_index = None
    active_pitches = set()

    if not tokens:
        return {
            "invalid_token_count": 0,
            "note_on_count": 0,
            "note_off_count": 0,
            "shift_count": 0,
            "duplicate_note_on_count": 0,
            "unmatched_note_off_count": 0,
            "unfinished_active_notes": 0,
            "trailing_token_count": 0,
        }

    if tokens[0] != "START" and not tokens[0].startswith("START_"):
        invalid_token_count += 1

    for index, token in enumerate(tokens[1:], start=1):
        if token == "END":
            end_index = index
            break
        if is_shift_token(token):
            shift_count += 1
            continue
        if is_note_on_token(token):
            note_on_count += 1
            pitch = parse_pitch(token)
            if pitch in active_pitches:
                duplicate_note_on_count += 1
                invalid_token_count += 1
            if pitch is not None:
                active_pitches.add(pitch)
            else:
                invalid_token_count += 1
            continue
        if is_note_off_token(token):
            note_off_count += 1
            pitch = parse_pitch(token)
            if pitch not in active_pitches:
                unmatched_note_off_count += 1
                invalid_token_count += 1
            else:
                active_pitches.remove(pitch)
            continue

        invalid_token_count += 1

    trailing_token_count = 0
    if end_index is not None and end_index < len(tokens) - 1:
        trailing_token_count = len(tokens) - end_index - 1
        invalid_token_count += trailing_token_count

    return {
        "invalid_token_count": invalid_token_count,
        "note_on_count": note_on_count,
        "note_off_count": note_off_count,
        "shift_count": shift_count,
        "duplicate_note_on_count": duplicate_note_on_count,
        "unmatched_note_off_count": unmatched_note_off_count,
        "unfinished_active_notes": len(active_pitches),
        "trailing_token_count": trailing_token_count,
    }


def build_generation_report(tokens, tokenizer_mode, metadata=None):
    if tokenizer_mode == "mono":
        mode_metrics = analyze_mono_tokens(tokens)
    else:
        mode_metrics = analyze_polyphonic_tokens(tokens)

    token_count = len(tokens)
    unique_token_count = len(set(tokens))
    invalid_token_count = mode_metrics.get("invalid_token_count", 0)

    return {
        "timestamp_utc": utc_timestamp(),
        "tokenizer_mode": tokenizer_mode,
        "token_count": token_count,
        "unique_token_count": unique_token_count,
        "unique_token_ratio": unique_token_count / token_count if token_count else 0.0,
        "start_token": tokens[0] if tokens else None,
        "ended_with_end": bool(tokens) and tokens[-1] == "END",
        "invalid_token_count": invalid_token_count,
        "invalid_token_ratio": invalid_token_count / token_count if token_count else 0.0,
        "metadata": metadata or {},
        "mode_metrics": mode_metrics,
    }
