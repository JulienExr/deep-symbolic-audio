import argparse
import json
import os
import random
from pathlib import Path

import torch
import tqdm

from midi_utils import load_mono_note, load_polyphonic_notes
from emotion_utils import EMOPIA_EMOTIONS, EMOPIA_START_TOKENS

TIME_STEP = 0.05
MAX_DUR = 16
MAX_REST = 16
MIN_PITCH = 36
MAX_PITCH = 96
DEFAULT_PREPROCESS_CONFIGS = {
    "mono": {
        "input_dir": "data/midi_mono",
        "dataset_output_path": "data/processed/dataset.pt",
        "vocab_output_path": "data/processed/vocab",
        "seq_length": 128,
        "stride": 8,
    },
    "poly": {
        "input_dir": "data/midi_poly",
        "dataset_output_path": "data/processed/dataset_poly.pt",
        "vocab_output_path": "data/processed/vocab_poly",
        "seq_length": 256,
        "stride": 48,
    },
    "emopia": {
        "input_dir": "data/midi_emopia",
        "dataset_output_path": "data/processed/dataset_emopia.pt",
        "vocab_output_path": "data/processed/vocab_emopia",
        "seq_length": 256,
        "stride": 48,
    },
}


def _midi_file_paths(input_dir, seed=None):
    midi_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_paths.append(os.path.join(root, file))

    randomizer = random.Random(seed)
    randomizer.shuffle(midi_paths)
    return midi_paths


def get_preprocess_defaults(tokenizer_mode):
    if tokenizer_mode not in DEFAULT_PREPROCESS_CONFIGS:
        raise ValueError(f"Mode de tokenisation non supporte: {tokenizer_mode}")
    return DEFAULT_PREPROCESS_CONFIGS[tokenizer_mode].copy()

def notes_to_tokens(notes,
                    time_step=TIME_STEP,
                    min_pitch=MIN_PITCH,
                    max_pitch=MAX_PITCH,
                    max_dur=MAX_DUR,
                    max_rest=MAX_REST):
    if not notes:
        return None
    
    tokens = ["START"]
    current_time = 0.0

    for note in notes:
        pitch = note.pitch
        if pitch < min_pitch or pitch > max_pitch:
            continue
        rest = note.start - current_time
        rest_steps = int(round(rest / time_step))
        if rest_steps > 0:
            if tokens[-1] != "START":
                tokens.append(f"REST_{min(rest_steps, max_rest)}")

        dur = note.end - note.start
        dur_steps = int(round(dur / time_step))
        tokens.append(f"NOTE_{pitch}")
        tokens.append(f"DUR_{min(dur_steps, max_dur)}")

        current_time = note.end

    tokens.append("END")

    if len(tokens) < 20:
        return None

    return tokens

def quantize_time(time, time_step=TIME_STEP):
    return int(round(time / time_step))

def notes_to_events(notes, time_step=TIME_STEP, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH):
    events = []
    
    if not notes:
        return None
    
    for note in notes:
        pitch = note.pitch
        if pitch < min_pitch or pitch > max_pitch:
            continue
        
        start_step = quantize_time(note.start, time_step)
        end_step = quantize_time(note.end, time_step)
        
        if end_step <= start_step:
            continue

        events.append((start_step, "NOTE_ON", pitch))
        events.append((end_step, "NOTE_OFF", pitch))

    return events

def sort_events(events):
    # NOTE_OFF must come before NOTE_ON for the same pitch/time so re-attacks
    # are reconstructed as two notes instead of a 1-step truncated note.
    return sorted(events, key=lambda x: (x[0], x[2], 0 if x[1] == "NOTE_OFF" else 1))

def events_to_tokens_polyphonic(events,
                               time_step=TIME_STEP,
                               min_pitch=MIN_PITCH,
                               max_pitch=MAX_PITCH,
                               max_dur=MAX_DUR,
                               max_rest=MAX_REST,
                               emotion=None):
    if not events:
        return None
    if emotion is not None:
        tokens = ["START_" + emotion]
    else:
        tokens = ["START"]
    current_time = events[0][0]
    
    for start_step, event_type, pitch in events:
        delta_steps = start_step - current_time

        while delta_steps > 0:
            shift = min(delta_steps, max_dur)
            tokens.append(f"SHIFT_{shift}")
            delta_steps -= shift
            current_time += shift

        if event_type == "NOTE_ON":
            tokens.append(f"NOTE_ON_{pitch}")
        elif event_type == "NOTE_OFF":
            tokens.append(f"NOTE_OFF_{pitch}") 
    
    tokens.append("END")
    return tokens

def tokenize_all_midis(input_dir, max_files=None, seed=None):
    all_tokens = []
    midi_paths = _midi_file_paths(input_dir, seed=seed)
    for input_path in tqdm.tqdm(midi_paths, desc="Tokenizing MIDI files"):
        mono_note = load_mono_note(input_path)
        tokens = notes_to_tokens(mono_note)
        if tokens is not None:
            all_tokens.append(tokens)
        if max_files is not None and len(all_tokens) >= max_files:
            break
    return all_tokens

def tokenize_all_midis_polyphonic(input_dir, max_files=None, emopia_mode=False, seed=None):
    all_tokens = []
    midi_paths = _midi_file_paths(input_dir, seed=seed)
    for input_path in tqdm.tqdm(midi_paths, desc="Tokenizing MIDI files"):
        notes, emotion = load_polyphonic_notes(input_path, emopia_mode=emopia_mode)
        events = notes_to_events(notes)
        events_sorted = sort_events(events)
        tokens = events_to_tokens_polyphonic(events_sorted, emotion=emotion)
        if tokens is not None:
            all_tokens.append(tokens)
        if max_files is not None and len(all_tokens) >= max_files:
            break
    return all_tokens

def build_vocab():
    specials_tokens = ["PAD", "START", "END"]
    note_tokens = [f"NOTE_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
    dur_tokens = [f"DUR_{d}" for d in range(1, MAX_DUR + 1)]
    rest_tokens = [f"REST_{r}" for r in range(1, MAX_REST + 1)]

    vocab = specials_tokens + note_tokens + dur_tokens + rest_tokens
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return vocab, token_to_id, id_to_token

def build_vocab_emopia():
    emotion_tokens = [f"START_{emotion}" for emotion in EMOPIA_EMOTIONS]
    specials_tokens = ["PAD", "END"]
    note_on_tokens = [f"NOTE_ON_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
    note_off_tokens = [f"NOTE_OFF_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
    shift_tokens = [f"SHIFT_{s}" for s in range(1, MAX_DUR + 1)]

    vocab = emotion_tokens + specials_tokens + note_on_tokens + note_off_tokens + shift_tokens
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return vocab, token_to_id, id_to_token

def build_vocab_polyphonic():
    specials_tokens = ["PAD", "START", "END"]
    note_on_tokens = [f"NOTE_ON_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
    note_off_tokens = [f"NOTE_OFF_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
    shift_tokens = [f"SHIFT_{s}" for s in range(1, MAX_DUR + 1)]

    vocab = specials_tokens + note_on_tokens + note_off_tokens + shift_tokens
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return vocab, token_to_id, id_to_token

def encode_tokens(tokens, token_to_id):
    return [token_to_id[token] for token in tokens]

def encode_tokens_list(tokens_list, token_to_id):
    return [encode_tokens(tokens, token_to_id) for tokens in tokens_list]

def make_training_examples(encoded_sequences, seq_length=128, stride=1, emotion_token_id_to_label=None):
    inputs = []
    targets = []
    emotions = []
    for seq in tqdm.tqdm(encoded_sequences, desc="Creating training examples"):
        emotion = None
        if emotion_token_id_to_label is not None:
            emotion = emotion_token_id_to_label.get(seq[0])
            if emotion is None:
                raise ValueError(f"Token de debut EMOPIA invalide: {seq[0]}")
        for i in range(0, len(seq) - seq_length, stride):
            inputs.append(seq[i:i+seq_length])
            targets.append(seq[i+1:i+seq_length+1])
            if emotion_token_id_to_label is not None:
                emotions.append(emotion)
    if emotion_token_id_to_label is not None:
        return inputs, targets, emotions
    return inputs, targets

def save_dataset(inputs, targets, output_path, emotions=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "inputs": torch.tensor(inputs, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }
    if emotions is not None:
        dataset["emotions"] = torch.tensor(emotions, dtype=torch.long)
    torch.save(dataset, output_path)
    print(f"Dataset saved to {output_path}")

def save_vocab(token_to_id, id_to_token, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_path) + "_token_to_id.json", "w") as f:
        json.dump(token_to_id, f, indent=2)
    print(f"Token to ID mapping saved to {output_path}_token_to_id.json")

    with open(str(output_path) + "_id_to_token.json", "w") as f:
        json.dump(id_to_token, f, indent=2)
    print(f"ID to Token mapping saved to {output_path}_id_to_token.json")

def create_vocab_and_dataset(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1, seed=None):
    all_tokens = tokenize_all_midis(input_dir, max_files=max_files, seed=seed)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    encoded = encode_tokens_list(all_tokens, token_to_id)
    print("Encoded tokens into integer IDs")
    inputs, targets = make_training_examples(encoded, seq_length=seq_length, stride=stride)
    print(f"Created {len(inputs)} training examples")
    save_dataset(inputs, targets, dataset_output_path)
    save_vocab(token_to_id, id_to_token, vocab_output_path)

def create_vocab_and_dataset_polyphonic(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1, seed=None):
    all_tokens = tokenize_all_midis_polyphonic(input_dir, max_files=max_files, seed=seed)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab_polyphonic()
    print(f"Vocabulary size: {len(vocab)}")
    random.Random(seed).shuffle(all_tokens)
    val_split = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:val_split]
    val_tokens = all_tokens[val_split:]
    print(f"Training files: {len(train_tokens)}, Validation files: {len(val_tokens)}")
    train_encoded = encode_tokens_list(train_tokens, token_to_id)
    val_encoded = encode_tokens_list(val_tokens, token_to_id)
    print("Encoded tokens into integer IDs")
    train_inputs, train_targets = make_training_examples(train_encoded, seq_length=seq_length, stride=stride)
    val_inputs, val_targets = make_training_examples(val_encoded, seq_length=seq_length, stride=stride)
    print(f"Created {len(train_inputs)} training examples and {len(val_inputs)} validation examples")
    save_dataset(train_inputs, train_targets, dataset_output_path.replace(".pt", "_train.pt"))
    save_dataset(val_inputs, val_targets, dataset_output_path.replace(".pt", "_val.pt"))
    save_vocab(token_to_id, id_to_token, vocab_output_path)

def create_vocab_and_dataset_emopia(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1, seed=None):
    all_tokens = tokenize_all_midis_polyphonic(input_dir, max_files=max_files, emopia_mode=True, seed=seed)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab_emopia()
    print(f"Vocabulary size: {len(vocab)}")
    emotion_token_id_to_label = {
        token_to_id[start_token]: label for start_token, label in EMOPIA_START_TOKENS.items()
    }
    random.Random(seed).shuffle(all_tokens)
    val_split = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:val_split]
    val_tokens = all_tokens[val_split:]
    print(f"Training files: {len(train_tokens)}, Validation files: {len(val_tokens)}")
    train_encoded = encode_tokens_list(train_tokens, token_to_id)
    val_encoded = encode_tokens_list(val_tokens, token_to_id)
    print("Encoded tokens into integer IDs")
    train_inputs, train_targets, train_emotions = make_training_examples(
        train_encoded,
        seq_length=seq_length,
        stride=stride,
        emotion_token_id_to_label=emotion_token_id_to_label,
    )
    val_inputs, val_targets, val_emotions = make_training_examples(
        val_encoded,
        seq_length=seq_length,
        stride=stride,
        emotion_token_id_to_label=emotion_token_id_to_label,
    )
    print(f"Created {len(train_inputs)} training examples and {len(val_inputs)} validation examples")
    save_dataset(train_inputs, train_targets, dataset_output_path.replace(".pt", "_train.pt"), emotions=train_emotions)
    save_dataset(val_inputs, val_targets, dataset_output_path.replace(".pt", "_val.pt"), emotions=val_emotions)
    save_vocab(token_to_id, id_to_token, vocab_output_path)


def create_vocab_and_dataset_for_mode(
    tokenizer_mode,
    input_dir,
    dataset_output_path,
    vocab_output_path,
    max_files=None,
    seq_length=None,
    stride=None,
    seed=None,
):
    defaults = get_preprocess_defaults(tokenizer_mode)
    seq_length = defaults["seq_length"] if seq_length is None else seq_length
    stride = defaults["stride"] if stride is None else stride

    if tokenizer_mode == "mono":
        return create_vocab_and_dataset(
            input_dir=input_dir,
            dataset_output_path=dataset_output_path,
            vocab_output_path=vocab_output_path,
            max_files=max_files,
            seq_length=seq_length,
            stride=stride,
            seed=seed,
        )

    if tokenizer_mode == "poly":
        return create_vocab_and_dataset_polyphonic(
            input_dir=input_dir,
            dataset_output_path=dataset_output_path,
            vocab_output_path=vocab_output_path,
            max_files=max_files,
            seq_length=seq_length,
            stride=stride,
            seed=seed,
        )

    if tokenizer_mode == "emopia":
        return create_vocab_and_dataset_emopia(
            input_dir=input_dir,
            dataset_output_path=dataset_output_path,
            vocab_output_path=vocab_output_path,
            max_files=max_files,
            seq_length=seq_length,
            stride=stride,
            seed=seed,
        )

    raise ValueError(f"Mode de tokenisation non supporte: {tokenizer_mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build token vocabularies and training datasets from MIDI folders.")
    parser.add_argument("--tokenizer-mode", choices=["mono", "poly", "emopia"], default="emopia")
    parser.add_argument("--input-dir", default=None, help="Directory containing the source MIDI files.")
    parser.add_argument("--dataset-output", default=None, help="Output dataset path (.pt).")
    parser.add_argument("--vocab-output", default=None, help="Output vocab prefix path.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = get_preprocess_defaults(args.tokenizer_mode)
    create_vocab_and_dataset_for_mode(
        tokenizer_mode=args.tokenizer_mode,
        input_dir=args.input_dir or defaults["input_dir"],
        dataset_output_path=args.dataset_output or defaults["dataset_output_path"],
        vocab_output_path=args.vocab_output or defaults["vocab_output_path"],
        max_files=args.max_files,
        seq_length=args.seq_length,
        stride=args.stride,
        seed=args.seed,
    )
