from midi_utils import load_mono_note, load_polyphonic_notes
from emotion_utils import EMOPIA_EMOTIONS, EMOPIA_START_TOKENS
import tqdm
import os
import torch
import json
import random

TIME_STEP = 0.125
MAX_DUR = 16
MAX_REST = 16
MIN_PITCH = 48
MAX_PITCH = 84

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
    return sorted(events, key=lambda x: (x[0], 0 if x[1] == "NOTE_ON" else 1))

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

def tokenize_all_midis(input_dir, max_files=None):
    all_tokens = []
    for root, _, files in os.walk(input_dir):
        files = sorted(files, key=lambda x: torch.randperm(len(files)).tolist())
        for file in tqdm.tqdm(files):
            if file.endswith(".midi") or file.endswith(".mid"):
                input_path = os.path.join(root, file)
                mono_note = load_mono_note(input_path)
                tokens = notes_to_tokens(mono_note)
                if tokens is not None:
                    all_tokens.append(tokens)
            if max_files is not None and len(all_tokens) >= max_files:
                break
    return all_tokens

def tokenize_all_midis_polyphonic(input_dir, max_files=None, emopia_mode=False):
    all_tokens = []
    for root, _, files in os.walk(input_dir):
        files = sorted(files, key=lambda x: torch.randperm(len(files)).tolist())
        for file in tqdm.tqdm(files):
            if file.endswith(".midi") or file.endswith(".mid"):
                input_path = os.path.join(root, file)
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
    dataset = {
        "inputs": torch.tensor(inputs, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }
    if emotions is not None:
        dataset["emotions"] = torch.tensor(emotions, dtype=torch.long)
    torch.save(dataset, output_path)
    print(f"Dataset saved to {output_path}")

def save_vocab(token_to_id, id_to_token, output_path):
    with open(output_path + "_token_to_id.json", "w") as f:
        json.dump(token_to_id, f, indent=2)
    print(f"Token to ID mapping saved to {output_path}_token_to_id.json")

    with open(output_path + "_id_to_token.json", "w") as f:
        json.dump(id_to_token, f, indent=2)
    print(f"ID to Token mapping saved to {output_path}_id_to_token.json")

def create_vocab_and_dataset(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1):
    all_tokens = tokenize_all_midis(input_dir, max_files=max_files)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    encoded = encode_tokens_list(all_tokens, token_to_id)
    print("Encoded tokens into integer IDs")
    inputs, targets = make_training_examples(encoded, seq_length=seq_length, stride=stride)
    print(f"Created {len(inputs)} training examples")
    save_dataset(inputs, targets, dataset_output_path)
    save_vocab(token_to_id, id_to_token, vocab_output_path)

def create_vocab_and_dataset_polyphonic(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1):
    all_tokens = tokenize_all_midis_polyphonic(input_dir, max_files=max_files)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab_polyphonic()
    print(f"Vocabulary size: {len(vocab)}")
    random.shuffle(all_tokens)
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

def create_vocab_and_dataset_emopia(input_dir, dataset_output_path, vocab_output_path, max_files=None, seq_length=128, stride=1):
    all_tokens = tokenize_all_midis_polyphonic(input_dir, max_files=max_files, emopia_mode=True)
    print(f"Tokenized {len(all_tokens)} MIDI files")
    vocab, token_to_id, id_to_token = build_vocab_emopia()
    print(f"Vocabulary size: {len(vocab)}")
    emotion_token_id_to_label = {
        token_to_id[start_token]: label for start_token, label in EMOPIA_START_TOKENS.items()
    }
    random.shuffle(all_tokens)
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

if __name__ == "__main__":
    # create_vocab_and_dataset("data/midi_mono", "data/processed/dataset.pt", "data/processed/vocab", max_files=1000, seq_length=128, stride=8)
    # create_vocab_and_dataset("data/midi_mono", "data/processed/dataset_test.pt", "data/processed/vocab_test", max_files=250, seq_length=128, stride=8)
    # create_vocab_and_dataset_polyphonic("data/test_poly", "data/processed/dataset_poly_test.pt", "data/processed/vocab_poly", max_files=10, seq_length=128, stride=8)
    # create_vocab_and_dataset_polyphonic("data/midi_poly", "data/processed/dataset_poly.pt", "data/processed/vocab_poly", max_files= 1200, seq_length=256, stride=48)
    create_vocab_and_dataset_emopia("data/midi_emopia", "data/processed/dataset_emopia.pt", "data/processed/vocab_emopia", max_files=1200, seq_length=256, stride=48)
