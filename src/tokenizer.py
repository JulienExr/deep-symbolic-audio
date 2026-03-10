from midi_utils import load_mono_note
import tqdm
import os
import torch
import json

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

def teokenize_all_midis(input_dir):
    all_tokens = []
    for root, _, files in os.walk(input_dir):
        for file in tqdm.tqdm(files):
            if file.endswith(".midi"):
                input_path = os.path.join(root, file)
                mono_note = load_mono_note(input_path)
                tokens = notes_to_tokens(mono_note)
                if tokens is not None:
                    all_tokens.append(tokens)
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

def encode_tokens(tokens, token_to_id):
    return [token_to_id[token] for token in tokens]

def encode_tokens_list(tokens_list, token_to_id):
    return [encode_tokens(tokens, token_to_id) for tokens in tokens_list]

def make_training_examples(encoded_sequences, seq_length=128):
    inputs = []
    targets = []
    for seq in encoded_sequences:
        for i in range(len(seq) - seq_length):
            inputs.append(seq[i:i+seq_length])
            targets.append(seq[i+1:i+seq_length+1])
    return inputs, targets

def save_dataset(inputs, targets, output_path):
    torch.save({
        "inputs": torch.tensor(inputs, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }, output_path)
    print(f"Dataset saved to {output_path}")

def save_vocab(token_to_id, id_to_token, output_path):
    with open(output_path + "_token_to_id.json", "w") as f:
        json.dump(token_to_id, f, indent=2)
    print(f"Token to ID mapping saved to {output_path}_token_to_id.json")

    with open(output_path + "_id_to_token.json", "w") as f:
        json.dump(id_to_token, f, indent=2)
    print(f"ID to Token mapping saved to {output_path}_id_to_token.json")

if __name__ == "__main__":
    all_tokens = teokenize_all_midis("data/test_mono")
    vocab, token_to_id, id_to_token = build_vocab()
    encoded = encode_tokens_list(all_tokens, token_to_id)
    inputs, targets = make_training_examples(encoded)
    save_dataset(inputs, targets, "data/processed/dataset.pt")
    save_vocab(token_to_id, id_to_token, "data/processed/vocab")