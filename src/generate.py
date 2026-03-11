import torch
import pretty_midi
from model import MusicLSTM
from dataset import load_dataloaders
from tokenizer import build_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_STEP = 0.125

def generate_lstm(model, start_token_id, id_to_token, max_tokens=200, temperature=1.0, device=device):
    model.eval()

    generated = [start_token_id]
    hidden = None

    x = torch.tensor([[start_token_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if id_to_token[next_token] == "END":
                break

            x = torch.tensor([[next_token]], dtype=torch.long).to(device)

    return generated

def tokens_to_midi(tokens, output_path, velocity=100, program=0):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    current_time = 0.0
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "START":
            i += 1
            continue
        
        if token == "END":
            break

        if token.startswith("REST_"):
            try:
                rest_steps = int(token.split("_")[1])
                current_time += rest_steps * TIME_STEP
            except Exception:
                pass
            i += 1
            continue
        
        if token.startswith("NOTE_"):
            try:
                pitch = int(token.split("_")[1])

            except Exception:
                i += 1
                continue
            
            if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                try:
                    dur_steps = int(tokens[i + 1].split("_")[1])
                    dur = dur_steps * TIME_STEP
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=current_time, end=current_time + dur)
                    instrument.notes.append(note)
                    current_time += dur
                    i += 2
                    continue
                
                except Exception:
                    i += 2
                    continue

            i += 1
            continue

        i += 1

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"Generated MIDI saved to {output_path}")


if __name__ == "__main__":
    vocab, token_to_id, id_to_token = build_vocab()
    vocab_size = len(vocab)
    model = MusicLSTM(vocab_size).to(device)
    model.load_state_dict(torch.load("models/lstm_final.pt", map_location=device))
    start_token_id = token_to_id["START"]
    generated_ids = generate_lstm(model, start_token_id, id_to_token, max_tokens=200, temperature=0.8)
    generated_tokens = [id_to_token[token_id] for token_id in generated_ids]
    print(generated_tokens)
    tokens_to_midi(generated_tokens, "outputs/generated_music.mid")