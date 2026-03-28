import io
import wave
from pathlib import Path

import numpy as np
import torch
import pretty_midi
from emotion_utils import EMOPIA_START_TOKENS, is_emopia_vocab
from models import MusicLSTM, build_music_lstm, build_music_transformer
from tokenizer import build_vocab, build_vocab_emopia, build_vocab_polyphonic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_STEP = 0.05
DEFAULT_AUDIO_SAMPLE_RATE = 44100


def find_available_soundfonts():
    candidate_roots = [
        Path.home() / ".soundfont.sf2",
        Path.home() / ".soundfonts",
        Path(__file__).resolve().parents[1] / "assets" / "soundfonts",
    ]

    pretty_midi_dir = Path(pretty_midi.__file__).resolve().parent
    candidate_roots.append(pretty_midi_dir)

    soundfonts = []
    seen = set()
    for root in candidate_roots:
        if root.is_file() and root.suffix.lower() in {".sf2", ".sf3"}:
            resolved = root.resolve()
            if resolved not in seen:
                soundfonts.append(root)
                seen.add(resolved)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".sf2", ".sf3"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            soundfonts.append(path)
            seen.add(resolved)
    return soundfonts

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

def get_emopia_emotion_id(start_token, token_to_id):
    if start_token not in EMOPIA_START_TOKENS:
        raise ValueError(f"Token EMOPIA invalide pour l'emotion globale: {start_token}")
    if start_token not in token_to_id:
        raise ValueError(f"Token EMOPIA introuvable dans le vocabulaire: {start_token}")
    return EMOPIA_START_TOKENS[start_token]


def generate_transformer(
    model,
    start_token_id,
    id_to_token,
    max_tokens=200,
    temperature=0.8,
    top_k=10,
    emotion_id=None,
    device=device,
):
    model.eval()
    generated = [start_token_id]
    emotion = None
    if emotion_id is not None:
        emotion = torch.tensor([emotion_id], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        x = torch.tensor([generated], dtype=torch.long, device=device)

        if x.size(1) > model.max_len:
            x = x[:, -model.max_len:]

        logits = model(x, emotion=emotion)
        logits = logits[:, -1, :] / temperature


        if top_k is not None and top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            min_topk = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if id_to_token[next_token] == "END":
            break

    return generated


def tokens_to_pretty_midi(tokens, velocity=100, program=0):
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

    return midi


def tokens_to_pretty_midi_polyphonic(tokens, velocity=100, program=0):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    current_time = 0.0
    active_notes = {}

    for token in tokens:
        if token == "START" or token.startswith("START_"):
            continue
        
        if token == "END":
            break

        if token.startswith("SHIFT_"):
            try:
                shift_steps = int(token.split("_")[1])
                current_time += shift_steps * TIME_STEP
            except Exception:
                pass
            continue

        if token.startswith("NOTE_ON_"):
            try:
                pitch = int(token.split("_")[2])
                if pitch in active_notes:
                    start_time = active_notes[pitch]
                    end_time = max(current_time, start_time + TIME_STEP)
                    instrument.notes.append(
                        pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                    )
                active_notes[pitch] = current_time
            except Exception:
                continue
            continue

        if token.startswith("NOTE_OFF_"):
            try:
                pitch = int(token.split("_")[2])
                if pitch in active_notes:
                    start_time = active_notes.pop(pitch)
                    end_time = max(current_time, start_time + TIME_STEP)
                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                    instrument.notes.append(note)
            except Exception:
                continue 
            continue

    for pitch, start_time in active_notes.items():
        end_time = max(current_time, start_time + TIME_STEP)
        instrument.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        )
    
    midi.instruments.append(instrument)

    return midi


def tokens_to_pretty_midi_dispatch(tokens, tokenizer_mode="mono", velocity=100, program=0):
    if tokenizer_mode in {"poly", "emopia"}:
        return tokens_to_pretty_midi_polyphonic(tokens, velocity=velocity, program=program)
    return tokens_to_pretty_midi(tokens, velocity=velocity, program=program)


def tokens_to_midi(tokens, output_path, velocity=100, program=0, tokenizer_mode="mono"):
    midi = tokens_to_pretty_midi_dispatch(tokens, tokenizer_mode=tokenizer_mode, velocity=velocity, program=program)
    midi.write(output_path)
    print(f"Generated MIDI saved to {output_path}")


def find_soundfont_path(preferred_path=None):
    if preferred_path is not None:
        preferred = Path(preferred_path).expanduser()
        if preferred.exists():
            return preferred
    soundfonts = find_available_soundfonts()
    if soundfonts:
        return soundfonts[0]
    return None


def synthesize_piano_like_audio(midi, sample_rate=16000):
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)

    if not notes:
        return np.zeros(sample_rate, dtype=np.float32)

    end_time = max(note.end for note in notes) + 1.0
    total_samples = max(int(np.ceil(end_time * sample_rate)), sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    for note in notes:
        start_idx = max(int(note.start * sample_rate), 0)
        end_idx = min(int(note.end * sample_rate), total_samples)
        if end_idx <= start_idx:
            continue

        duration = (end_idx - start_idx) / sample_rate
        t = np.linspace(0, duration, end_idx - start_idx, endpoint=False, dtype=np.float32)
        frequency = float(pretty_midi.note_number_to_hz(note.pitch))
        amplitude = max(note.velocity / 127.0, 0.05)

        attack = min(0.01, duration)
        decay = min(0.25, max(duration - attack, 1e-4))
        attack_samples = max(int(attack * sample_rate), 1)
        decay_samples = max(int(decay * sample_rate), 1)

        envelope = np.ones_like(t, dtype=np.float32)
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False, dtype=np.float32)

        sustain_start = min(attack_samples, envelope.size)
        sustain_end = min(attack_samples + decay_samples, envelope.size)
        if sustain_end > sustain_start:
            decay_curve = np.exp(-3.5 * np.linspace(0.0, 1.0, sustain_end - sustain_start, dtype=np.float32))
            envelope[sustain_start:sustain_end] = decay_curve
        if sustain_end < envelope.size:
            tail = np.exp(-4.5 * np.linspace(0.0, 1.0, envelope.size - sustain_end, dtype=np.float32))
            envelope[sustain_end:] = tail * 0.35

        signal = (
            1.00 * np.sin(2 * np.pi * frequency * t)
            + 0.45 * np.sin(2 * np.pi * 2 * frequency * t + 0.10)
            + 0.20 * np.sin(2 * np.pi * 3 * frequency * t + 0.20)
            + 0.08 * np.sin(2 * np.pi * 4 * frequency * t + 0.35)
        )
        signal *= envelope * amplitude
        audio[start_idx:end_idx] += signal.astype(np.float32)

    delay_samples = int(0.06 * sample_rate)
    if delay_samples > 0 and delay_samples < audio.size:
        audio[delay_samples:] += 0.18 * audio[:-delay_samples]

    return audio


def render_midi_audio(midi, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE, soundfont_path=None):
    soundfont_path = find_soundfont_path(preferred_path=soundfont_path)

    if soundfont_path is not None:
        try:
            import fluidsynth  # noqa: F401

            audio = midi.fluidsynth(fs=sample_rate, synthesizer=str(soundfont_path))
            return np.asarray(audio, dtype=np.float32), f"FluidSynth ({soundfont_path.name})"
        except Exception:
            pass

    return synthesize_piano_like_audio(midi, sample_rate=sample_rate), "Piano-like fallback"


def midi_to_wav_bytes(midi, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE, return_renderer=False, soundfont_path=None):
    audio, renderer_name = render_midi_audio(
        midi,
        sample_rate=sample_rate,
        soundfont_path=soundfont_path,
    )
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        audio = np.zeros(sample_rate, dtype=np.float32)

    max_value = np.max(np.abs(audio))
    if max_value > 0:
        audio = audio / max_value

    pcm_audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())

    wav_bytes = buffer.getvalue()
    if return_renderer:
        return wav_bytes, renderer_name
    return wav_bytes


def tokens_to_wav_bytes(
    tokens,
    velocity=100,
    program=0,
    sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
    return_renderer=False,
    tokenizer_mode="mono",
    soundfont_path=None,
):
    midi = tokens_to_pretty_midi_dispatch(tokens, tokenizer_mode=tokenizer_mode, velocity=velocity, program=program)
    return midi_to_wav_bytes(
        midi,
        sample_rate=sample_rate,
        return_renderer=return_renderer,
        soundfont_path=soundfont_path,
    )


def load_generation_model(model_name, checkpoint_path, tokenizer_mode="mono", device=device):
    if tokenizer_mode == "poly":
        vocab, token_to_id, id_to_token = build_vocab_polyphonic()
    elif tokenizer_mode == "emopia":
        vocab, token_to_id, id_to_token = build_vocab_emopia()
    else:
        vocab, token_to_id, id_to_token = build_vocab()
    vocab_size = len(vocab)

    checkpoint_state = torch.load(checkpoint_path, map_location=device)

    if model_name == "lstm":
        model = build_music_lstm(vocab_size)
    elif model_name in ["transformer", "transformer_giantmidi"]:
        emotion_mode = (
            tokenizer_mode == "emopia"
            or "emotion_embedding.weight" in checkpoint_state
            or is_emopia_vocab(token_to_id)
        )
        model = build_music_transformer(vocab_size, emotion_mode=emotion_mode)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(checkpoint_state)
    model.to(device)
    return model, token_to_id, id_to_token


def generate_tokens(
    model_name,
    checkpoint_path,
    max_tokens=200,
    temperature=0.8,
    top_k=10,
    tokenizer_mode="mono",
    start_token=None,
    device=device,
):
    model, token_to_id, id_to_token = load_generation_model(
        model_name,
        checkpoint_path,
        tokenizer_mode=tokenizer_mode,
        device=device,
    )
    if start_token is None:
        start_token = "START_HAPPY" if tokenizer_mode == "emopia" else "START"
    if start_token not in token_to_id:
        raise ValueError(f"Token de depart introuvable dans le vocabulaire: {start_token}")
    start_token_id = token_to_id[start_token]
    emotion_id = None
    if tokenizer_mode == "emopia" and model_name in ["transformer", "transformer_giantmidi"]:
        emotion_id = get_emopia_emotion_id(start_token, token_to_id)

    if model_name == "lstm":
        generated_ids = generate_lstm(
            model,
            start_token_id,
            id_to_token,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
    else:
        generated_ids = generate_transformer(
            model,
            start_token_id,
            id_to_token,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            emotion_id=emotion_id,
            device=device,
        )

    return [id_to_token[token_id] for token_id in generated_ids]


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
