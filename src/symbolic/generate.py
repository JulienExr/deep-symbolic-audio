import io
import json
import wave
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from modeling.architectures import build_music_lstm, build_music_transformer
from symbolic.tokenizer import build_vocab, build_vocab_polyphonic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_STEP = 0.05
DEFAULT_AUDIO_SAMPLE_RATE = 44100
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAVED_VOCAB_FILES = {
    "mono": ["vocab_token_to_id.json", "vocab_test_token_to_id.json"],
    "poly": ["vocab_poly_token_to_id.json", "vocab_giantmidi_token_to_id.json"],
}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _checkpoint_vocab_size(checkpoint_state):
    for key in ("embedding.weight", "fc.weight", "head.weight"):
        tensor = checkpoint_state.get(key)
        if tensor is not None:
            return tensor.shape[0]
    raise ValueError("Impossible de trouver la taille du vocabulaire dans le checkpoint.")


def _load_token_to_id(vocab_path):
    with open(vocab_path, "r") as file:
        raw_mapping = json.load(file)
    return {token: int(index) for token, index in raw_mapping.items()}


def _vocab_from_token_to_id(token_to_id):
    ordered_tokens = sorted(token_to_id.items(), key=lambda item: item[1])
    ids = [index for _, index in ordered_tokens]
    if ids != list(range(len(ids))):
        raise ValueError("Le vocabulaire n'est pas indexé de façon contigue à partir de 0.")

    vocab = [token for token, _ in ordered_tokens]
    id_to_token = {index: token for token, index in token_to_id.items()}
    return vocab, token_to_id, id_to_token


def _infer_tokenizer_mode_from_vocab(vocab):
    if any(
        token.startswith("NOTE_ON_")
        or token.startswith("NOTE_OFF_")
        or token.startswith("SHIFT_")
        for token in vocab
    ):
        return "poly"
    return "mono"


def _candidate_mode_order(tokenizer_mode):
    modes = []
    if tokenizer_mode in {"mono", "poly"}:
        modes.append(tokenizer_mode)
    for mode in ("mono", "poly"):
        if mode not in modes:
            modes.append(mode)
    return modes


def _iter_saved_vocab_candidates():
    seen_paths = set()
    for declared_mode, filenames in SAVED_VOCAB_FILES.items():
        for filename in filenames:
            vocab_path = PROCESSED_DATA_DIR / filename
            if not vocab_path.exists():
                continue
            seen_paths.add(vocab_path.resolve())
            token_to_id = _load_token_to_id(vocab_path)
            vocab, token_to_id, id_to_token = _vocab_from_token_to_id(token_to_id)
            yield {
                "vocab": vocab,
                "token_to_id": token_to_id,
                "id_to_token": id_to_token,
                "tokenizer_mode": declared_mode,
                "source": str(vocab_path),
            }

    if not PROCESSED_DATA_DIR.exists():
        return

    for vocab_path in sorted(PROCESSED_DATA_DIR.glob("*_token_to_id.json")):
        if vocab_path.resolve() in seen_paths:
            continue
        token_to_id = _load_token_to_id(vocab_path)
        vocab, token_to_id, id_to_token = _vocab_from_token_to_id(token_to_id)
        yield {
            "vocab": vocab,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "tokenizer_mode": _infer_tokenizer_mode_from_vocab(vocab),
            "source": str(vocab_path),
        }


def _iter_generated_vocab_candidates():
    for mode, builder in (("mono", build_vocab), ("poly", build_vocab_polyphonic)):
        vocab, token_to_id, id_to_token = builder()
        yield {
            "vocab": vocab,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "tokenizer_mode": mode,
            "source": f"generated:{mode}",
        }


def _ordered_vocab_candidates(tokenizer_mode):
    candidates = list(_iter_saved_vocab_candidates()) + list(_iter_generated_vocab_candidates())
    mode_rank = {mode: index for index, mode in enumerate(_candidate_mode_order(tokenizer_mode))}

    indexed_candidates = list(enumerate(candidates))
    return [
        candidate
        for _, candidate in sorted(
            indexed_candidates,
            key=lambda indexed_candidate: (
                mode_rank.get(indexed_candidate[1]["tokenizer_mode"], len(mode_rank)),
                1 if indexed_candidate[1]["source"].startswith("generated:") else 0,
                indexed_candidate[0],
            ),
        )
    ]


def resolve_generation_vocab(tokenizer_mode="mono", checkpoint_state=None):
    checkpoint_vocab_size = None
    if checkpoint_state is not None:
        checkpoint_vocab_size = _checkpoint_vocab_size(checkpoint_state)

    candidates = _ordered_vocab_candidates(tokenizer_mode)
    if checkpoint_vocab_size is None:
        return candidates[0]

    for candidate in candidates:
        if len(candidate["vocab"]) == checkpoint_vocab_size:
            return candidate

    available_sizes = ", ".join(
        f"{candidate['tokenizer_mode']}:{len(candidate['vocab'])} ({candidate['source']})"
        for candidate in candidates
    )
    raise ValueError(
        "Aucun vocabulaire de génération ne correspond au checkpoint "
        f"({checkpoint_vocab_size} tokens). Tailles disponibles: {available_sizes}"
    )


def infer_generation_tokenizer_mode(checkpoint_path, fallback_mode="mono"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = _extract_state_dict(checkpoint)
    vocab_info = resolve_generation_vocab(
        tokenizer_mode=fallback_mode,
        checkpoint_state=checkpoint_state,
    )
    return vocab_info["tokenizer_mode"]


def _infer_lstm_num_layers(checkpoint_state):
    layer_indices = []
    prefix = "lstm.weight_ih_l"
    for key in checkpoint_state:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):].split("_reverse", 1)[0]
        if suffix.isdigit():
            layer_indices.append(int(suffix))
    return max(layer_indices) + 1 if layer_indices else 2


def _infer_transformer_num_layers(checkpoint_state):
    layer_indices = []
    prefix = "transformer.layers."
    for key in checkpoint_state:
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            layer_indices.append(int(parts[2]))
    return max(layer_indices) + 1 if layer_indices else 6


def _infer_transformer_heads(d_model):
    preferred_heads = {384: 6, 256: 4}
    preferred = preferred_heads.get(d_model)
    if preferred is not None and d_model % preferred == 0:
        return preferred

    for n_heads in (8, 6, 4, 2, 1):
        if d_model % n_heads == 0:
            return n_heads
    return 1


def _build_model_for_checkpoint(model_name, vocab_size, token_to_id, checkpoint_state):
    embedding = checkpoint_state.get("embedding.weight")
    if embedding is None:
        raise ValueError("Le checkpoint ne contient pas embedding.weight.")

    if model_name == "lstm":
        output_weight = checkpoint_state.get("fc.weight")
        if output_weight is None:
            raise ValueError("Le checkpoint LSTM ne contient pas fc.weight.")
        return build_music_lstm(
            vocab_size,
            emb_dim=embedding.shape[1],
            hidden_dim=output_weight.shape[1],
            num_layers=_infer_lstm_num_layers(checkpoint_state),
        )

    if model_name in ["transformer", "transformer_giantmidi"]:
        d_model = embedding.shape[1]
        linear1_weight = checkpoint_state.get("transformer.layers.0.linear1.weight")
        pos_embedding = checkpoint_state.get("pos_embedding.weight")
        return build_music_transformer(
            vocab_size,
            d_model=d_model,
            n_heads=_infer_transformer_heads(d_model),
            n_layers=_infer_transformer_num_layers(checkpoint_state),
            d_ff=linear1_weight.shape[0] if linear1_weight is not None else 1536,
            max_len=pos_embedding.shape[0] if pos_embedding is not None else 512,
            pad_token_id=token_to_id.get("PAD", 0),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def find_available_soundfonts():
    candidate_roots = [
        Path.home() / ".soundfont.sf2",
        Path.home() / ".soundfonts",
        Path(__file__).resolve().parents[2] / "assets" / "soundfonts",
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

def generate_transformer(
    model,
    start_token_id,
    id_to_token,
    max_tokens=200,
    temperature=0.8,
    top_k=10,
    device=device,
):
    model.eval()
    generated = [start_token_id]

    for _ in range(max_tokens):
        x = torch.tensor([generated], dtype=torch.long, device=device)

        if x.size(1) > model.max_len:
            x = x[:, -model.max_len:]

        logits = model(x)
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
    if tokenizer_mode == "poly":
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = _extract_state_dict(checkpoint)
    if "emotion_embedding.weight" in checkpoint_state:
        raise ValueError("Les checkpoints EMOPIA/emotion ne sont plus supportes par ce projet.")

    vocab_info = resolve_generation_vocab(
        tokenizer_mode=tokenizer_mode,
        checkpoint_state=checkpoint_state,
    )
    vocab = vocab_info["vocab"]
    token_to_id = vocab_info["token_to_id"]
    id_to_token = vocab_info["id_to_token"]
    vocab_size = len(vocab)

    model = _build_model_for_checkpoint(
        model_name=model_name,
        vocab_size=vocab_size,
        token_to_id=token_to_id,
        checkpoint_state=checkpoint_state,
    )

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
        start_token = "START"
    if start_token not in token_to_id:
        raise ValueError(f"Token de depart introuvable dans le vocabulaire: {start_token}")
    start_token_id = token_to_id[start_token]

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
            device=device,
        )

    return [id_to_token[token_id] for token_id in generated_ids]
