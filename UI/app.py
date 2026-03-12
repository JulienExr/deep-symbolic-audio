import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate import generate_tokens, tokens_to_midi, tokens_to_wav_bytes


st.set_page_config(page_title="Deep Symbolic Audio UI", layout="wide")


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_tokenizer_mode(checkpoint_path, fallback_mode):
    checkpoint_name = checkpoint_path.name.lower()
    if "_poly_" in checkpoint_name or checkpoint_name.endswith("_poly.pt"):
        return "poly"
    if "_mono_" in checkpoint_name or checkpoint_name.endswith("_mono.pt"):
        return "mono"
    return fallback_mode


def find_checkpoints(model_name):
    candidates = []

    model_dir = ROOT_DIR / "models" / model_name
    if model_dir.exists():
        candidates.extend(sorted(model_dir.glob("*.pt")))

    legacy_model_dir = ROOT_DIR / "models"
    candidates.extend(sorted(legacy_model_dir.glob(f"{model_name}_*.pt")))

    legacy_checkpoint_dir = ROOT_DIR / "checkpoints" / model_name
    if legacy_checkpoint_dir.exists():
        candidates.extend(sorted(legacy_checkpoint_dir.rglob("*.pt")))

    unique_candidates = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique_candidates.append(path)
            seen.add(resolved)

    return unique_candidates


def save_generation(tokens, model_name, tokenizer_mode):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "outputs" / "ui_generations"
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_path = output_dir / f"{model_name}_{tokenizer_mode}_{timestamp}.mid"
    tokens_to_midi(tokens, str(midi_path), tokenizer_mode=tokenizer_mode)
    wav_bytes, renderer_name = tokens_to_wav_bytes(tokens, return_renderer=True, tokenizer_mode=tokenizer_mode)
    return midi_path, wav_bytes, renderer_name


st.title("Deep Symbolic Audio")
st.caption("Génère et écoute directement un fichier MIDI créé par le modèle LSTM ou Transformer.")

left_col, right_col = st.columns([1, 1])

with left_col:
    model_name = st.selectbox("Modèle", ["lstm", "transformer"])
    tokenizer_mode = st.selectbox("Mode de tokens", ["mono", "poly"])
    available_checkpoints = find_checkpoints(model_name)

    if not available_checkpoints:
        st.error(f"Aucun checkpoint trouvé pour {model_name} dans models/{model_name}/")
        st.stop()

    checkpoint_path = st.selectbox(
        "Checkpoint",
        available_checkpoints,
        format_func=lambda path: str(path.relative_to(ROOT_DIR)),
    )
    effective_tokenizer_mode = infer_tokenizer_mode(checkpoint_path, tokenizer_mode)

    max_tokens = st.slider("Nombre max de tokens", min_value=32, max_value=512, value=200, step=8)
    temperature = st.slider("Température", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    device = resolve_device()

    generate_button = st.button("Générer", type="primary", use_container_width=True)

with right_col:
    st.markdown("### Infos")
    st.write(f"Device: `{device}`")
    st.write(f"Checkpoint sélectionné: `{checkpoint_path.relative_to(ROOT_DIR)}`")
    st.write(f"Mode de tokens demandé: `{tokenizer_mode}`")
    st.write(f"Mode de tokens utilisé: `{effective_tokenizer_mode}`")


if generate_button:
    with st.spinner("Génération en cours..."):
        tokens = generate_tokens(
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            max_tokens=max_tokens,
            temperature=temperature,
            tokenizer_mode=effective_tokenizer_mode,
            device=device,
        )
        midi_path, wav_bytes, renderer_name = save_generation(tokens, model_name, effective_tokenizer_mode)

    st.success("Génération terminée")

    preview_col, token_col = st.columns([1, 1])

    with preview_col:
        st.markdown("### Écoute")
        st.caption(f"Rendu audio utilisé : {renderer_name}")
        st.audio(wav_bytes, format="audio/wav")

        with open(midi_path, "rb") as midi_file:
            midi_bytes = midi_file.read()

        st.download_button(
            label="Télécharger le MIDI",
            data=midi_bytes,
            file_name=midi_path.name,
            mime="audio/midi",
            use_container_width=True,
        )

        st.write(f"Fichier sauvegardé : `{midi_path.relative_to(ROOT_DIR)}`")

    with token_col:
        st.markdown("### Tokens générés")
        st.code(" ".join(tokens), language="text")