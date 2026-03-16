import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    find_available_soundfonts,
    generate_tokens,
    tokens_to_midi,
    tokens_to_wav_bytes,
)


st.set_page_config(page_title="Deep Symbolic Audio UI", layout="wide")

EMOPIA_START_TOKENS = {
    "HAPPY": "START_HAPPY",
    "SAD": "START_SAD",
    "ANGRY": "START_ANGRY",
    "RELAXED": "START_RELAXED",
}
UI_SOUNDFONT_DIR = ROOT_DIR / "assets" / "soundfonts"


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_path(path):
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def infer_tokenizer_mode(checkpoint_path, fallback_mode):
    checkpoint_name = checkpoint_path.name.lower()
    if "emopia" in checkpoint_name or "emotion" in checkpoint_name:
        return "emopia"
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


def ensure_uploaded_soundfont(uploaded_file):
    if uploaded_file is None:
        return None
    UI_SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UI_SOUNDFONT_DIR / uploaded_file.name
    output_path.write_bytes(uploaded_file.getbuffer())
    return output_path


def save_generation(
    tokens,
    model_name,
    tokenizer_mode,
    selected_emotion=None,
    sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
    soundfont_path=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / "outputs" / "ui_generations"
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_suffix = f"_{selected_emotion.lower()}" if tokenizer_mode == "emopia" and selected_emotion else ""
    midi_path = output_dir / f"{model_name}_{tokenizer_mode}{emotion_suffix}_{timestamp}.mid"
    tokens_to_midi(tokens, str(midi_path), tokenizer_mode=tokenizer_mode)
    wav_bytes, renderer_name = tokens_to_wav_bytes(
        tokens,
        sample_rate=sample_rate,
        return_renderer=True,
        tokenizer_mode=tokenizer_mode,
        soundfont_path=str(soundfont_path) if soundfont_path is not None else None,
    )
    return midi_path, wav_bytes, renderer_name


st.title("Deep Symbolic Audio")
st.caption("Génère et écoute directement un fichier MIDI créé par le modèle, y compris les versions fine-tunées avec émotions.")

left_col, right_col = st.columns([1, 1])

with left_col:
    model_name = st.selectbox("Modèle", ["lstm", "transformer"])
    tokenizer_mode = st.selectbox("Mode de tokens", ["mono", "poly", "emopia"])
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
    selected_emotion = None
    start_token = None

    if effective_tokenizer_mode == "emopia":
        selected_emotion = st.selectbox("Emotion", list(EMOPIA_START_TOKENS.keys()))
        start_token = EMOPIA_START_TOKENS[selected_emotion]

    max_tokens = st.slider("Nombre max de tokens", min_value=32, max_value=512, value=200, step=8)
    temperature = st.slider("Température", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
    top_k = st.slider("Top-k", min_value=1, max_value=64, value=10, step=1)
    sample_rate = st.select_slider(
        "Qualité audio",
        options=[16000, 22050, 32000, 44100],
        value=DEFAULT_AUDIO_SAMPLE_RATE,
        format_func=lambda value: f"{value / 1000:.1f} kHz",
    )

    available_soundfonts = find_available_soundfonts()
    soundfont_labels = {display_path(path): path for path in available_soundfonts}
    soundfont_options = ["Auto (meilleur disponible)"] + list(soundfont_labels)
    selected_soundfont_option = st.selectbox("Soundfont", soundfont_options)
    uploaded_soundfont = st.file_uploader(
        "Ou importer un soundfont piano (.sf2/.sf3)",
        type=["sf2", "sf3"],
        help="Pour un vrai son de piano, importe un soundfont de piano. Il sera sauvegardé dans assets/soundfonts/.",
    )
    selected_soundfont_path = None
    if uploaded_soundfont is not None:
        selected_soundfont_path = ensure_uploaded_soundfont(uploaded_soundfont)
    elif selected_soundfont_option != "Auto (meilleur disponible)":
        selected_soundfont_path = soundfont_labels[selected_soundfont_option]

    device = resolve_device()

    generate_button = st.button("Générer", type="primary", use_container_width=True)

with right_col:
    st.markdown("### Infos")
    st.write(f"Device: `{device}`")
    st.write(f"Checkpoint sélectionné: `{checkpoint_path.relative_to(ROOT_DIR)}`")
    st.write(f"Mode de tokens demandé: `{tokenizer_mode}`")
    st.write(f"Mode de tokens utilisé: `{effective_tokenizer_mode}`")
    st.write(f"Qualité audio: `{sample_rate} Hz`")
    if selected_soundfont_path is not None:
        st.write(f"Soundfont utilisé: `{display_path(selected_soundfont_path)}`")
    elif available_soundfonts:
        st.write(f"Soundfont auto: `{display_path(available_soundfonts[0])}`")
    else:
        st.warning("Aucun soundfont trouvé. L'UI utilisera le rendu de secours, plus synthétique.")
    if start_token is not None:
        st.write(f"Emotion demandée: `{selected_emotion}`")
        st.write(f"Token de départ: `{start_token}`")


if generate_button:
    with st.spinner("Génération en cours..."):
        tokens = generate_tokens(
            model_name=model_name,
            checkpoint_path=str(checkpoint_path),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            tokenizer_mode=effective_tokenizer_mode,
            start_token=start_token,
            device=device,
        )
        midi_path, wav_bytes, renderer_name = save_generation(
            tokens,
            model_name,
            effective_tokenizer_mode,
            selected_emotion=selected_emotion,
            sample_rate=sample_rate,
            soundfont_path=selected_soundfont_path,
        )

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
