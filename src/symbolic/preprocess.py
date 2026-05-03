import os
from pathlib import Path

import pretty_midi

from common.midi import notes_to_monophonic_grid, save_monophonic_midi


DATA_RAW_DIR = "data/raw_midi/maestro-v3.0.0/"
DATA_PROCESSED_DIR = "data/midi_mono/"

TIME_STEP = 0.125



def preprocess_midi_to_mono(input_path, output_path):
    midi = pretty_midi.PrettyMIDI(input_path)
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        all_notes.extend(instrument.notes)

    mono_notes = notes_to_monophonic_grid(all_notes, TIME_STEP)
    save_monophonic_midi(mono_notes, output_path)


def preprocess_all_midis_to_mono(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    compteur = 1
    file_name = f"{compteur:04d}.midi"
    midi_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.lower().endswith((".mid", ".midi"))
    ]
    nb_files = len(midi_files)

    for input_path in midi_files:
        output_path = os.path.join(output_dir, file_name)
        preprocess_midi_to_mono(input_path, output_path)
        if compteur % 10 == 0:
            print(f"Processed {compteur}/{nb_files} files")
        compteur += 1
        file_name = f"{compteur:04d}.midi"


if __name__ == "__main__":
    preprocess_all_midis_to_mono(DATA_RAW_DIR, DATA_PROCESSED_DIR)
