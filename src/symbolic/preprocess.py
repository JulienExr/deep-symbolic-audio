import os

import pretty_midi

from common.midi import notes_to_monophonic_grid, save_monophonic_midi


DATA_RAW_DIR = "data/raw_midi/maestro-v3.0.0/"
DATA_PROCESSED_DIR = "data/midi_mono/"

TIME_STEP = 0.125  # secondes



def preprocess_midi_to_mono(input_path, output_path):
    midi = pretty_midi.PrettyMIDI(input_path)
    all_notes = []
    for instrument in midi.instruments:
        all_notes.extend(instrument.notes)
    
        mono_notes = notes_to_monophonic_grid(all_notes, TIME_STEP)
    save_monophonic_midi(mono_notes, output_path)

def preprocess_all_midis_to_mono(input_dir, output_dir):
    compteur = 1
    file_name = f"{compteur:04d}.midi"
    nb_files = sum(len(files) for _, _, files in os.walk(input_dir))
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".midi"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file_name)
                preprocess_midi_to_mono(input_path, output_path)
                compteur += 1
                file_name = f"{compteur:04d}.midi"
            if compteur % 10 == 0:
                print(f"Processed {compteur}/{nb_files} files")


if __name__ == "__main__":
    preprocess_all_midis_to_mono(DATA_RAW_DIR, DATA_PROCESSED_DIR)
