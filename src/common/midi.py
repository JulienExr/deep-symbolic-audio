import numpy as np
import pretty_midi


def notes_to_monophonic_grid(notes, time_step):
    if not notes:
        return []

    max_end = max(note.end for note in notes)
    n_steps = int(np.ceil(max_end / time_step))

    active_pitch_per_step = [None] * n_steps

    for step in range(n_steps):
        t = step * time_step
        active_notes = [
            note for note in notes
            if note.start <= t < note.end
        ]

        if active_notes:
            highest = max(active_notes, key=lambda n: n.pitch)
            active_pitch_per_step[step] = highest.pitch

    mono_notes = []
    current_pitch = None
    current_start = None

    for step, pitch in enumerate(active_pitch_per_step):
        t = step * time_step

        if pitch != current_pitch:
            if current_pitch is not None:
                mono_notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=current_pitch,
                        start=current_start,
                        end=t
                    )
                )
            current_pitch = pitch
            current_start = t if pitch is not None else None

    # fermer la dernière note
    if current_pitch is not None:
        mono_notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=current_pitch,
                start=current_start,
                end=n_steps * time_step
            )
        )

    return mono_notes


def save_monophonic_midi(mono_notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    instrument.notes.extend(mono_notes)
    midi.instruments.append(instrument)

    midi.write(output_path)


def load_mono_note(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    notes.sort(key=lambda n: n.start)
    return notes

def load_polyphonic_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    notes.sort(key=lambda n: n.start)
    return notes

if __name__ == "__main__":
    midi_path = "/home/julien/Documents/UQAC(nogit)/deep_learning/deep-symbolic-audio/data/midi_poly/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi"

    
    notes = load_mono_note(midi_path)
    print(f" Notes extraites du MIDI (format PrettyMIDI.Note):")
    print(notes[:3])    

    print("\n---\n")
    
    notes_mono = notes_to_monophonic_grid(notes, time_step=0.125)
    print(f" Notes allignées sur la grille temporelle:")
    print(notes_mono[:3])
