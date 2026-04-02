import math
from collections import Counter
from itertools import combinations

from symbolic.generate import tokens_to_pretty_midi_dispatch


CONSONANT_INTERVAL_CLASSES = {3, 4, 5}


def get_notes_from_midi(midi):
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))


def build_onset_groups(notes):
    groups = []
    current_key = None
    current_notes = []

    for note in notes:
        onset_key = round(float(note.start), 6)
        if onset_key != current_key:
            if current_notes:
                groups.append((current_key, current_notes))
            current_key = onset_key
            current_notes = [note]
        else:
            current_notes.append(note)

    if current_notes:
        groups.append((current_key, current_notes))
    return groups


def build_activity_frames(notes, total_duration):
    events = []
    for note in notes:
        events.append((float(note.start), 1, int(note.pitch)))
        events.append((float(note.end), -1, int(note.pitch)))

    events.sort(key=lambda event: (event[0], event[1], event[2]))

    frames = []
    active_pitches = Counter()
    last_time = 0.0
    index = 0

    while index < len(events):
        current_time = events[index][0]
        duration = current_time - last_time
        if duration > 0:
            frames.append((duration, tuple(sorted(active_pitches.elements()))))

        while index < len(events) and events[index][0] == current_time:
            _, delta, pitch = events[index]
            if delta > 0:
                active_pitches[pitch] += 1
            elif active_pitches[pitch] > 1:
                active_pitches[pitch] -= 1
            else:
                active_pitches.pop(pitch, None)
            index += 1
        last_time = current_time

    if total_duration > last_time:
        frames.append((total_duration - last_time, tuple(sorted(active_pitches.elements()))))

    return frames


def compute_pitch_class_profile(notes):
    profile = [0.0] * 12
    total_weight = 0.0

    for note in notes:
        duration = max(float(note.end - note.start), 1e-6)
        profile[int(note.pitch) % 12] += duration
        total_weight += duration

    if total_weight == 0.0:
        return profile, 0.0, 0.0

    normalized_profile = [value / total_weight for value in profile]
    entropy = 0.0
    for value in normalized_profile:
        if value > 0:
            entropy -= value * math.log2(value)

    max_entropy = math.log2(12)
    tonal_center_strength = max(normalized_profile)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return normalized_profile, normalized_entropy, tonal_center_strength


def repetition_ratio(sequence, n):
    total_ngrams = len(sequence) - n + 1
    if total_ngrams <= 0:
        return 0.0

    counts = Counter(tuple(sequence[index:index + n]) for index in range(total_ngrams))
    repeated_occurrences = sum(count for count in counts.values() if count > 1)
    return repeated_occurrences / total_ngrams


def compute_rhythmic_diversity(onset_groups):
    if len(onset_groups) < 2:
        return 0.0

    inter_onset_intervals = [
        round(float(next_onset) - float(current_onset), 6)
        for (current_onset, _), (next_onset, _) in zip(onset_groups, onset_groups[1:])
    ]
    return len(set(inter_onset_intervals)) / len(inter_onset_intervals)


def compute_consonance_ratio(frames):
    consonant_weight = 0.0
    total_weight = 0.0

    for duration, active_pitches in frames:
        if duration <= 0 or len(active_pitches) < 2:
            continue

        unique_pitch_classes = sorted({pitch % 12 for pitch in active_pitches})
        if len(unique_pitch_classes) < 2:
            continue

        interval_classes = []
        for first_pitch_class, second_pitch_class in combinations(unique_pitch_classes, 2):
            raw_interval = abs(second_pitch_class - first_pitch_class) % 12
            interval_classes.append(min(raw_interval, 12 - raw_interval))

        total_weight += len(interval_classes) * duration
        consonant_weight += sum(
            1 for interval_class in interval_classes if interval_class in CONSONANT_INTERVAL_CLASSES
        ) * duration

    return consonant_weight / total_weight if total_weight > 0 else 0.0


def build_empty_music_metrics(total_duration, note_count=0):
    return {
        "note_count": note_count,
        "total_duration_sec": total_duration,
        "note_density_per_sec": 0.0,
        "silence_ratio": 1.0 if total_duration > 0 else 0.0,
        "mean_polyphony": 0.0,
        "pitch_class_entropy": 0.0,
        "tonal_center_strength": 0.0,
        "consonance_ratio": 0.0,
        "harmonic_motif_repetition_ratio_4": 0.0,
        "rhythmic_diversity_ratio": 0.0,
        "pitch_class_profile": [0.0] * 12,
    }


def compute_music_metrics_from_midi(midi):
    notes = get_notes_from_midi(midi)
    total_duration = max(float(midi.get_end_time()), 0.0)

    if not notes or total_duration == 0.0:
        return build_empty_music_metrics(total_duration, note_count=len(notes))

    onset_groups = build_onset_groups(notes)
    frames = build_activity_frames(notes, total_duration)
    pitch_class_profile, pitch_class_entropy, tonal_center_strength = compute_pitch_class_profile(notes)

    silence_ratio = sum(duration for duration, active_pitches in frames if not active_pitches) / total_duration
    mean_polyphony = sum(len(active_pitches) * duration for duration, active_pitches in frames) / total_duration
    onset_signatures = [
        tuple(sorted({int(note.pitch) % 12 for note in group_notes}))
        for _, group_notes in onset_groups
    ]

    return {
        "note_count": len(notes),
        "total_duration_sec": total_duration,
        "note_density_per_sec": len(notes) / total_duration,
        "silence_ratio": silence_ratio,
        "mean_polyphony": mean_polyphony,
        "pitch_class_entropy": pitch_class_entropy,
        "tonal_center_strength": tonal_center_strength,
        "consonance_ratio": compute_consonance_ratio(frames),
        "harmonic_motif_repetition_ratio_4": repetition_ratio(onset_signatures, 4),
        "rhythmic_diversity_ratio": compute_rhythmic_diversity(onset_groups),
        "pitch_class_profile": pitch_class_profile,
    }


def compute_music_metrics(tokens, tokenizer_mode):
    midi = tokens_to_pretty_midi_dispatch(tokens, tokenizer_mode=tokenizer_mode)
    return compute_music_metrics_from_midi(midi)
