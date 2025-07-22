import librosa
import numpy as np


def frequency_to_note(frequency):
    """Convert frequency in Hz to musical note name."""
    if frequency <= 0:
        return "N/A"

    # A4 = 440 Hz is our reference
    A4 = 440
    C0 = A4 * np.power(2, -4.75)  # C0 frequency

    # Calculate the number of half steps from C0
    if frequency > 0:
        half_steps = round(12 * np.log2(frequency / C0))
    else:
        return "N/A"

    # Chromatic scale starting from C
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate octave and note
    octave = half_steps // 12
    note_index = half_steps % 12

    return f"{note_names[note_index]}{octave}"


def extract_features(audio_file):
    """Extract audio features like tempo, pitch, and screaming presence."""
    y, sr = librosa.load(audio_file)
    # Extract tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Extract pitches
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # Get pitches with significant magnitude
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitches = pitches[pitches > 0]
    if len(pitches) > 0:
        min_pitch = float(np.min(pitches))
        max_pitch = float(np.max(pitches))
        mean_pitch = float(np.mean(pitches))
        # Convert to notes
        min_note = frequency_to_note(min_pitch)
        max_note = frequency_to_note(max_pitch)
        mean_note = frequency_to_note(mean_pitch)
    else:
        min_pitch = max_pitch = mean_pitch = 0.0
        min_note = max_note = mean_note = "N/A"
    # Detect screaming: simple amplitude threshold
    max_amp = np.max(np.abs(y))
    screaming = max_amp > 0.8  # Arbitrary threshold
    return {
        "tempo": float(tempo),
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "mean_pitch": mean_pitch,
        "min_note": min_note,
        "max_note": max_note,
        "mean_note": mean_note,
        "screaming": screaming,
    }
