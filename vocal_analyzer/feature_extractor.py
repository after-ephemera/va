import librosa
import numpy as np


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
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        mean_pitch = np.mean(pitches)
    else:
        min_pitch = max_pitch = mean_pitch = 0
    # Detect screaming: simple amplitude threshold
    max_amp = np.max(np.abs(y))
    screaming = max_amp > 0.8  # Arbitrary threshold
    return {
        "tempo": tempo,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "mean_pitch": mean_pitch,
        "screaming": screaming,
    }
