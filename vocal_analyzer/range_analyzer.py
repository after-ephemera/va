import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


class RangeAnalyzer:
    """Analyze vocal range and plot pitch distribution."""

    def __init__(self, audio_file, output_dir):
        self.audio_file = audio_file
        self.output_dir = output_dir

    def analyze(self):
        """Analyze pitch range and generate a histogram."""
        y, sr = librosa.load(self.audio_file)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Get pitches where magnitude is above threshold
        threshold = np.median(magnitudes)
        pitches = pitches[magnitudes > threshold]
        pitches = pitches[pitches > 0]
        if len(pitches) == 0:
            return {"min_pitch": 0, "max_pitch": 0, "plot_file": None}
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(pitches, bins=50, color="#4CAF50", edgecolor="#000000")
        plt.xlabel("Pitch (Hz)")
        plt.ylabel("Frequency")
        plt.title("Vocal Pitch Distribution")
        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        plot_file = os.path.join(self.output_dir, f"{base_name}_pitch_distribution.png")
        plt.savefig(plot_file)
        plt.close()
        return {"min_pitch": min_pitch, "max_pitch": max_pitch, "plot_file": plot_file}
