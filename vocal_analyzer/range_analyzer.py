import librosa
import matplotlib.pyplot as plt
import numpy as np
import os


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


def note_to_frequency(note_name):
    """Convert a note name like 'C4' to frequency in Hz."""
    if note_name == "N/A":
        return 0

    # Parse note name and octave
    if len(note_name) < 2:
        return 0

    note_part = note_name[:-1]  # Everything except the last character (octave)
    octave_part = note_name[-1]  # Last character (octave)

    try:
        octave = int(octave_part)
    except ValueError:
        return 0

    # Chromatic scale starting from C
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    if note_part not in note_names:
        return 0

    note_index = note_names.index(note_part)

    # Calculate frequency
    A4 = 440
    C0 = A4 * np.power(2, -4.75)
    half_steps = octave * 12 + note_index

    return C0 * np.power(2, half_steps / 12)


def generate_all_notes_in_range(min_freq, max_freq):
    """Generate all chromatic notes within a frequency range."""
    if min_freq <= 0 or max_freq <= 0:
        return [], []

    # Get the min and max notes
    min_note = frequency_to_note(min_freq)
    max_note = frequency_to_note(max_freq)

    if min_note == "N/A" or max_note == "N/A":
        return [], []

    # Parse octave ranges
    min_octave = int(min_note[-1])
    max_octave = int(max_note[-1])

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    all_notes = []
    all_frequencies = []

    # Generate all notes from min to max
    for octave in range(min_octave, max_octave + 1):
        for note_name in note_names:
            note = f"{note_name}{octave}"
            freq = note_to_frequency(note)

            # Only include notes within our frequency range
            if min_freq <= freq <= max_freq:
                all_notes.append(note)
                all_frequencies.append(freq)

    return all_frequencies, all_notes


def get_octave_boundaries(frequencies, notes):
    """Find frequencies where octaves change for adding visual separators."""
    boundaries = []
    current_octave = None

    for freq, note in zip(frequencies, notes):
        octave = int(note[-1])
        if current_octave is not None and octave != current_octave:
            boundaries.append(freq)
        current_octave = octave

    return boundaries


def notes_for_plotting(min_freq, max_freq, num_bins=50):
    """Generate note labels for plotting within a frequency range."""
    return generate_all_notes_in_range(min_freq, max_freq)


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
            return {
                "min_pitch": 0,
                "max_pitch": 0,
                "min_note": "N/A",
                "max_note": "N/A",
                "plot_file": None,
            }

        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        total_samples = len(pitches)

        # Convert to notes
        min_note = frequency_to_note(min_pitch)
        max_note = frequency_to_note(max_pitch)

        # Plot histogram
        plt.figure(figsize=(14, 6))
        n, bins, patches = plt.hist(
            pitches, bins=50, color="#4CAF50", edgecolor="#000000", alpha=0.7
        )

        # Set up x-axis with note labels for every note in range
        freq_positions, note_labels = notes_for_plotting(min_pitch, max_pitch)
        if freq_positions and note_labels:
            plt.xticks(freq_positions, note_labels, rotation=45, fontsize=8)

            # Add octave boundary lines
            octave_boundaries = get_octave_boundaries(freq_positions, note_labels)
            for boundary_freq in octave_boundaries:
                plt.axvline(
                    x=boundary_freq, color="red", linestyle="--", alpha=0.5, linewidth=1
                )

            # Add octave labels at the top
            octaves_in_range = set(int(note[-1]) for note in note_labels)
            for octave in sorted(octaves_in_range):
                # Find first C note in this octave for labeling position
                c_note = f"C{octave}"
                if c_note in note_labels:
                    c_freq = note_to_frequency(c_note)
                    plt.text(
                        c_freq,
                        max(n) * 1.05,
                        f"Octave {octave}",
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="darkblue",
                        alpha=0.8,
                    )

        plt.xlabel("Musical Notes (with Octave Numbers)")
        plt.ylabel("Frequency")
        plt.title(f"Vocal Pitch Distribution ({total_samples:,} samples)")
        plt.tight_layout()  # Adjust layout to prevent label cutoff

        base_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        plot_file = os.path.join(self.output_dir, f"{base_name}_pitch_distribution.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "min_pitch": min_pitch,
            "max_pitch": max_pitch,
            "min_note": min_note,
            "max_note": max_note,
            "total_samples": total_samples,
            "plot_file": plot_file,
        }
