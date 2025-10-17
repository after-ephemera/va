import os


def generate_output(output_dir, range_results, llm_results, input_file, key_info, transcription=""):
    """Generate a Markdown file with analysis results.

    Args:
        output_dir: Directory for output files
        range_results: Dictionary with range analysis results
        llm_results: String with LLM analysis results
        input_file: Path to input audio file
        key_info: Dictionary with key detection results or None
        transcription: String with transcription results or empty string
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    analysis_file = os.path.join(output_dir, f"{base_name}_analysis.md")

    # Extract range results if available
    min_pitch = None
    max_pitch = None
    min_note = "N/A"
    max_note = "N/A"
    total_samples = 0
    plot_file = None

    if range_results:
        min_pitch = float(range_results["min_pitch"])
        max_pitch = float(range_results["max_pitch"])
        min_note = range_results.get("min_note", "N/A")
        max_note = range_results.get("max_note", "N/A")
        total_samples = range_results.get("total_samples", 0)
        plot_file = range_results.get("plot_file")

    with open(analysis_file, "w") as f:
        f.write(f"# Analysis of {input_file}\n\n")

        # Only write key section if key detection was enabled
        if key_info:
            f.write("## Musical Key\n\n")
            f.write(f"**Key:** {key_info['key']} (correlation: {key_info['correlation']:.3f})\n\n")
            if key_info['alt_key'] is not None:
                f.write(f"**Also possible:** {key_info['alt_key']} (correlation: {key_info['alt_correlation']:.3f})\n\n")
            f.write("*Key detected using the Krumhansl-Schmuckler key-finding algorithm*\n\n")

        # Write transcription section if transcription was performed
        if transcription:
            f.write("## Transcription\n\n")
            f.write(f"{transcription}\n\n")

        # Only write LLM section if analysis was performed
        if llm_results:
            f.write("## LLM Analysis\n\n")
            f.write(llm_results + "\n\n")

        # Only write range section if range analysis was performed
        if range_results:
            f.write("## Range Analysis\n\n")

            if min_note != "N/A" and max_note != "N/A":
                f.write(
                    f"The vocal range is from **{min_note}** to **{max_note}** "
                    f"({min_pitch:.2f} Hz to {max_pitch:.2f} Hz).\n\n"
                )
                if total_samples > 0:
                    f.write(f"Analysis based on **{total_samples:,}** pitch samples.\n\n")
            else:
                f.write("No vocal range detected in the audio file.\n\n")

            # Include pitch distribution plot if available
            if plot_file:
                plot_filename = os.path.basename(plot_file)
                f.write(f"![Pitch Distribution]({plot_filename})\n\n")

    return analysis_file
