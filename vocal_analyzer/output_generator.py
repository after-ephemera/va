import os


def generate_output(output_dir, range_results, llm_results, input_file):
    """Generate a Markdown file with analysis results."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    analysis_file = os.path.join(output_dir, f"{base_name}_analysis.md")

    # Convert NumPy values to Python floats for proper formatting
    min_pitch = float(range_results["min_pitch"])
    max_pitch = float(range_results["max_pitch"])
    min_note = range_results.get("min_note", "N/A")
    max_note = range_results.get("max_note", "N/A")
    total_samples = range_results.get("total_samples", 0)

    with open(analysis_file, "w") as f:
        f.write(f"# Analysis of {input_file}\n\n")
        f.write("## LLM Analysis\n\n")
        f.write(llm_results + "\n\n")
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

        if range_results["plot_file"]:
            plot_filename = os.path.basename(range_results["plot_file"])
            f.write(
                f'<image-card alt="Pitch Distribution" src="{plot_filename}" ></image-card>\n'
            )
    return analysis_file
