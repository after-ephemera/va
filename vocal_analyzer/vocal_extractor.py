import os
from audio_separator.separator import Separator


def extract_vocals(input_file, output_dir):
    """Extract vocals from an audio file using audio-separator."""
    # Initialize the separator with optimized settings for vocal extraction
    separator = Separator(
        model_file_dir="/tmp/audio-separator-models/",  # Cache models here
        output_dir=output_dir,
        output_format="wav",
        output_single_stem="Vocals",  # Only output vocals stem
        normalization_threshold=0.9,  # Good default for vocal analysis
        sample_rate=44100,  # Standard sample rate
    )

    # Load a high-quality vocal separation model (downloads automatically if needed)
    # This model has excellent vocal separation performance
    separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")

    try:
        # Perform separation
        output_files = separator.separate(input_file)

        # Find the vocals file from the output
        vocal_file = None

        # audio-separator returns full paths, look for the vocals file
        for file_path in output_files:
            # Check if this is the vocals file (contains "Vocals" in filename)
            if (
                "Vocals" in os.path.basename(file_path)
                or "vocals" in os.path.basename(file_path).lower()
            ):
                if os.path.exists(file_path):
                    vocal_file = file_path
                    break

        # If not found by name, check all files in output directory
        if not vocal_file and output_files:
            for file_path in output_files:
                if os.path.exists(file_path):
                    vocal_file = file_path
                    break

        # If still not found, try to find any .wav file in the output directory
        if not vocal_file:
            for file in os.listdir(output_dir):
                if file.endswith(".wav") and (
                    "vocal" in file.lower() or "Vocal" in file
                ):
                    full_path = os.path.join(output_dir, file)
                    if os.path.exists(full_path):
                        vocal_file = full_path
                        break

        if vocal_file and os.path.exists(vocal_file):
            return vocal_file
        else:
            # Debug information
            available_files = (
                os.listdir(output_dir) if os.path.exists(output_dir) else []
            )
            raise Exception(
                f"No vocal file found. Output files returned: {output_files}, Available files in {output_dir}: {available_files}"
            )

    except Exception as e:
        raise Exception(f"Vocal extraction failed: {str(e)}")
