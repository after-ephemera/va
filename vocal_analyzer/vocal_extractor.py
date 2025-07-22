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


def extract_all_stems(
    input_file, output_dir, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
):
    """Extract all available stems from an audio file using audio-separator.

    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory where output files will be saved
        model_filename (str): Model to use for separation (defaults to high-quality roformer model)

    Returns:
        dict: Dictionary mapping stem names to their file paths

    Example:
        stems = extract_all_stems("song.mp3", "/output")
        # Returns: {"vocals": "/output/song_Vocals.wav", "instrumental": "/output/song_Instrumental.wav"}
    """
    # Initialize the separator WITHOUT output_single_stem to get all stems
    separator = Separator(
        model_file_dir="/tmp/audio-separator-models/",  # Cache models here
        output_dir=output_dir,
        output_format="wav",
        # NOTE: No output_single_stem parameter - this means all stems will be output
        normalization_threshold=0.9,  # Good default for vocal analysis
        sample_rate=44100,  # Standard sample rate
    )

    # Load the specified model
    separator.load_model(model_filename=model_filename)

    try:
        # Perform separation - this will output all available stems
        output_files = separator.separate(input_file)

        # Convert relative paths to absolute paths
        absolute_output_files = []
        for file_path in output_files:
            if os.path.isabs(file_path):
                absolute_output_files.append(file_path)
            else:
                # Join with output directory to get absolute path
                absolute_path = os.path.join(output_dir, file_path)
                absolute_output_files.append(absolute_path)

        print(f"Found {len(absolute_output_files)} output files from separation.")

        # Get information about what stems this model provides
        models_info = separator.get_simplified_model_list()
        model_info = None
        for filename, info in models_info.items():
            if model_filename in filename:
                model_info = info
                break

        if not model_info:
            raise Exception(f"Could not find information for model {model_filename}")

        # Parse stem names from the model info
        # The format is like: ['vocals* (11.8)', 'instrumental (16.5)']
        available_stems = []
        for stem_info in model_info["Stems"]:
            # Extract just the stem name (remove asterisk and score info)
            stem_name = stem_info.split("(")[0].strip().rstrip("*").strip()
            available_stems.append(stem_name)

        print(f"Model '{model_info['Name']}' can produce stems: {available_stems}")

        # Map output files to stem names
        stem_files = {}

        for file_path in absolute_output_files:
            if not os.path.exists(file_path):
                continue

            filename = os.path.basename(file_path)
            file_matched = False

            # Try to determine which stem this file represents
            for stem_name in available_stems:
                # Skip if this stem is already assigned
                if stem_name in stem_files:
                    continue

                # Check various possible naming patterns
                # Audio-separator typically uses patterns like _(Vocals)_ or _(Instrumental)_
                stem_patterns = [
                    f"_({stem_name.capitalize()})_",  # _(Vocals)_, _(Instrumental)_
                    f"({stem_name.capitalize()})",  # (Vocals), (Instrumental)
                    f"_{stem_name.capitalize()}_",  # _Vocals_, _Instrumental_
                    f"_({stem_name.lower()})_",  # _(vocals)_, _(instrumental)_
                    f"({stem_name.lower()})",  # (vocals), (instrumental)
                    f"_{stem_name.lower()}_",  # _vocals_, _instrumental_
                    f"_({stem_name.upper()})_",  # _(VOCALS)_, _(INSTRUMENTAL)_
                    f"({stem_name.upper()})",  # (VOCALS), (INSTRUMENTAL)
                    f"_{stem_name.upper()}_",  # _VOCALS_, _INSTRUMENTAL_
                ]

                for pattern in stem_patterns:
                    if pattern in filename:
                        stem_files[stem_name] = file_path
                        print(f"Found {stem_name} stem: {filename}")
                        file_matched = True
                        break

                if file_matched:
                    break

            if not file_matched:
                print(f"Could not identify stem type for file: {filename}")

        # If we couldn't match by name patterns, fall back to the order of files
        if len(stem_files) < len(available_stems) and len(absolute_output_files) >= len(
            available_stems
        ):
            print("Warning: Could not match all stems by filename, using file order")
            for i, stem_name in enumerate(available_stems):
                if stem_name not in stem_files and i < len(absolute_output_files):
                    stem_files[stem_name] = absolute_output_files[i]
                    print(
                        f"Assigned {stem_name} stem to: {os.path.basename(absolute_output_files[i])}"
                    )

        if not stem_files:
            available_files = (
                os.listdir(output_dir) if os.path.exists(output_dir) else []
            )
            raise Exception(
                f"No stem files found. Output files returned: {output_files}, Available files in {output_dir}: {available_files}"
            )

        return stem_files

    except Exception as e:
        raise Exception(f"All stems extraction failed: {str(e)}")


def get_model_stem_info(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt"):
    """Get information about what stems a model can produce.

    Args:
        model_filename (str): Model filename to check

    Returns:
        dict: Model information including available stems
    """
    separator = Separator(info_only=True)
    models_info = separator.get_simplified_model_list()

    for filename, info in models_info.items():
        if model_filename in filename:
            return info

    return None
