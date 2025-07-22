import os
from spleeter.separator import Separator


def extract_vocals(input_file, output_dir):
    """Extract vocals from an audio file using Spleeter."""
    separator = Separator("spleeter:2stems")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, base_name)
    separator.separate_to_file(input_file, output_path)
    vocal_file = os.path.join(output_path, "vocals.wav")
    return vocal_file
