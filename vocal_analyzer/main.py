import argparse
import os
from .vocal_extractor import extract_vocals, extract_all_stems, get_model_stem_info
from .transcriber import transcribe_audio
from .feature_extractor import extract_features
from .range_analyzer import RangeAnalyzer
from .llm_analyzer import LLMAnalyzer
from .output_generator import generate_output


def main():
    """Main function to orchestrate vocal analysis."""
    parser = argparse.ArgumentParser(description="Vocal Analyzer")
    parser.add_argument("input_file", nargs="?", help="Input audio file (WAV or MP3)")
    parser.add_argument("-o", "--output_dir", help="Output directory", default=None)
    parser.add_argument("-q", "--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--all-stems",
        action="store_true",
        help="Extract all available stems from the audio instead of just vocals",
    )
    parser.add_argument(
        "--model",
        default="htdemucs_6s.yaml",
        help="Model to use for separation (default: model_bs_roformer_ep_317_sdr_12.9755.ckpt)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and their supported stems",
    )
    args = parser.parse_args()

    # Handle list models option
    if args.list_models:
        from audio_separator.separator import Separator

        sep = Separator(info_only=True)
        models = sep.get_simplified_model_list()

        print("Available models and their supported stems:")
        print("=" * 60)

        # Group by model type
        model_types = {}
        for filename, info in models.items():
            model_type = info["Type"]
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append((filename, info))

        for model_type in sorted(model_types.keys()):
            print(f"\n{model_type} Models:")
            print("-" * 30)
            for filename, info in model_types[model_type][
                :5
            ]:  # Show top 5 of each type
                print(f"  {filename}")
                print(f"    Name: {info['Name']}")
                print(f"    Stems: {', '.join(info['Stems'])}")
                print()

        print("Use --model <filename> to specify which model to use.")
        return

    # Check if input file is provided when needed
    if not args.input_file:
        print("Error: Input file is required when not using --list-models")
        parser.print_help()
        return

    # Validate input file
    print("starting")
    if not os.path.exists(args.input_file):
        print("Error: Input file does not exist")
        return
    if not args.input_file.lower().endswith((".wav", ".mp3")):
        print("Error: Input file must be WAV or MP3")
        return

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create analysis subdirectory next to original file
        input_dir = os.path.dirname(args.input_file)
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.join(input_dir, f"{input_basename}-analysis")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        if args.all_stems:
            # Extract all available stems
            if not args.quiet:
                print(f"Extracting all stems using model: {args.model}")

                # Show what stems this model can produce
                model_info = get_model_stem_info(args.model)
                if model_info:
                    stems_list = [
                        stem.split("(")[0].strip().rstrip("*").strip()
                        for stem in model_info["Stems"]
                    ]
                    print(f"This model can produce: {', '.join(stems_list)}")

            stem_files = extract_all_stems(args.input_file, output_dir, args.model)

            if not args.quiet:
                print("All stems extracted successfully:")
                for stem_name, file_path in stem_files.items():
                    print(f"  {stem_name}: {os.path.basename(file_path)}")

            # For analysis, we still need the vocals file specifically
            vocal_file = stem_files.get("vocals") or stem_files.get("Vocals")
            if not vocal_file:
                # If no vocals stem, try to find the best one for analysis
                if "vocal" in stem_files:
                    vocal_file = stem_files["vocal"]
                else:
                    # Use the first available stem as fallback
                    vocal_file = next(iter(stem_files.values()))
                    if not args.quiet:
                        print(
                            f"No vocals stem found, using {list(stem_files.keys())[0]} for analysis"
                        )

            output_files_list = list(stem_files.values())
        else:
            # Extract vocals only (original behavior)
            vocal_file = extract_vocals(args.input_file, output_dir)
            output_files_list = [vocal_file]

        # Transcribe vocals (or the chosen stem for analysis)
        transcription = transcribe_audio(vocal_file)

        # Extract features
        features = extract_features(vocal_file)

        # Initialize and run analyzers
        range_analyzer = RangeAnalyzer(vocal_file, output_dir)
        range_results = range_analyzer.analyze()

        llm_analyzer = LLMAnalyzer(transcription, features)
        llm_results = llm_analyzer.analyze()

        # Generate output
        analysis_file = generate_output(
            output_dir, range_results, llm_results, args.input_file
        )

        # Print summary
        if not args.quiet:
            if args.all_stems:
                print(
                    f"Analysis complete. Stem files: {', '.join([os.path.basename(f) for f in output_files_list])}"
                )
                print(f"Analysis file: {os.path.basename(analysis_file)}")
            else:
                print(
                    f"Analysis complete. Output files: {os.path.basename(vocal_file)}, {os.path.basename(analysis_file)}"
                )

    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
