import argparse
import os
from .vocal_extractor import extract_vocals
from .transcriber import transcribe_audio
from .feature_extractor import extract_features
from .range_analyzer import RangeAnalyzer
from .llm_analyzer import LLMAnalyzer
from .output_generator import generate_output


def main():
    """Main function to orchestrate vocal analysis."""
    parser = argparse.ArgumentParser(description="Vocal Analyzer")
    parser.add_argument("input_file", help="Input audio file (WAV or MP3)")
    parser.add_argument("-o", "--output_dir", help="Output directory", default=None)
    parser.add_argument("-q", "--quiet", action="store_true", help="Run in quiet mode")
    args = parser.parse_args()

    # Validate input file
    print("starting")
    if not os.path.exists(args.input_file):
        print("Error: Input file does not exist")
        return
    if not args.input_file.lower().endswith((".wav", ".mp3")):
        print("Error: Input file must be WAV or MP3")
        return

    # Set output directory
    output_dir = (
        args.output_dir if args.output_dir else os.path.dirname(args.input_file)
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Extract vocals
        vocal_file = extract_vocals(args.input_file, output_dir)

        # Transcribe vocals
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
            print(f"Analysis complete. Output files: {vocal_file}, {analysis_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
