# Vocal Analyzer

A tool to analyze vocals from audio files using AI-powered transcription and feature extraction.

## Features

- Extract vocals from audio files (MP3/WAV)
- Extract all available stems (vocals, drums, bass, etc.)
- Transcribe vocal content
- Analyze vocal range and characteristics
- Generate detailed analysis reports with AI insights

## Installation with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Vocal Analyzer

```bash
# Clone or navigate to the project directory
cd /path/to/va

# Install the project with uv
uv pip install -e .
```

## Configuration

Vocal Analyzer can be configured using a TOML configuration file to enable/disable features and customize behavior.

### Config File Locations

The tool looks for config files in the following order:
1. Path specified with `--config` argument
2. `config.toml` in the current directory
3. `~/.config/vocal-analyzer/config.toml`

### Creating a Config File

Copy the example config and customize it:

```bash
cp config.example.toml config.toml
# Edit config.toml to enable/disable features
```

See `config.example.toml` for all available options.

## Usage

### Basic Usage - Extract and Analyze Vocals

```bash
va path/to/audio.mp3
```

This will:
1. Extract vocals from the audio file
2. Transcribe the vocals
3. Analyze vocal features (pitch, range, etc.)
4. Generate an analysis report

Output files will be created in a new directory: `audio-analysis/` next to your input file.

### Using a Custom Config File

```bash
va path/to/audio.mp3 --config my-config.toml
```

### Extract All Stems

To extract all available stems (vocals, drums, bass, etc.) instead of just vocals:

```bash
va path/to/audio.mp3 --all-stems
```

### Specify Output Directory

```bash
va path/to/audio.mp3 -o /path/to/output
```

### List Available Models

To see all available separation models and their supported stems:

```bash
va --list-models
```

### Use a Specific Model

```bash
va path/to/audio.mp3 --model htdemucs_6s.yaml
```

Default model: `htdemucs_6s.yaml` (supports 6-stem separation including vocals, drums, bass, guitar, piano, and other)

### Quiet Mode

```bash
va path/to/audio.mp3 -q
```

## Example

```bash
# Analyze vocals from a song
va ~/Downloads/song.mp3

# Extract all stems using the default model
va ~/Downloads/song.mp3 --all-stems

# Use a different model
va ~/Downloads/song.mp3 --all-stems --model model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

## Output

The tool creates an analysis directory containing:
- `*_vocals.wav` - Extracted vocal track (or multiple stem files with `--all-stems`)
- `*_analysis.txt` - Detailed analysis report including:
  - Transcription
  - Vocal range analysis
  - AI-powered insights on vocal style and technique

## Requirements

- Python >= 3.11
- Dependencies are managed via `pyproject.toml`
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Development

```bash
# Install in editable mode with uv
uv pip install -e .

# Run directly
python -m vocal_analyzer.main path/to/audio.mp3
```

## License

See LICENSE file for details.
