"""Configuration management for vocal analyzer."""
import os
import sys
from pathlib import Path

# Use tomllib (built-in for Python 3.11+) or tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class Config:
    """Configuration for vocal analyzer features."""

    def __init__(self, config_path=None):
        """Load configuration from TOML file or use defaults.

        Args:
            config_path: Path to config file. If None, looks for:
                1. --config CLI argument
                2. config.toml in current directory
                3. ~/.config/vocal-analyzer/config.toml
                4. Falls back to defaults
        """
        self.features = {
            "vocal_extraction": True,
            "transcription": True,
            "range_analysis": True,
            "llm_analysis": True,
            "pitch_visualization": True,
        }

        self.extraction = {
            "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "extract_all_stems": False,
        }

        self.transcription = {
            "enabled": True,
            "max_file_size_mb": 25,
            "compression_bitrate": "64k",
        }

        self.analysis = {
            "llm_model": "gpt-4.1-nano",
            "fallback_on_error": True,
        }

        self.output = {
            "format": "markdown",
            "include_pitch_plot": True,
            "quiet_mode": False,
        }

        # Load from file if specified
        if config_path:
            self._load_config(config_path)
        else:
            # Try to find config in standard locations
            self._try_load_default_config()

    def _try_load_default_config(self):
        """Try to load config from standard locations."""
        search_paths = [
            Path("config.toml"),
            Path.home() / ".config" / "vocal-analyzer" / "config.toml",
        ]

        for path in search_paths:
            if path.exists():
                self._load_config(path)
                return

    def _load_config(self, config_path):
        """Load configuration from TOML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

        # Update features
        if "features" in config_data:
            self.features.update(config_data["features"])

        # Update extraction settings
        if "extraction" in config_data:
            self.extraction.update(config_data["extraction"])

        # Update transcription settings
        if "transcription" in config_data:
            self.transcription.update(config_data["transcription"])

        # Update analysis settings
        if "analysis" in config_data:
            self.analysis.update(config_data["analysis"])

        # Update output settings
        if "output" in config_data:
            self.output.update(config_data["output"])

    def is_enabled(self, feature):
        """Check if a feature is enabled.

        Args:
            feature: Feature name (e.g., 'transcription', 'llm_analysis')

        Returns:
            bool: True if feature is enabled
        """
        return self.features.get(feature, True)
