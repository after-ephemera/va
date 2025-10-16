import openai
import logging


class LLMAnalyzer:
    """Analyze vocal style using OpenAI's GPT model."""

    def __init__(self, transcription, features):
        self.transcription = transcription
        self.features = features

    def analyze(self):
        """Craft a prompt and query the LLM for vocal style analysis."""
        # Convert NumPy values to Python floats for proper formatting
        min_pitch = float(self.features["min_pitch"])
        max_pitch = float(self.features["max_pitch"])
        mean_pitch = float(self.features["mean_pitch"])
        tempo = float(self.features["tempo"])

        # Get note information
        min_note = self.features.get("min_note", "N/A")
        max_note = self.features.get("max_note", "N/A")
        mean_note = self.features.get("mean_note", "N/A")

        prompt = (
            f"Analyze the vocal style of this song based on the following information:\n"
            f"- Tempo: {tempo:.2f} BPM\n"
        )

        if min_note != "N/A" and max_note != "N/A":
            prompt += (
                f"- Vocal range: {min_note} to {max_note} "
                f"({min_pitch:.2f} to {max_pitch:.2f} Hz)\n"
                f"- Average pitch: {mean_note} ({mean_pitch:.2f} Hz)\n"
            )
        else:
            prompt += (
                f"- Pitch range: {min_pitch:.2f} to {max_pitch:.2f} Hz\n"
                f"- Average pitch: {mean_pitch:.2f} Hz\n"
            )

        prompt += f"- Lyrics: {self.transcription}\n"

        if self.features["screaming"]:
            prompt += "- Includes extreme vocals, possibly screaming.\n"
        prompt += (
            "Provide a detailed analysis of the vocal style, considering these aspects. "
            "Focus on the musical characteristics, vocal techniques, and style elements."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-nano", messages=[{"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAI API error occurred: {e}")
            return self._generate_fallback_analysis()
        except Exception as e:
            logging.warning(f"Unexpected error during LLM analysis: {e}")
            return self._generate_fallback_analysis()

    def _generate_fallback_analysis(self):
        """Generate a basic analysis when LLM API is unavailable."""
        min_pitch = float(self.features["min_pitch"])
        max_pitch = float(self.features["max_pitch"])
        mean_pitch = float(self.features["mean_pitch"])
        tempo = float(self.features["tempo"])

        min_note = self.features.get("min_note", "N/A")
        max_note = self.features.get("max_note", "N/A")
        mean_note = self.features.get("mean_note", "N/A")

        analysis = "**Note: LLM analysis unavailable - providing basic technical analysis**\n\n"

        # Tempo analysis
        if tempo < 60:
            tempo_desc = "very slow"
        elif tempo < 90:
            tempo_desc = "slow"
        elif tempo < 120:
            tempo_desc = "moderate"
        elif tempo < 150:
            tempo_desc = "fast"
        else:
            tempo_desc = "very fast"

        analysis += f"**Tempo**: {tempo:.2f} BPM - This is a {tempo_desc} tempo.\n\n"

        # Pitch range analysis
        if min_note != "N/A" and max_note != "N/A":
            analysis += f"**Vocal Range**: {min_note} to {max_note} ({min_pitch:.2f} to {max_pitch:.2f} Hz)\n"
            analysis += f"**Average Pitch**: {mean_note} ({mean_pitch:.2f} Hz)\n\n"

            # Calculate range in semitones (approximate)
            range_hz = max_pitch - min_pitch
            if range_hz > 800:
                range_desc = "very wide"
            elif range_hz > 400:
                range_desc = "wide"
            elif range_hz > 200:
                range_desc = "moderate"
            else:
                range_desc = "narrow"

            analysis += f"This represents a {range_desc} vocal range.\n\n"
        else:
            analysis += f"**Pitch Range**: {min_pitch:.2f} to {max_pitch:.2f} Hz\n"
            analysis += f"**Average Pitch**: {mean_pitch:.2f} Hz\n\n"

        # Extreme vocals detection
        if self.features["screaming"]:
            analysis += (
                "**Vocal Style**: Contains extreme vocals or screaming elements.\n\n"
            )

        # Basic transcription analysis
        if self.transcription and len(self.transcription.strip()) > 0:
            word_count = len(self.transcription.split())
            analysis += (
                f"**Lyrics**: Contains {word_count} words of transcribed content.\n\n"
            )
        else:
            analysis += "**Lyrics**: No clear lyrics detected or transcribed.\n\n"

        analysis += "For a more detailed vocal style analysis, please ensure the LLM service is available."

        return analysis
