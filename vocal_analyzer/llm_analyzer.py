import openai


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
        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano", messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
