import openai


class LLMAnalyzer:
    """Analyze vocal style using OpenAI's GPT model."""

    def __init__(self, transcription, features):
        self.transcription = transcription
        self.features = features

    def analyze(self):
        """Craft a prompt and query the LLM for vocal style analysis."""
        prompt = (
            f"Analyze the vocal style of this song based on the following information:\n"
            f"- Tempo: {self.features['tempo']:.2f} BPM\n"
            f"- Pitch range: {self.features['min_pitch']:.2f} to {self.features['max_pitch']:.2f} Hz\n"
            f"- Average pitch: {self.features['mean_pitch']:.2f} Hz\n"
            f"- Lyrics: {self.transcription}\n"
        )
        if self.features["screaming"]:
            prompt += "- Includes extreme vocals, possibly screaming.\n"
        prompt += (
            "Provide a detailed analysis of the vocal style, considering these aspects."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
