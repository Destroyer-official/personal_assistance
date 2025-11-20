from faster_whisper import WhisperModel
import config
import os

class HearingAgent:
    def __init__(self):
        print("👂 Initializing Ears...", end="", flush=True)
        # Use 'int8' for speed on CPU/GPU
        try:
            self.model = WhisperModel(config.WHISPER_MODEL_SIZE, device="auto", compute_type="int8")
            print(" Ready.")
        except Exception as e:
            print(f"\n❌ Whisper Init Error: {e}")

    def listen(self, audio_file_path):
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found."

        segments, info = self.model.transcribe(audio_file_path, beam_size=5)
        text = "".join([segment.text for segment in segments])
        return text.strip()
