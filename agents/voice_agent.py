import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro
import config
import os

class VoiceAgent:
    def __init__(self):
        print("🔊 Initializing Voice...", end="", flush=True)
        if not os.path.exists(config.KOKORO_MODEL):
            print("\n⚠️ Voice models missing. Run download_voice.py (from previous steps).")
            self.kokoro = None
            return

        try:
            self.kokoro = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
            print(" Ready.")
        except Exception as e:
            print(f"\n❌ Voice Init Error: {e}")
            self.kokoro = None

    def speak(self, text):
        if not self.kokoro: return
        if not text: return

        # Generate audio (fast!)
        # 'af_sarah' is a good default female voice. Change to 'am_michael' for male.
        try:
            samples, sample_rate = self.kokoro.create(
                text, voice="af_sarah", speed=1.0, lang="en-us"
            )
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Speaking error: {e}")
