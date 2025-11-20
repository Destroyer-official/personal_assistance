import sounddevice as sd
from kokoro_onnx import Kokoro
import config
import os

class VoiceAgent:
    def __init__(self):
        print("🔊 Initializing Voice...", end="", flush=True)
        try:
            self.kokoro = Kokoro(config.VOICE["model"], config.VOICE["voices"])
            print(" Ready.")
        except:
            print("❌ Voice Failed.")
            self.kokoro = None

    def speak(self, text):
        if not self.kokoro or not text: return
        try:
            samples, sample_rate = self.kokoro.create(
                text, voice=config.VOICE["speaker"], speed=1.0, lang="en-us"
            )
            sd.play(samples, sample_rate)
            sd.wait()
        except:
            pass
