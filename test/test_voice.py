import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.voice_agent import VoiceAgent

def main():
    print("🔊 TESTING VOICE AGENT (Kokoro)...")

    agent = VoiceAgent()
    if not agent.kokoro:
        print("❌ Voice model not loaded.")
        return

    text = "System diagnostics complete. All systems functional."
    print(f"\n🗣️ Speaking: '{text}'")

    start = time.time()
    agent.speak(text)
    duration = time.time() - start

    print(f"⏱️ Generation + Playback Time: {duration:.2f}s")
    print("(Did you hear the voice?)")

if __name__ == "__main__":
    main()
