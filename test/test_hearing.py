import time
import os
import sys
import wave
import struct

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.hearing_agent import HearingAgent

def create_dummy_audio(filename):
    # Create 1 sec of silence
    with wave.open(filename, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack('<h', 0) * 16000)

def main():
    print("👂 TESTING HEARING AGENT (Distil-Whisper)...")

    wav_file = "test_audio.wav"
    create_dummy_audio(wav_file)

    agent = HearingAgent()

    print("\n🎤 Listening to dummy audio file...")
    start = time.time()
    text = agent.listen(wav_file)
    duration = time.time() - start

    print(f"📝 Transcribed: '{text}' (Should be empty/silence)")
    print(f"⏱️ Processing Time: {duration:.2f}s")

    if os.path.exists(wav_file):
        os.remove(wav_file)

if __name__ == "__main__":
    main()
