import os
import time
import sys
import json
import platform
import wave
import struct
import random
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "my_jarvis_models")

# Define the roster
AGENTS = {
    "FRIEND (SmolLM2)": {
        "path": os.path.join(MODELS_DIR, "friend", "smollm2-1.7b-instruct-q4_k_m.gguf"),
        "type": "llm"
    },
    "CODER (Qwen 2.5)": {
        "path": os.path.join(MODELS_DIR, "coder", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        "type": "llm"
    },
    "THINKER (DeepSeek)": {
        "path": os.path.join(MODELS_DIR, "thinker", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
        "type": "llm"
    },
    "VISION (Qwen2-VL)": {
        "path": os.path.join(MODELS_DIR, "vision", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        "type": "vision"
    },
    "EARS (Whisper)": {
        "path": "distil-small.en", # Handled by library
        "type": "audio_in"
    },
    "VOICE (Kokoro)": {
        "model": os.path.join(MODELS_DIR, "voice", "kokoro-v0_19.onnx"),
        "voices": os.path.join(MODELS_DIR, "voice", "voices.bin"),
        "type": "audio_out"
    }
}

results = []

def create_dummy_image():
    """Creates a small black image for vision testing"""
    img_path = "test_image.png"
    Image.new('RGB', (100, 100), color='red').save(img_path)
    return img_path

def create_dummy_audio():
    """Creates 1 second of silence for audio testing"""
    wav_path = "test_audio.wav"
    with wave.open(wav_path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        data = struct.pack('<h', 0) * 16000
        f.writeframes(data)
    return wav_path

def log(name, status, detail):
    print(f"[{name.ljust(18)}] {status.ljust(10)} | {detail}")
    results.append({"model": name, "status": status, "detail": detail})

def test_llm(name, path):
    try:
        from llama_cpp import Llama
        start_load = time.time()

        # Force GPU Load
        llm = Llama(
            model_path=path,
            n_gpu_layers=-1, # TRY FULL GPU
            n_ctx=1024,
            verbose=False
        )
        load_time = time.time() - start_load

        # Speed Test
        start_gen = time.time()
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Count to 10."}],
            max_tokens=50
        )
        gen_time = time.time() - start_gen

        tokens = output['usage']['completion_tokens']
        tps = tokens / gen_time

        # Heuristic to check GPU: if speed > 10 t/s on this hardware, it's likely GPU
        device_status = "🚀 GPU (Vega 8)" if tps > 10 else "⚠️ CPU (Slow)"

        log(name, "PASSED", f"{tps:.2f} tokens/sec ({device_status})")

    except Exception as e:
        log(name, "FAILED", str(e))

def test_vision(name, path):
    try:
        from llama_cpp import Llama, LlamaChatCompletionHandler
        img_path = create_dummy_image()

        handler = LlamaChatCompletionHandler.from_pretrained(
            repo_id="bartowski/Qwen2-VL-2B-Instruct-GGUF",
            filename="Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
        )

        llm = Llama(
            model_path=path,
            chat_handler=handler,
            n_gpu_layers=-1,
            n_ctx=1024,
            verbose=False
        )

        start = time.time()
        llm.create_chat_completion(
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "What color is this?"},
                    {"type": "image_url", "image_url": {"url": img_path}}
                ]}
            ]
        )
        duration = time.time() - start
        log(name, "PASSED", f"Processed image in {duration:.2f}s")
        os.remove(img_path)

    except Exception as e:
        log(name, "FAILED", str(e))

def test_ears(name):
    try:
        from faster_whisper import WhisperModel
        audio_path = create_dummy_audio()

        # Check if CUDA is visible
        model = WhisperModel("distil-small.en", device="auto", compute_type="int8")

        start = time.time()
        segments, _ = model.transcribe(audio_path)
        list(segments) # Consume generator
        duration = time.time() - start

        used_device = model.model.device
        log(name, "PASSED", f"Transcribed in {duration:.2f}s on {used_device.upper()}")
        os.remove(audio_path)
    except Exception as e:
        log(name, "FAILED", str(e))

def test_voice(name, model_path, voice_path):
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf # Verify we can generate audio array

        if not os.path.exists(model_path):
            raise FileNotFoundError("Kokoro model missing")

        kokoro = Kokoro(model_path, voice_path)

        start = time.time()
        # Generate raw audio samples
        samples, rate = kokoro.create("System check complete.", voice="af_sarah", speed=1.0, lang="en-us")
        duration = time.time() - start

        log(name, "PASSED", f"Generated speech in {duration:.2f}s")

    except Exception as e:
        log(name, "FAILED", str(e))

def main():
    print(f"\n🧪 STARTING JARVIS SYSTEM DIAGNOSTIC")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print("="*65)
    print(f"{'MODEL':<20} | {'STATUS':<10} | {'PERFORMANCE / DETAILS'}")
    print("-" * 65)

    for name, info in AGENTS.items():
        if info["type"] == "llm":
            test_llm(name, info["path"])
        elif info["type"] == "vision":
            test_vision(name, info["path"])
        elif info["type"] == "audio_in":
            test_ears(name)
        elif info["type"] == "audio_out":
            test_voice(name, info["model"], info["voices"])

    print("-" * 65)
    print("✨ Diagnostic Complete.")

if __name__ == "__main__":
    main()
