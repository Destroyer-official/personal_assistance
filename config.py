import os

# --- SYSTEM PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "my_jarvis_models")

# --- MODEL PATHS ---
AGENTS = {
    "coder": {
        "path": os.path.join(MODELS_DIR, "coder", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        "context": 4096,
        "gpu_layers": -1
    },
    "thinker": {
        "path": os.path.join(MODELS_DIR, "thinker", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
        "context": 4096,
        "gpu_layers": -1
    },
    "friend": {
        "path": os.path.join(MODELS_DIR, "friend", "smollm2-1.7b-instruct-q4_k_m.gguf"),
        "context": 8192,
        "gpu_layers": -1
    },
    "vision": {
        "path": os.path.join(MODELS_DIR, "vision", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
        "projector": os.path.join(MODELS_DIR, "vision", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf"),
        "context": 2048,
        "gpu_layers": -1
    }
}

# --- VOICE ---
VOICE = {
    "model": os.path.join(MODELS_DIR, "voice", "kokoro-v0_19.onnx"),
    "voices": os.path.join(MODELS_DIR, "voice", "voices.bin"),
    "speaker": "af_sarah"
}
