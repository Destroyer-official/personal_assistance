from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
from io import BytesIO
import base64
import config
import os
import traceback

class VisionAgent:
    def __init__(self):
        self.model = None

    def _hyper_speed_resize(self, image_path):
        """Resizes to 448px and converts to Base64 safely."""
        try:
            with Image.open(image_path) as img:
                # 1. Resize
                max_size = 448
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    print(f"   (⚡ Resize to {img.size}...)")

                # 2. Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # 3. Encode
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)

                # 4. Safe extraction (Fixes the "resize" crash)
                image_bytes = buffer.getvalue()
                b64 = base64.b64encode(image_bytes).decode("ascii")
                buffer.close()

                return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return None

    def _ensure_model(self):
        if self.model: return True

        print("\n👁️ Waking up Vision Core (Qwen2-VL)...")
        conf = config.AGENTS["vision"]

        if not os.path.exists(conf["projector"]):
            print(f"❌ Error: Missing projector: {conf['projector']}")
            return False

        try:
            chat_handler = Llava15ChatHandler(clip_model_path=conf["projector"])
            self.model = Llama(
                model_path=conf["path"],
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            return True
        except Exception as e:
            print(f"❌ Vision Crash: {e}")
            return False

    def analyze(self, text, image_path):
        if not os.path.exists(image_path):
            return "❌ Error: Image file not found."

        if not self._ensure_model():
            return "❌ Vision model failed to load."

        print(f"[VISION]: Scanning {os.path.basename(image_path)}...", end="", flush=True)

        image_data = self._hyper_speed_resize(image_path)
        if not image_data: return "❌ Could not process image."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ]

        try:
            print(" Analyzing...", end="", flush=True)
            response = self.model.create_chat_completion(messages=messages, max_tokens=150)
            content = response["choices"][0]["message"]["content"]
            print(" Done.")
            return content
        except Exception as e:
            return f"❌ Analysis Failed: {e}"
