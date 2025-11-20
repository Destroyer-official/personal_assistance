from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import io
import base64
import config
import os

class VisionAgent:
    def __init__(self):
        self.model = None

    def _optimize_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                # Resize to 512px for SPEED
                max_size = 512
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size))

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=70)
                base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return f"data:image/jpeg;base64,{base64_str}"
        except Exception as e:
            print(f"❌ Image Error: {e}")
            return None

    def analyze(self, text, image_path):
        if not self.model:
            print("\n👁️ Opening Eyes...", end="", flush=True)
            try:
                chat_handler = Llava15ChatHandler(clip_model_path=config.AGENTS["vision"]["projector"])
                self.model = Llama(
                    model_path=config.AGENTS["vision"]["path"],
                    chat_handler=chat_handler,
                    n_gpu_layers=-1,
                    n_ctx=2048,
                    verbose=False
                )
                print(" Ready.")
            except Exception as e:
                return f"❌ Vision Crash: {e}"

        print(f"[VISION]: Scanning...", end="", flush=True)
        image_data = self._optimize_image(image_path)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ]

        response = self.model.create_chat_completion(messages=messages, max_tokens=200)
        return response["choices"][0]["message"]["content"]
