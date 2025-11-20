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
        """
        Resizes image to max 1024px and converts to Base64.
        This drastically speeds up the 'image slice encoded' phase.
        """
        try:
            with Image.open(image_path) as img:
                # Resize if too big (Speed Hack)
                max_size = 1024
                if img.width > max_size or img.height > max_size:
                    print(f"   (Resizing image from {img.size} to fit {max_size}px limit...)")
                    img.thumbnail((max_size, max_size))

                # Convert to RGB (fixes weird PNG transparency issues)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to memory as JPEG (Faster than PNG for AI)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return f"data:image/jpeg;base64,{base64_str}"
        except Exception as e:
            print(f"❌ Image Processing Error: {e}")
            return None

    def analyze(self, text, image_path):
        if not os.path.exists(image_path):
            return "❌ Error: Image file not found."

        # 1. Load Model (Lazy Loading)
        if not self.model:
            print("\n👁️ Opening Eyes (Loading Vision Model)...")
            model_path = config.AGENTS["vision"]["path"]
            mmproj_path = os.path.join(os.path.dirname(model_path), "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")

            if not os.path.exists(mmproj_path):
                return "❌ Error: Missing 'mmproj' file. Run fix_vision_download.py."

            try:
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                self.model = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_gpu_layers=-1, # Full GPU
                    n_ctx=2048,
                    verbose=False
                )
            except Exception as e:
                return f"❌ Vision Crash: {e}"

        # 2. Process Image
        print(f"[VISION]: Processing {os.path.basename(image_path)}...", end="", flush=True)
        image_data = self._optimize_image(image_path)

        if not image_data:
            return "❌ Failed to process image."

        # 3. Prepare Prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ]

        # 4. Generate
        try:
            print(" Analyzing...", end="", flush=True)
            response = self.model.create_chat_completion(messages=messages, max_tokens=300)
            answer = response["choices"][0]["message"]["content"]
            print(" Done.")
            return answer
        except Exception as e:
            return f"❌ Analysis Failed: {e}"
