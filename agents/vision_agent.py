from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
from io import BytesIO
import base64
import config
import os
import traceback
import time

class VisionAgent:
    def __init__(self, max_size: int = 448, fast_encode: bool = True):
        """
        Production Vision Agent for Ryzen 3 + Vega 8.

        Args:
            max_size: 448px (Optimal for Qwen2-VL speed).
            fast_encode: True = JPEG (Fast), False = WEBP (Small).
        """
        self.model = None
        self.max_size = max_size
        self.fast_encode = fast_encode

        # Cache to avoid re-processing the same image file
        self._cached_key = None
        self._cached_data_uri = None

    def _image_file_key(self, path: str):
        """Generate a unique key based on file path, modified time, and size."""
        try:
            st = os.stat(path)
            return (path, int(st.st_mtime), st.st_size)
        except Exception:
            return None

    def _hyper_speed_resize_and_encode(self, image_path: str) -> str | None:
        # 1. Check Cache
        key = self._image_file_key(image_path)
        if key and key == self._cached_key and self._cached_data_uri:
            print("   (⚡ Cache Hit: Image already processed)")
            return self._cached_data_uri

        start = time.perf_counter()
        try:
            with Image.open(image_path) as img:
                # 2. Resize (Downscale only)
                if img.width > self.max_size or img.height > self.max_size:
                    img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)

                # 3. Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # 4. Encode
                if self.fast_encode:
                    fmt = "JPEG"
                    save_kwargs = {"quality": 75, "optimize": True}
                else:
                    fmt = "WEBP"
                    save_kwargs = {"quality": 80, "method": 4}

                # Write to buffer
                with BytesIO() as buf:
                    img.save(buf, format=fmt, **save_kwargs)
                    # .getvalue() is safe and creates a copy, preventing buffer errors
                    image_bytes = buf.getvalue()
                    b64_bytes = base64.b64encode(image_bytes)
                    data_uri = f"data:image/{fmt.lower()};base64,{b64_bytes.decode('ascii')}"

                    # Update Cache
                    self._cached_key = key
                    self._cached_data_uri = data_uri

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    print(f"   (⚡ Encoded: {fmt} | {len(b64_bytes)/1024:.1f} KB | {elapsed_ms:.2f}ms)")
                    return data_uri

        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            traceback.print_exc()
            return None

    def _ensure_model(self):
        """Lazy load model with Hardware Acceleration settings."""
        if self.model:
            return True, None

        print("\n👁️ Waking up Vision Core (Qwen2-VL)...")
        conf = config.AGENTS["vision"]

        if not os.path.exists(conf["projector"]):
            return False, f"❌ Error: Missing projector file at: {conf['projector']}"

        try:
            chat_handler = Llava15ChatHandler(clip_model_path=conf["projector"])

            # GPU & CPU Optimizations for Vega 8
            self.model = Llama(
                model_path=conf["path"],
                chat_handler=chat_handler,
                n_gpu_layers=-1,     # Full GPU Offload
                n_ctx=2048,          # 2k Context is plenty for vision
                n_threads=4,         # Use all 4 CPU cores for encoding tasks
                n_batch=512,         # Batch processing for speed
                flash_attn=True,     # Enable Flash Attention (Critical for speed)
                verbose=False
            )
            return True, None
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"❌ Vision Crash: {e}\n{tb}"

    def analyze(self, text: str, image_path: str, max_tokens: int = 150):
        # 1. Validate File
        if not os.path.exists(image_path):
            return "❌ Error: Image file not found."

        # 2. Load Model
        ok, err = self._ensure_model()
        if not ok: return err

        print(f"[VISION]: Scanning {os.path.basename(image_path)}...", end="", flush=True)

        # 3. Process Image (Cached)
        t0 = time.perf_counter()
        image_data = self._hyper_speed_resize_and_encode(image_path)
        if not image_data: return "❌ Could not process image."

        # 4. Inference
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ]

        try:
            print(" Analyzing...", end="", flush=True)
            response = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens)

            # 5. Defensive Extraction
            if not isinstance(response, dict): return f"❌ Unexpected response: {response}"

            choices = response.get("choices")
            if not choices: return "❌ No response generated."

            content = choices[0].get("message", {}).get("content")
            if not content: return "❌ Empty response content."

            elapsed = (time.perf_counter() - t0) * 1000
            print(f" Done. (Total: {elapsed/1000:.2f}s)")

            return content

        except Exception as e:
            tb = traceback.format_exc()
            return f"❌ Analysis Failed: {e}\n{tb}"
