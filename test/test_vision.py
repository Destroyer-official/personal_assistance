import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.vision_agent import VisionAgent

def main():
    print("👁️ TESTING VISION AGENT (Qwen2-VL)...")

    # Use your local a.png
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a.png")

    if not os.path.exists(image_path):
        print(f"❌ Error: {image_path} not found. Please put a.png in the main folder.")
        return

    agent = VisionAgent()

    print(f"\n📸 Analyzing: {image_path}")
    start = time.time()

    # This triggers the resize logic + inference
    desc = agent.analyze("Describe this image briefly.", image_path)

    duration = time.time() - start
    print(f"\n⏱️ Total Vision Time: {duration:.2f}s")

    if duration > 10:
        print("⚠️ STATUS: TOO SLOW (Check if 512px resize is active)")
    else:
        print("🚀 STATUS: OPTIMAL SPEED")

if __name__ == "__main__":
    main()
