import sys
import os
import time

# --- 1. FIX PATHS (CRITICAL) ---
# Get the folder where this script lives (.../test)
current_test_folder = os.path.dirname(os.path.abspath(__file__))
# Get the project root folder (.../personal_assistance)
project_root = os.path.dirname(current_test_folder)

# Add project root to Python path so we can import 'config' and 'agents'
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. NOW IMPORT MODULES ---
try:
    import config
    from agents.vision_agent import VisionAgent
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"   Debug: Looked in {project_root}")
    sys.exit(1)

def main():
    print("👁️ TESTING VISION AGENT (Qwen2-VL)...")

    # Define path to image (a.png in the test folder)
    image_path = os.path.join(current_test_folder, "a.png")

    # Verify image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at: {image_path}")
        print("   👉 Please put 'a.png' inside the 'test' folder.")
        return

    # Initialize Agent
    print("   • Initializing Agent...", end="", flush=True)
    try:
        agent = VisionAgent()
        print(" Ready.")
    except Exception as e:
        print(f"\n❌ Failed to load VisionAgent: {e}")
        return

    print(f"\n📸 Analyzing: {os.path.basename(image_path)}")
    start = time.time()

    # Run Analysis
    # The agent now handles the 512px resize automatically for speed
    desc = agent.analyze("Describe this image briefly.", image_path)

    duration = time.time() - start

    print(f"\n📝 RESULT: {desc}")
    print(f"⏱️ Time: {duration:.2f}s")

    if duration > 10:
        print("⚠️ STATUS: SLOW (Check 'vision_agent.py' for '_optimize_image' resize logic)")
    else:
        print("🚀 STATUS: OPTIMAL SPEED")

if __name__ == "__main__":
    main()
