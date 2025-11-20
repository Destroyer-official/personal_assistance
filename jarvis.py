import sys
import os

# Allow importing from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent
# Vision and Voice are imported on demand to start fast

def main():
    print("\n🚀 STARTING JARVIS [PRODUCTION]...")
    print("   • Architecture: Modular Multi-Agent")
    print("   • Vision Mode: 512px (Fast)")
    print("   • Hardware: Ryzen 3 + Vega 8 (2GB VRAM)")

    # 1. Initialize Core
    try:
        brain = LLMAgent()
        # 2. Initialize Chat History (The Shared Memory)
        chat_history = [{"role": "system", "content": "You are Jarvis, an intelligent and helpful AI assistant."}]

        # 3. Start with Friend Agent
        brain.load("friend")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        input("Press Enter to exit...")
        return

    print("\n✅ System Online. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break

            # --- COMMAND: VISION ---
            # Detects "look at file.png"
            if user_input.lower().startswith("look at "):
                image_path = user_input[8:].strip().strip('"')

                # Check file first
                if not os.path.exists(image_path):
                    print(f"❌ File not found: {image_path}")
                    continue

                # Load Vision
                from agents.vision_agent import VisionAgent
                eyes = VisionAgent()

                # Analyze
                desc = eyes.analyze("Describe this image briefly.", image_path)
                print(f"[VISION]: {desc}")

                # INJECT MEMORY: This fixes the "Amnesia"
                # We tell the chat agent that IT saw the image
                chat_history.append({"role": "user", "content": f"I am showing you an image at {image_path}. What is it?"})
                chat_history.append({"role": "assistant", "content": f"(I looked at the image) {desc}"})

                # Switch back to Friend (Chat mode)
                brain.load("friend")
                continue

            # --- COMMAND: ROUTER ---
            role = "friend"
            # Keywords to trigger specific agents
            if any(x in user_input.lower() for x in ["code", "python", "script", "html", "css", "error"]):
                role = "coder"
            elif any(x in user_input.lower() for x in ["math", "solve", "logic", "calculate", "proof", "why"]):
                role = "thinker"

            brain.load(role)

            # Update System Prompt for context
            if role == "coder":
                sys_p = "You are an expert coding assistant. Write clean, efficient code."
            elif role == "thinker":
                sys_p = "You are a deep reasoning engine. Think step-by-step."
            else:
                sys_p = "You are Jarvis, a helpful and friendly AI assistant."

            # Update system prompt in history (keeps context fresh)
            if chat_history[0]["role"] == "system":
                chat_history[0]["content"] = sys_p

            # Add User Input
            chat_history.append({"role": "user", "content": user_input})

            # Generate Answer
            response = brain.chat(chat_history)

            # Save Answer
            chat_history.append({"role": "assistant", "content": response})

            # Speak (Optional)
            # from agents.voice_agent import VoiceAgent
            # mouth = VoiceAgent()
            # mouth.speak(response)

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"\n❌ Runtime Error: {e}")

if __name__ == "__main__":
    main()
