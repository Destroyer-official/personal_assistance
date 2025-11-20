import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent

def main():
    print("\n🚀 STARTING JARVIS [FINAL BUILD]...")
    print("   • Hardware: Ryzen 3 + Vega 8 (Optimized)")

    try:
        brain = LLMAgent()
        # Initialize Chat History
        chat_history = [{"role": "system", "content": "You are Jarvis, a helpful AI assistant."}]
        brain.load("friend")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        return

    print("\n✅ System Online. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break

            # --- VISION COMMAND ---
            if user_input.lower().startswith("look at "):
                target = user_input[8:].strip().strip('"')
                if not os.path.isabs(target):
                    target = os.path.join(os.getcwd(), target)

                if os.path.exists(target):
                    from agents.vision_agent import VisionAgent
                    eyes = VisionAgent()
                    desc = eyes.analyze("Describe this image briefly.", target)
                    print(f"[VISION]: {desc}")

                    # Inject Memory
                    chat_history.append({"role": "user", "content": f"I showed you an image at {target}."})
                    chat_history.append({"role": "assistant", "content": f"I saw the image. Description: {desc}"})

                    brain.load("friend") # Switch back to chat
                    continue
                else:
                    print(f"❌ File not found: {target}")
                    continue

            # --- ROUTING ---
            role = "friend"
            if any(x in user_input.lower() for x in ["code", "python", "script", "html"]): role = "coder"
            elif any(x in user_input.lower() for x in ["math", "solve", "logic"]): role = "thinker"

            brain.load(role)

            # --- CHAT ---
            chat_history.append({"role": "user", "content": user_input})
            response = brain.chat(chat_history)
            chat_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
