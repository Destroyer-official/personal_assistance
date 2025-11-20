import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent
from agents.voice_agent import VoiceAgent
# Vision is imported on demand

def main():
    print("\n🚀 STARTING JARVIS [PRODUCTION]...")

    brain = LLMAgent()
    # Uncomment next line to enable voice
    # voice = VoiceAgent()

    chat_history = [{"role": "system", "content": "You are Jarvis, an AI assistant."}]
    brain.load("friend")

    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break

            # --- VISION ---
            if user_input.lower().startswith("look at "):
                image_path = user_input[8:].strip().strip('"')
                from agents.vision_agent import VisionAgent
                eyes = VisionAgent()
                desc = eyes.analyze("Describe this.", image_path)
                print(f"\n[VISION]: {desc}")
                chat_history.append({"role": "user", "content": f"Analyze image: {image_path}"})
                chat_history.append({"role": "assistant", "content": desc})
                brain.load("friend")
                continue

            # --- ROUTER ---
            role = "friend"
            if any(x in user_input.lower() for x in ["code", "python", "html"]): role = "coder"
            elif any(x in user_input.lower() for x in ["math", "solve", "logic"]): role = "thinker"

            brain.load(role)

            # --- CHAT ---
            chat_history.append({"role": "user", "content": user_input})
            response = brain.chat(chat_history)
            chat_history.append({"role": "assistant", "content": response})

            # --- SPEAK ---
            # if voice: voice.speak(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
