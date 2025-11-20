import sys
import os

# Allow importing from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.llm_agent import LLMAgent
# Vision Agent loaded on demand

def main():
    print("\n🚀 STARTING JARVIS MODULAR SYSTEM...")
    print("   (Speed Optimized: 512px Vision)")

    brain = LLMAgent()

    # Persistent Chat History
    chat_history = [
        {"role": "system", "content": "You are Jarvis, a helpful and intelligent AI assistant."}
    ]

    brain.load("friend")

    while True:
        try:
            user_input = input("\nYOU: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break

            # --- VISION COMMAND ---
            if user_input.lower().startswith("look at "):
                image_path = user_input[8:].strip().strip('"')

                # 1. Load Vision
                from agents.vision_agent import VisionAgent
                eyes = VisionAgent()

                # 2. Analyze (Now fast!)
                description = eyes.analyze("Describe this image.", image_path)
                print(f"[VISION]: {description}")

                # 3. INJECT INTO MEMORY (The Fix)
                # We pretend the User asked to analyze, and the Assistant (Jarvis) answered.
                chat_history.append({"role": "user", "content": f"Analyze this image: {image_path}"})
                chat_history.append({"role": "assistant", "content": description})

                # 4. Switch back to Friend immediately so we are ready to chat
                brain.load("friend")
                continue

            # --- NORMAL CHAT ---
            role = "friend"
            if any(x in user_input.lower() for x in ["code", "python", "html"]): role = "coder"
            elif any(x in user_input.lower() for x in ["math", "solve", "logic"]): role = "thinker"

            brain.load(role)

            # Add user message
            chat_history.append({"role": "user", "content": user_input})

            # Generate Answer
            response = brain.chat(chat_history)

            # Save answer
            chat_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
