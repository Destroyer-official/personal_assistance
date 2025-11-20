import time
import os
import sys
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_agent import LLMAgent

def main():
    print("🧠 TESTING THINKER AGENT (DeepSeek R1)...")

    bot = LLMAgent()
    bot.load("thinker")

    # Logic Puzzle
    prompt = "If I have 3 apples and eat one, how many do I have? Explain your reasoning."
    print(f"\n❓ Query: {prompt}\n")

    start = time.time()
    history = [{"role": "user", "content": prompt}]
    response = bot.chat(history)
    duration = time.time() - start

    print(f"\n⏱️ Thought Process Time: {duration:.2f}s")

    if "2" in response or "two" in response:
        print("✅ Accuracy Check: PASSED")
    else:
        print("❌ Accuracy Check: FAILED")

if __name__ == "__main__":
    main()
