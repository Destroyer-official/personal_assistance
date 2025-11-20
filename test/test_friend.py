import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_agent import LLMAgent

def main():
    print("💬 TESTING FRIEND AGENT (SmolLM2)...")

    bot = LLMAgent()
    bot.load("friend")

    prompt = "Tell me a short joke."
    print(f"\n❓ Asking: {prompt}\n")

    start = time.time()
    history = [{"role": "user", "content": prompt}]
    bot.chat(history)
    duration = time.time() - start

    print(f"\n⏱️ Response Time: {duration:.2f}s")

if __name__ == "__main__":
    main()
