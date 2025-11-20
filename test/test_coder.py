import time
import os
import sys
import psutil

# Fix imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_agent import LLMAgent

def main():
    print("👨‍💻 TESTING CODER AGENT...")

    try:
        bot = LLMAgent()
        start = time.time()
        bot.load("coder") # Correct method
        print(f"   [+] Load Time: {time.time() - start:.2f}s")

        prompt = "Write a Python function to calculate Fibonacci."
        print(f"   [+] Generating: {prompt}")

        history = [{"role": "user", "content": prompt}]
        bot.chat(history)

        print("✅ Test Passed")
    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    main()
