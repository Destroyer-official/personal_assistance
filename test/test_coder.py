import time
import os
import sys
import psutil

# Setup paths to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_agent import LLMAgent

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    print("👨‍💻 TESTING CODER AGENT (Qwen 2.5 1.5B)...")

    start_ram = get_memory_usage()
    print(f"   [1] Initial RAM: {start_ram:.2f} MB")

    # 1. Load
    start_time = time.time()
    bot = LLMAgent()
    bot.load("coder")
    load_time = time.time() - start_time

    loaded_ram = get_memory_usage()
    print(f"   [2] Model Loaded in {load_time:.2f}s")
    print(f"   [3] RAM Usage: {loaded_ram - start_ram:.2f} MB (Model + Context)")

    # 2. Generate
    prompt = "Write a Python function to calculate the Fibonacci sequence."
    print(f"\n   [4] Prompt: '{prompt}'")
    print("   [5] Generating...")

    gen_start = time.time()
    history = [{"role": "user", "content": prompt}]
    response = bot.chat(history)
    gen_time = time.time() - gen_start

    # 3. Stats
    tokens = len(response.split()) * 1.3 # Rough estimate
    tps = tokens / gen_time

    print(f"\n   [6] Generation Time: {gen_time:.2f}s")
    print(f"   [7] Estimated Speed: {tps:.2f} tokens/sec")
    print(f"   [8] Verdict: {'🚀 FAST' if tps > 15 else '⚠️ SLOW'}")

if __name__ == "__main__":
    main()
