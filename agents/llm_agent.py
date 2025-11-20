from llama_cpp import Llama
import config

class LLMAgent:
    def __init__(self):
        self.current_role = None
        self.model = None

    def load_role(self, role):
        if self.current_role == role: return

        print(f"\n🔄 Switching Brain to: {role.upper()}...", end="", flush=True)
        if self.model: del self.model

        try:
            self.model = Llama(
                model_path=config.AGENTS[role]["path"],
                n_gpu_layers=config.AGENTS[role]["gpu_layers"],
                n_ctx=config.AGENTS[role]["context"],
                n_threads=4,
                verbose=False
            )
            self.current_role = role
            print(" Ready!")
        except Exception as e:
            print(f"\n❌ Failed to load {role}: {e}")

    def chat(self, history):
        # "history" is now a list of dictionaries: [{"role": "user", "content": "..."}]
        # Llama.cpp handles the formatting automatically
        try:
            response = self.model.create_chat_completion(messages=history, stream=True)

            print(f"[{self.current_role.upper()}]: ", end="", flush=True)
            full_text = ""
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    text = chunk["choices"][0]["delta"]["content"]
                    print(text, end="", flush=True)
                    full_text += text
            print()
            return full_text
        except Exception as e:
            return f"Error generating response: {e}"
