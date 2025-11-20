from llama_cpp import Llama
import config

class LLMAgent:
    def __init__(self):
        self.current_role = None
        self.model = None

    def load(self, role):
        if self.current_role == role: return

        print(f"\n🔄 Swapping to {role.upper()} Agent...", end="", flush=True)
        if self.model: del self.model

        try:
            self.model = Llama(
                model_path=config.AGENTS[role]["path"],
                n_gpu_layers=config.AGENTS[role]["gpu_layers"],
                n_ctx=config.AGENTS[role]["context"],
                verbose=False
            )
            self.current_role = role
            print(" Online!")
        except Exception as e:
            print(f"\n❌ Error loading {role}: {e}")

    def chat(self, history):
        if not self.model: return "❌ Brain not loaded."

        output = self.model.create_chat_completion(
            messages=history,
            stream=True,
            temperature=0.7
        )

        print(f"[{self.current_role.upper()}]: ", end="", flush=True)
        full_text = ""
        for chunk in output:
            if "content" in chunk["choices"][0]["delta"]:
                text = chunk["choices"][0]["delta"]["content"]
                print(text, end="", flush=True)
                full_text += text
        print()
        return full_text
