from llama_cpp import Llama
import config

class LLMAgent:
    def __init__(self):
        self.current_role = None
        self.model = None  # Unified variable name

    def load(self, role):
        """Swaps the active model in VRAM."""
        if self.current_role == role:
            return

        print(f"\n🔄 Swapping to {role.upper()} Agent...", end="", flush=True)

        # Unload previous model to free VRAM
        if self.model:
            del self.model
            self.model = None

        conf = config.AGENTS[role]
        try:
            self.model = Llama(
                model_path=conf["path"],
                n_gpu_layers=conf["gpu_layers"],
                n_ctx=conf["context"],
                verbose=False
            )
            self.current_role = role
            print(" Online!")
        except Exception as e:
            print(f"\n❌ Error loading {role}: {e}")
            self.current_role = None

    def chat(self, history):
        """Generates a response based on history."""
        if not self.model:
            print("❌ No model loaded!")
            return ""

        try:
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
        except Exception as e:
            return f"Error: {e}"
