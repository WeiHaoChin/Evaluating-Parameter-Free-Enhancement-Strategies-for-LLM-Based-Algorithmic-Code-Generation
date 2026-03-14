# textgrad_ollama_demo_fixed.py
import requests
import textgrad as tg
from textgrad import Variable, TextLoss, TGD

# 1️⃣ Ollama wrapper (callable only)
class OllamaLLM:
    def __init__(self, model="llama3", host="http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def __call__(self, prompt):
        """Called by TextGrad to generate output"""
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": str(prompt),
                "stream": False
            }
        )
        data = response.json()
        return data["response"]

# 2️⃣ Initialize prompt
prompt = Variable(
    "Explain Dijkstra's algorithm simply and clearly for beginners.",
    requires_grad=True,
    role_description="Prompt to LLM"
)

# 3️⃣ Loss function
loss_fn = TextLoss("The explanation should be simple, correct, and beginner-friendly.")

# 4️⃣ Initialize OllamaLLM and optimizer
model = OllamaLLM("llama3")
optimizer = TGD([prompt])

# 5️⃣ TextGrad optimization loop
print("Starting prompt optimization...\n")
for step in range(5):
    # Generate output
    response = model(prompt)
    
    # Compute loss
    loss = loss_fn(response)
    
    # Backpropagate
    loss.backward()
    
    # Update prompt
    optimizer.step()
    
    print(f"Step {step+1} - Updated prompt:\n{prompt.value}\n")

# 6️⃣ Final LLM output
final_output = model(prompt)
print("=== Final LLM Output ===")
print(final_output)