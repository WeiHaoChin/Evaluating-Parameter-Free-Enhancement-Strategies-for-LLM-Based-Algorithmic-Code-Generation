# textgrad_gemini.py


import requests,ollama
from textgrad import Variable, TGD
import textgrad as tg
# -----------------------------
# 1️⃣ Gemini API Wrapper
# -----------------------------
class GeminiLLM:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }# If TextGrad provides a system_prompt, add it to the 'system_instruction' field
        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": str(system_prompt)}]
            }
        response = requests.post(self.endpoint, headers=headers, json=payload)
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    
class OllamaLLM:
    def __init__(self, model="gemma3:4b", host="http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/generate"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        """Generates LLM output via Ollama"""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})
        
        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']
        if system_prompt:
            prompt = system_prompt
        response = ollama.generate(model=self.model, prompt=f"{prompt}")

        return response['response']
class OllamaLoss:
    def __init__(self, instruction, model):
        self.instruction = instruction
        self.model = model
"""
    def __call__(self, text):
        Return a mock 'loss' object with backward()
        # Here we can actually call the LLM for critique if we want
        critique = self.model(f"{text}\n\nInstruction: {self.instruction}")
        # Simulate a loss object
        class Loss:
            def backward(self_inner):
                nonlocal text
                # You could implement gradient suggestions here
                # For simplicity, we'll just append a suggestion
                text += " (simplified for beginners)"
        return Loss()
"""
# -----------------------------
# 3️⃣ Custom Loss for TextGrad
# -----------------------------
class GeminiLoss:
    def __init__(self, instruction, model):
        self.instruction = instruction
        self.model = model

    def __call__(self, text,):
        # Call Gemini to get critique or output
        output = self.model(f"{text}\nInstruction: {self.instruction}")
        # Simulate a "loss" object for TextGrad
        class Loss:
            def backward(self_inner):
                nonlocal text
                # Here you can append, adjust, or tweak the prompt
                # Example: add a hint for clarity
                text += " (simplified for beginners)"
        return Loss()


# -----------------------------
# 2️⃣ Initialize LLM & prompt
# -----------------------------
with open("secret.txt", "r", encoding="utf-8") as file:
    API_KEY = file.read()  # my_text now contains all the text from the file
#llm = GeminiLLM(API_KEY, model="gemini-2.5-flash")

llm = OllamaLLM()  # Using Ollama for testing without API calls
llm1= OllamaLLM(model="gemma3:4b")
# 1️⃣ Set the engine used for gradient feedback
tg.set_backward_engine(llm1, override=True)
# 3️⃣ Create the model (forward LLM)
model = tg.BlackboxLLM(llm)

#loss_fn = GeminiLoss("Make the explanation clear and beginner-friendly.", llm)
#loss_fn = OllamaLoss("Make the explanation clear and beginner-friendly.", llm)
loss_system_prompt = tg.Variable("""The explanation must be clear and beginner-friendly.""",
                                 requires_grad=True,
                                 role_description="system prompt")
prompt = tg.Variable(
    "Teach me about the quantum computing",
    requires_grad=True,
    role_description="Prompt for LLM"
)
#answer.set_role_description("LLM output for the given prompt")
optimizer = tg.TGD(parameters=[prompt])
# define the loss function (llm critique)
#loss_fn = tg.TextLoss(loss_system_prompt)
# 1️⃣ Ollama wrapper




# -----------------------------
# 4️⃣ Optimizer
# -----------------------------
#optimizer = TGD([prompt],engine=llm)

# -----------------------------
# 5️⃣ Optimization loop
# -----------------------------
for step in range(5):
    #output = llm(prompt.value)
    print("=== Starting Prompt ===")
    print(f"Prompt: {prompt.value} System Prompt: {loss_system_prompt.value}\n")
    answer = model(loss_system_prompt+prompt)
    loss_fn = tg.TextLoss(
            "Evaluate this answer. It should be factual, clear, and directly answer the question."
        )
    loss = loss_fn(answer)
    print("=== Critique Output ===")
    print(loss.value)  # This will print the critique or output from the loss function
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #time.sleep(5)  # Sleep to respect API rate limits
    print("=== Updated Prompt ===")
    print(f"Step {step+1} - Updated prompt: {prompt.value}\n")

# -----------------------------
# 6️⃣ Final Output
# -----------------------------
'''
final_output = llm(prompt.value)
print("=== Final Gemini Output ===")
print(final_output)
'''