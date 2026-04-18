# textgrad_gemini.py


import requests,ollama
from textgrad import Variable, TGD
import textgrad as tg

class OllamaLLM:
    def __init__(self, model="gemma3:4b", host="http://localhost:11434",api_key=None):
        self.model = model
        self.api_key = api_key
        if(api_key==None):
            self.url = f"{host}/api/generate"
        else:
            self.url=f"https://ollama.com/api/generate"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        """Generates LLM output via Ollama"""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})
        
        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']


# -----------------------------
# 2️⃣ Initialize LLM & prompt
# -----------------------------
with open("secret.txt", "r", encoding="utf-8") as file:
    API_KEY = file.read()  # my_text now contains all the text from the file
#llm = GeminiLLM(API_KEY, model="gemini-2.5-flash")

llm = OllamaLLM()  # Using Ollama for testing without API calls
llm1= OllamaLLM(model="gpt-oss:120b-cloud",api_key=API_KEY)
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
    print("===Model Output ===")
    print(answer.value)
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
print("===Final Output ===")
answer = model(loss_system_prompt+prompt)
print(answer.value)
