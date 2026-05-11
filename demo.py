import os
import ollama
import textgrad as tg
from textgrad import Variable, TGD
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('OLLAMA_API_KEY')

class OllamaLLM:
    def __init__(self, model='gemma3:4b', host='http://localhost:11434', api_key=None):
        self.model = model
        self.api_key = api_key
        self.host = host

    def __call__(self, prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})

        response = ollama.chat(model=self.model, messages=messages)
        return response['message']['content']


def create_textgrad_model(model='gemma3:4b', api_key=None):
    llm = OllamaLLM(model=model, api_key=api_key)
    feedback_llm = OllamaLLM(model='gpt-oss:120b-cloud', api_key=api_key)
    tg.set_backward_engine(feedback_llm, override=True)
    return tg.BlackboxLLM(llm)


def run_textgrad(prompt_text, system_prompt, loops=1, model='gemma3:4b', api_key=None, loss_prompt="Evaluate this answer. It should be factual, clear, and directly answer the question."):
    if loops < 1:
        loops = 1

    textgrad_model = create_textgrad_model(model=model, api_key=api_key)
    prompt = Variable(prompt_text, requires_grad=True, role_description='Prompt for LLM')
    system_prompt_var = Variable(system_prompt, requires_grad=True, role_description='system prompt')
    optimizer = TGD(parameters=[prompt])

    answer_text = ''
    for _ in range(loops):
        answer = textgrad_model(system_prompt_var + prompt)
        answer_text = answer.value
        loss_fn = tg.TextLoss(loss_prompt)
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return answer_text


if __name__ == '__main__':
    result = run_textgrad(
        prompt_text='Teach me about quantum computing',
        system_prompt='The explanation must be clear and beginner-friendly.',
        loops=1,
        model='gemma3:4b',
        api_key=API_KEY,
        loss_prompt='Evaluate this answer. It should be factual, clear, and directly answer the question.'
    )
    print(result)
