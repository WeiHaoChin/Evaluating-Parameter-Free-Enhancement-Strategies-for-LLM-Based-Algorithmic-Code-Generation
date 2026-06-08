import os
import textgrad as tg
from textgrad import Variable, TGD
from dotenv import load_dotenv
from llm_clients import OllamaLLM, GoogleGenerativeAI

import time
import random

load_dotenv()
API_KEY = os.getenv('OLLAMA_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def create_textgrad_model(textGradModel, model, api_key=None):
    llm = None
    feedback_llm = None

    if model.startswith('gemini-'):
        llm = GoogleGenerativeAI(model=model, api_key=api_key or GOOGLE_API_KEY)
    elif model.startswith('gemma3:') or model.startswith('gpt-oss:') or model.startswith('deepseek-'):
        llm = OllamaLLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type for main LLM: {model}")
    
    if textGradModel.startswith('gemini-'):
        feedback_llm = GoogleGenerativeAI(model=textGradModel, api_key=api_key or GOOGLE_API_KEY)
    elif textGradModel.startswith('gemma3:') or textGradModel.startswith('gpt-oss:') or textGradModel.startswith('deepseek-'):
        feedback_llm = OllamaLLM(model=textGradModel, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model type for TextGrad LLM: {textGradModel}")

    tg.set_backward_engine(feedback_llm, override=True)
    return tg.BlackboxLLM(llm)


def run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None):
    """Generator version that yields events for streaming."""
    if loops < 1:
        loops = 1

    textgrad_model = create_textgrad_model(textGradModel=textGradModel, model=model, api_key=api_key)
    prompt = Variable(prompt_text, requires_grad=False, role_description='The user input/question provided to the model. This is fixed and should not be modified.')
    system_prompt_var = Variable(system_prompt, requires_grad=True, role_description="The system prompt that defines the model's behavior and instructions. Optimize this to improve the quality, accuracy, and clarity of the model's responses.")
    optimizer = TGD(parameters=[system_prompt_var])

    answer_text = ''
    for loop_idx in range(loops):
        # Send iteration start event with original prompt
        yield {
            'type': 'iteration_start',
            'loop': loop_idx + 1,
            'original_prompt': prompt.value
        }

        # Get LLM response
        answer = textgrad_model(system_prompt_var + prompt)
        answer_text = answer.value
        print(f"--- Iteration {loop_idx+1} ---")
        print(f"Response: {answer_text}\n")
        
        # Send LLM response event
        yield {
            'type': 'llm_response',
            'data': answer_text,
            'loop': loop_idx + 1
        }

        # Get critic feedback (loss)
        loss_fn = tg.TextLoss(loss_prompt)
        loss = loss_fn(answer)
        critic_response = str(loss)
        print(f"--- Iteration {loop_idx+1} ---")
        print(f"Critic Response: {critic_response}\n")
        
        # Send critic feedback event
        yield {
            'type': 'critic_feedback',
            'data': critic_response,
            'loop': loop_idx + 1
        }

        # Optimize
        loss.backward()
        optimizer.step()
        
        # Send updated prompt event update system prompt after optimization
        updated_prompt = system_prompt_var.value
        yield {
            'type': 'prompt_updated',
            'original': prompt_text if loop_idx == 0 else None,
            'updated': updated_prompt,
            'loop': loop_idx + 1
        }
        print(f"Updated System Prompt: {updated_prompt}\n")

        optimizer.zero_grad()

        # Send iteration complete event
        yield {
            'type': 'iteration_complete',
            'loop': loop_idx + 1
        }

    # Send final complete event with answer
    yield {
        'type': 'complete',
        'answer': answer_text
    }


def run_textgrad_sync(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None):
    """Synchronous version that collects all events and returns final answer (for backward compatibility)."""
    answer_text = ''
    for event in run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops, api_key):
        if event['type'] == 'complete':
            answer_text = event['answer']
    return answer_text


if __name__ == '__main__':
    result = run_textgrad_sync(
        prompt_text='Teach me about graph theory',
        system_prompt='The explanation must be clear and beginner-friendly. Do not question the user or ask for clarification. Just provide the best answer you can.',
        loops=1,
        textGradModel='gpt-oss:120b-cloud',
        model='gemma3:4b',
        api_key=API_KEY,
        loss_prompt='Evaluate this answer. It should be factual, clear, and directly answer the question.'
    )
    print(result)
