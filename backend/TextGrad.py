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

def create_textgrad_model(textGradModel, model, system_prompt, api_key=None, textGrad_api_key=None,temperature=0.0):
    llm = None
    feedback_llm = None

    if model.startswith('gemini-'):
        llm = GoogleGenerativeAI(model=model, api_key=api_key or GOOGLE_API_KEY, temperature=temperature)
    elif model.startswith('gemma3:') or model.startswith('gpt-oss:') or model.startswith('deepseek-'):
        llm = OllamaLLM(model=model, api_key=api_key, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type for main LLM: {model}")
    
    if textGradModel.startswith('gemini-'):
        feedback_llm = GoogleGenerativeAI(model=textGradModel, api_key=textGrad_api_key or GOOGLE_API_KEY, temperature=temperature)
    elif textGradModel.startswith('gemma3:') or textGradModel.startswith('gpt-oss:') or textGradModel.startswith('deepseek-'):
        feedback_llm = OllamaLLM(model=textGradModel, api_key=textGrad_api_key, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model type for TextGrad LLM: {textGradModel}")
    # Set the backward engine (feedback LLM) that generates textual gradients
    # during TextGrad's optimization loop — analogous to backprop in neural networks.
    # override=True replaces any previously configured backward engine.
    tg.set_backward_engine(feedback_llm, override=True)
    # Wrap the main LLM as a BlackboxLLM node in TextGrad's computation graph.
    # "Blackbox" means TextGrad interacts with it purely via text in/out,
    # using the backward engine to critique and iteratively refine its outputs.
    return tg.BlackboxLLM(llm,system_prompt=system_prompt)


def run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None,textGrad_api_key=None,temperature=0.0):
    """Generator version that yields events for streaming."""
    if loops < 1:
        loops = 1

    prompt = Variable(prompt_text, requires_grad=False, role_description='The user input/question provided to the model. This is fixed and should not be modified.')
    system_prompt_var = Variable(system_prompt, requires_grad=True, role_description="The system prompt that defines the model's behavior and instructions. Optimize this to improve the quality, accuracy, and clarity of the model's responses.")
    textgrad_model = create_textgrad_model(textGradModel=textGradModel, model=model, system_prompt=system_prompt_var, api_key=api_key,textGrad_api_key=textGrad_api_key,temperature=temperature)
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
        answer = textgrad_model(prompt)
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
        yield {
            'type': 'prompt_updated',
            'original': prompt_text if loop_idx == 0 else None,
            'updated': system_prompt_var.value,
            'loop': loop_idx + 1
        }
        print(f"Updated System Prompt: {system_prompt_var.value}\n")

        optimizer.zero_grad()

        # Send iteration complete event
        yield {
            'type': 'iteration_complete',
            'loop': loop_idx + 1
        }
    print(f"Updated input: {system_prompt_var}\n")
    print(f"Textgrad model parameters: {textgrad_model.parameters()}\n")
    final_answer = textgrad_model(prompt)
    answer_text = final_answer.value
    print(f"Final Updated Answer from textgrad loop: {answer_text}\n")
    # Send final complete event with answer
    yield {
        'type': 'complete',
        'answer': answer_text
    }


def run_textgrad_sync(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None, textGrad_api_key=None, temperature=0.0):
    """Synchronous version that collects all events and returns final answer (for backward compatibility)."""
    answer_text = ''
    for event in run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops, api_key, textGrad_api_key, temperature):
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
        textGrad_api_key=API_KEY,
        loss_prompt='Evaluate this answer. It should be factual, clear, and directly answer the question.',
        temperature=0.0
    )
    print(result)
