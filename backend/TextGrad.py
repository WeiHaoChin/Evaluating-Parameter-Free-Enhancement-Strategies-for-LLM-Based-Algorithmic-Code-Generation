import os
import textgrad as tg
from textgrad import Variable, TGD
from dotenv import load_dotenv
from llm_clients import create_llm_client
from config.models import get_model_provider

import time
import random
from functools import partial
from textgrad.variable import _backward_idempotent

load_dotenv()
API_KEY = os.getenv('OLLAMA_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def create_textgrad_model(textGradModel, model, system_prompt, api_key=None, textGrad_api_key=None,temperature=0.0):
    main_api_key = api_key or (GOOGLE_API_KEY if get_model_provider(model) == "google" else None)
    feedback_api_key = textGrad_api_key or (
        GOOGLE_API_KEY if get_model_provider(textGradModel) == "google" else None
    )
    llm = create_llm_client(model=model, api_key=main_api_key, temperature=temperature)
    feedback_llm = create_llm_client(
        model=textGradModel,
        api_key=feedback_api_key,
        temperature=temperature,
    )
    # Wrap the main LLM as a BlackboxLLM node in TextGrad's computation graph.
    # "Blackbox" means TextGrad interacts with it purely via text in/out,
    # using the backward engine to critique and iteratively refine its outputs.
    # Keep the backward engine local to this run. The previous global engine
    # made simultaneous benchmark modes race with one another.
    return tg.BlackboxLLM(llm, system_prompt=system_prompt), feedback_llm


def run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None,textGrad_api_key=None,temperature=0.0, initial_answer=None, progress_callback=None):
    """Generator version that yields events for streaming."""
    loops = max(1, min(int(loops), 5))
    if loops < 1:
        loops = 1

    prompt = Variable(prompt_text, requires_grad=False, role_description='The user input/question provided to the model. This is fixed and should not be modified.')
    system_prompt_var = Variable(system_prompt, requires_grad=True, role_description="The system prompt that defines the model's behavior and instructions. Optimize this to improve the quality, accuracy, and clarity of the model's responses.")
    textgrad_model, feedback_llm = create_textgrad_model(textGradModel=textGradModel, model=model, system_prompt=system_prompt_var, api_key=api_key,textGrad_api_key=textGrad_api_key,temperature=temperature)
    optimizer = TGD(parameters=[system_prompt_var], engine=feedback_llm)

    answer_text = ''
    for loop_idx in range(loops):
        if progress_callback:
            progress_callback("generating", f"Generating answer (iteration {loop_idx + 1}/{loops})")
        # Send iteration start event with original prompt
        yield {
            'type': 'iteration_start',
            'loop': loop_idx + 1,
            'original_prompt': prompt.value
        }

        # When the benchmark already generated an answer for this exact prompt,
        # evaluate that answer instead of paying for a duplicate initial call.
        # The identity gradient connects its critique to the optimised system
        # prompt just as the normal BlackboxLLM output would.
        if loop_idx == 0 and initial_answer is not None:
            answer = Variable(
                initial_answer,
                predecessors=[system_prompt_var],
                requires_grad=True,
                role_description="initial solution reused from the paired benchmark mode",
            )
            answer.set_grad_fn(partial(
                _backward_idempotent,
                variables=[system_prompt_var],
                summation=answer,
            ))
        else:
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
        if progress_callback:
            progress_callback("getting_feedback", f"Getting feedback (iteration {loop_idx + 1}/{loops})")
        loss_fn = tg.TextLoss(loss_prompt, engine=feedback_llm)
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
        if progress_callback:
            progress_callback("optimizing", f"Applying feedback (iteration {loop_idx + 1}/{loops})")
        loss.backward(engine=feedback_llm)
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
    # print(f"Updated System Prompt: {system_prompt_var.value}\n")
    # print(f"Textgrad model parameters: {textgrad_model.parameters()}\n")
    if progress_callback:
        progress_callback("generating", "Generating final improved answer")
    final_answer = textgrad_model(prompt)
    answer_text = final_answer.value
    print(f"Final Updated Answer from textgrad loop: {answer_text}\n")
    # Send final complete event with answer
    yield {
        'type': 'complete',
        'answer': answer_text
    }


def run_textgrad_sync(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1, api_key=None, textGrad_api_key=None, temperature=0.0, initial_answer=None, progress_callback=None):
    """Synchronous version that collects all events and returns final answer (for backward compatibility)."""
    answer_text = ''
    for event in run_textgrad(prompt_text, system_prompt, textGradModel, model, loss_prompt, loops, api_key, textGrad_api_key, temperature, initial_answer, progress_callback):
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
