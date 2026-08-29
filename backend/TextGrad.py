import os
import textgrad as tg
from textgrad import Variable, TGD
from textgrad.loss import MultiFieldEvaluation
from dotenv import load_dotenv
from llm_clients import create_llm_client
from config.models import get_model_provider
from config.generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
)

load_dotenv()
API_KEY = os.getenv('OLLAMA_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def create_textgrad_model(
    textGradModel, model, system_prompt, api_key=None, textGrad_api_key=None,
    temperature=0.0, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    internal_max_output_tokens=DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
    generation_records=None, mode="textgrad_only",
):
    main_api_key = api_key or (GOOGLE_API_KEY if get_model_provider(model) == "google" else None)
    feedback_api_key = textGrad_api_key or (
        GOOGLE_API_KEY if get_model_provider(textGradModel) == "google" else None
    )
    llm = create_llm_client(
        model=model, api_key=main_api_key, temperature=temperature,
        max_output_tokens=max_output_tokens, metadata_sink=generation_records,
        mode=mode, call_type="initial_generation",
    )
    feedback_llm = create_llm_client(
        model=textGradModel,
        api_key=feedback_api_key,
        temperature=temperature,
        max_output_tokens=internal_max_output_tokens,
        metadata_sink=generation_records,
        mode=mode,
        call_type="critique_evaluation",
    )
    # Wrap the main LLM as a BlackboxLLM node in TextGrad's computation graph.
    # "Blackbox" means TextGrad interacts with it purely via text in/out,
    # using the backward engine to critique and iteratively refine its outputs.
    # Keep the backward engine local to this run. The previous global engine
    # made simultaneous benchmark modes race with one another.
    return tg.BlackboxLLM(llm, system_prompt=system_prompt), feedback_llm, llm


def run_textgrad(
    prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1,
    api_key=None, textGrad_api_key=None, temperature=0.0,
    progress_callback=None, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    internal_max_output_tokens=DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
    generation_records=None,
    mode="textgrad_only",
):
    """Generator version that yields events for streaming."""
    loops = max(1, min(int(loops), 5))
    if loops < 1:
        loops = 1

    prompt = Variable(prompt_text, requires_grad=False, role_description='The user input/question provided to the model. This is fixed and should not be modified.')
    system_prompt_var = Variable(system_prompt, requires_grad=True, role_description="The system prompt that defines the model's behavior and instructions. Optimize this to improve the quality, accuracy, and clarity of the model's responses. The solution must remain Python-only")
    textgrad_model, feedback_llm, main_llm = create_textgrad_model(
        textGradModel=textGradModel, model=model, system_prompt=system_prompt_var,
        api_key=api_key, textGrad_api_key=textGrad_api_key, temperature=temperature,
        max_output_tokens=max_output_tokens,
        internal_max_output_tokens=internal_max_output_tokens,
        generation_records=generation_records, mode=mode,
    )
    optimizer = TGD(parameters=[system_prompt_var], engine=feedback_llm)

    answer_text = ''
    initial_answer_text = ''
    for loop_idx in range(loops):
        if progress_callback:
            progress_callback("generating", f"Generating answer (iteration {loop_idx + 1}/{loops})")
        # Send iteration start event with original prompt
        yield {
            'type': 'iteration_start',
            'loop': loop_idx + 1,
            'original_prompt': prompt.value
        }

        main_llm.set_generation_context(
            "initial_generation" if loop_idx == 0 else "regenerated_solution"
        )
        answer = textgrad_model(prompt)
        answer_text = answer.value
        if loop_idx == 0:
            initial_answer_text = answer_text
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
        feedback_llm.set_generation_context("critique_evaluation")
        evaluation_instruction = Variable(
            loss_prompt,
            requires_grad=False,
            role_description="competitive programming evaluation instruction",
        )
        loss_fn = MultiFieldEvaluation(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=[
                "problem prompt and optional retrieval context",
                "generated Python solution",
            ],
            engine=feedback_llm,
        )
        loss = loss_fn([prompt, answer])
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
        feedback_llm.set_generation_context("critique_backward")
        loss.backward(engine=feedback_llm)
        feedback_llm.set_generation_context("prompt_optimization")
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
    main_llm.set_generation_context("final_generation")
    final_answer = textgrad_model(prompt)
    answer_text = final_answer.value
    print(f"Final Updated Answer from textgrad loop: {answer_text}\n")
    # Send final complete event with answer
    yield {
        'type': 'complete',
        'answer': answer_text,
        'initial_answer': initial_answer_text,
        'generation_records': generation_records or [],
    }


def run_textgrad_sync(
    prompt_text, system_prompt, textGradModel, model, loss_prompt, loops=1,
    api_key=None, textGrad_api_key=None, temperature=0.0,
    progress_callback=None, return_details=False,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    internal_max_output_tokens=DEFAULT_TEXTGRAD_INTERNAL_MAX_OUTPUT_TOKENS,
    generation_records=None,
    mode="textgrad_only",
):
    """Collect TextGrad events and optionally return the optimized prompt."""
    answer_text = ''
    initial_answer_text = ''
    improved_system_prompt = system_prompt
    for event in run_textgrad(
        prompt_text, system_prompt, textGradModel, model, loss_prompt, loops,
        api_key, textGrad_api_key, temperature, progress_callback,
        max_output_tokens, internal_max_output_tokens, generation_records, mode,
    ):
        if event['type'] == 'prompt_updated':
            improved_system_prompt = event['updated']
        if event['type'] == 'complete':
            answer_text = event['answer']
            initial_answer_text = event['initial_answer']
    if return_details:
        return answer_text, improved_system_prompt, initial_answer_text
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
