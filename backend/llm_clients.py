from ollama import Client
from google import genai

class OllamaLLM:
    def __init__(self, model, host='https://ollama.com', api_key=None, temperature=0.0):
        self.model = model
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.client = Client(host=host, headers=headers)
        self.temperature = temperature

    def __call__(self, prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': str(system_prompt)})
        messages.append({'role': 'user', 'content': str(prompt)})

        response = self.client.chat(model=self.model, messages=messages, options={'temperature': self.temperature})
        return response['message']['content']

class GoogleGenerativeAI:
    def __init__(self, model, api_key,temperature=0.0):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.max_retries = 5
        self.initial_delay = 1  # Start with 1 second delay
        self.temperature = temperature

    def __call__(self, prompt, system_prompt=None, **kwargs):
        # For Google models, system_prompt is usually part of the prompt in a multi-turn conversation
        # or set as a safety setting. Here, we'll prepend it to the prompt if provided.
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else str(prompt)
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(model=self.model, contents=full_prompt, config={genai.types.GenerationConfig(temperature=self.temperature)})
                return response.text
            except Exception as e:
                error_str = str(e)
                # Check if it's a 503 or rate limit error
                if '503' in error_str or 'UNAVAILABLE' in error_str or 'high demand' in error_str:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = self.initial_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"API rate limited (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time:.2f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Max retries ({self.max_retries}) exceeded. Raising error.")
                        raise
                else:
                    # For other errors, don't retry
                    raise