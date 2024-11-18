import requests
import os
from services.answering.answering_congif.answering_config import AnsweringConfig
from util.prompt import Prompt

from dotenv import load_dotenv

load_dotenv()
# Fireworks API key - > reading from the env file.
API_KEY = os.getenv("API_KEY")
# Fireworks API endpoint
API_URL = os.getenv("API_URL")


class AnsweringService:

    def __init__(self):
        self.prompt = Prompt()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__), AnsweringConfig.system_prompt
        )
        return self.prompt.get_system_prompt(prompt_path)

    def get_fireworks_response(self, user_input):
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json',
        }
        data = {
            'model': AnsweringConfig.model_name,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_input}],
            'max_tokens': AnsweringConfig.max_tokens,
            'temperature': AnsweringConfig.temperature,
            'top_p': AnsweringConfig.top_p,
            'n': AnsweringConfig.n,
            'frequency_penalty': AnsweringConfig.frequency_penalty,
            'stream': False,
        }
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
