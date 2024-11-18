import os
import sys

class Prompt:

    def read_prompt_template(self, prompt_template_path: str):
        try:
            with open(prompt_template_path, "r") as file:
                prompt_template = file.read()
                return prompt_template
        except Exception as e:
            return str(e)

    def get_system_prompt(self, system_prompt_path):
        system_prompt = self.read_prompt_template(system_prompt_path)
        return system_prompt