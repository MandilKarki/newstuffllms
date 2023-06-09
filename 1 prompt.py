class PromptManager:
    def __init__(self, prompt):
        self.prompt = prompt

    def append_prompt(self, text):
        return self.prompt + " " + text
