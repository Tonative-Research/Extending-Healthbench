from anthropic import Anthropic


class ClaudeProvider:
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
    def translate(self, sytem_prompt: str, text: str):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000, 
            temperature=0.3,
            system=sytem_prompt,
            messages=[{"role": "user", "content": text}]
        )
        
        return response.content[0].text.strip()