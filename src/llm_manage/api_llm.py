import retry
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
import os


class APILLM:
    def __init__(self, model='gpt-4o', base_url=None, api_key=None, temperature=0.1):
        self.llm = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.n_retries = 10
        
    def chat(self, messages, **kwargs):
        # print(messages)
        try:
            if 'n' in kwargs:
                if kwargs['n'] > 1:
                    return self._chat_multiple(messages, **kwargs)
            return self._chat(messages, **kwargs)
        except Exception as e:
            raise e
        
    def generate(self, prompt, sys='You are a helpful assistant.', **kwargs):
        if len(sys) > 0:
            messages=[{"role": "system", "content": sys}, {"role": "user","content": prompt}]
        else:
            messages=[{"role": "user","content": prompt}]
        try:
            return self._chat(messages)
        except Exception as e:
            print(f"Call {self.model} error: {e}")
    
    @retry.retry(tries=5, delay=1)
    def _chat(self, messages, **kwargs):
        try:
            completion = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=300,
            **kwargs
            )
            output = completion.choices[0].message.content
            if not output:
                raise Exception(f"Empty response from {self.model}")
            return output
        except Exception as e:
            raise e
    
    @retry.retry(tries=5, delay=1)
    def _chat_multiple(self, messages, n=1, **kwargs):
        try:
            completion = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            n=n,
            timeout=300,
            **kwargs
            )
            predictions = [choice.message.content for choice in completion.choices]
            return predictions
        except Exception as e:
            raise e
    
    
    
if __name__ == "__main__":
    llm = APILLM(model='gemini-2.0-flash', base_url='https://generativelanguage.googleapis.com/v1beta/openai/', temperature=0.1, api_key=os.getenv('GEMINI_API_KEY'))
    print(llm.generate("What is the capital of France?"))