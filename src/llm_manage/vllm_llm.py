import retry
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv(override=True)
import os


class VllmLLM:
    def __init__(self, model, tensor_parallel_size=8, temperature=0.1):
        self.temperature = temperature
        self.model = model
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, gpu_memory_utilization=0.9)
        self.n_retries = 5

    def chat(self, messages, **kwargs):
        try:
            if 'n' in kwargs:
                if kwargs['n'] > 1:
                    return self._chat_multiple(messages, **kwargs)
            return self._chat(messages, **kwargs)
        except Exception as e:
            raise Exception(f"Call {self.model} error: {e}")
    
    def generate(self, prompt, sys='You are a helpful assistant.', **kwargs):
        if len(sys) > 0:
            messages=[{"role": "system", "content": sys}, {"role": "user","content": prompt}]
        else:
            messages=[{"role": "user","content": prompt}]
        try:
            return self._chat(messages)
        except Exception as e:
            raise Exception(f"Call {self.model} error: {e}")
    
    @retry.retry(tries=5, delay=1)
    def _chat(self, messages, **kwargs):
        outputs = self.llm.chat(messages, sampling_params=SamplingParams(temperature=self.temperature, max_tokens=32000, **kwargs), use_tqdm=False)
        output = outputs[0].outputs[0].text
        if not output:
            raise Exception(f"Empty response from {self.model}")
        return output
    
    @retry.retry(tries=5, delay=1)
    def _chat_multiple(self, messages, n=1, **kwargs):
        outputs = self.llm.chat(messages, sampling_params=SamplingParams(temperature=self.temperature, max_tokens=32000, n=n, **kwargs), use_tqdm=False)
        predictions = [output.text for output in outputs[0].outputs]
        return predictions