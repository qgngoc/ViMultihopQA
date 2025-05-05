from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self):
        pass

    def chat(self, messages, **kwargs):
        try:
            return self._chat(messages, **kwargs)
        except Exception as e:
            raise Exception(f"Call {self.model} error: {e}")
    
    @abstractmethod
    def _chat(self, messages, **kwargs):
        pass