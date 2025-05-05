from tqdm import tqdm
from transformers import pipeline

class ContextFilter:
    def __init__(self):
        self.ner_model = pipeline("ner", model="NlpHUST/ner-vietnamese-electra-base")

    def filter_context_pairs(self, context_pairs):
        filtered_pairs = []
        for context_pair in tqdm(context_pairs):
            try:
                if self.filter_context_pair(context_pair):
                    filtered_pairs.append(context_pair)
            except Exception as e:
                print(f"Error processing context pair {context_pair['id']}: {e}")
                continue
        return filtered_pairs

    def filter_context_pair(self, context_pair):
        title_1 = context_pair['title_1']
        context_1 = context_pair['context_1']
        title_2 = context_pair['title_2']
        context_2 = context_pair['context_2']

        if self.filter_length(context_1) and self.filter_length(context_2):
            if self.is_named_entity(title_1) and self.is_named_entity(title_2):
                return True

        return False
    
    def filter_length(self, context: str, min_length: int = 30, max_length: int = 1000):
        n_words = len(context.split())
        return min_length <= n_words <= max_length

    
    def is_named_entity(self, title: str):
        return len(self.ner_model(title)) > 0