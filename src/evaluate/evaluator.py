
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import constants
from llm_manage.vllm_llm import VllmLLM
from llm_manage.api_llm import APILLM

class Evaluator:
    def __init__(self, llm, judge_llm):
        self.llm = llm
        self.judge_llm = judge_llm

    def evaluate_qa_list(self, generated_qa_list):
        if isinstance(self.llm, APILLM):
            with ThreadPoolExecutor(max_workers=constants.N_CONCURENTS) as executor:
                futures = [executor.submit(self.evaluate_qa, generated_qa) for generated_qa in generated_qa_list]
                results = []
                for future in tqdm(futures, desc="Evaluating Difficulty", total=len(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
        elif isinstance(self.llm, VllmLLM):
            results = [self.evaluate_difficulty_qa(generated_qa) for generated_qa in tqdm(generated_qa_list, desc="Evaluating Difficulty")]
        else:
            raise ValueError("Unsupported LLM type. Please use either APILLM or VllmLLM.")
        results = [qa for qa in results if qa is not None]
        return results

    def evaluate_qa(self, generated_qa):
        try:
            question = generated_qa['question']
            # answer = generated_qa['answer']
            ground_truth_label = generated_qa['ground_truth_label']
            context_1 = generated_qa['context_1']
            context_2 = generated_qa['context_2']

            prompt = constants.GENERATE_ANSWER_PROMPT.format(
                question=question,
                context=f"Context 1: {context_1}\nContext 2: {context_2}",
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # predictions = [self.llm.chat(messages) for context in [context_1, context_2]]
            prediction = self.llm.chat(messages)
            judgement = self.judge(question, prediction, ground_truth_label)
            generated_qa['is_correct'] = judgement
            
            return generated_qa
        except Exception as e:
            print(f"Error in evaluate_difficulty_qa: {e}")
            return None
    
    def judge(self, question, answer, ground_truth_label):
        prompt = constants.JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth_label=ground_truth_label
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.judge_llm.chat(messages)
        return 'yes' in response.lower()
    
    def aggregate_results(self, results):
        correct_count = sum(1 for result in results if result['is_correct'])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        return {
            'model_name': self.llm.model,
            'judge_model_name': self.judge_llm.model,
            'correct_count': correct_count,
            'total_count': total_count,
            'accuracy': accuracy
        }