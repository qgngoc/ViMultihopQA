

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import constants
from llm_manage.vllm_llm import VllmLLM
from llm_manage.api_llm import APILLM
class UtilizationFilter:
    def __init__(self, llm):
        self.llm = llm

    def filter_qa(self, generated_qa):
        try:
            context_1 = generated_qa['context_1']
            context_2 = generated_qa['context_2']
            question = generated_qa['question']
            ground_truth_label = generated_qa['ground_truth_label']
            prediction_1 = self.gen_answer(question, context_1)
            judgment_1 = self.judge(question, prediction_1, ground_truth_label)
            if judgment_1:
                generated_qa['context_utilization'] = False
                return generated_qa
            prediction_2 = self.gen_answer(question, context_2)
            judgment_2 = self.judge(question, prediction_2, ground_truth_label)
            if judgment_2:
                generated_qa['context_utilization'] = False
                return generated_qa
            generated_qa['context_utilization'] = True

            return generated_qa
        except Exception as e:
            print(f"Error in filter_qa: {e}")
            return None
    
    
    def filter_qa_list(self, generated_qa_list):
        filtered_list = []
        if isinstance(self.llm, APILLM):
            with ThreadPoolExecutor(max_workers=constants.N_CONCURENTS) as executor:
                futures = [executor.submit(self.filter_qa, generated_qa) for generated_qa in generated_qa_list]
                for future in tqdm(futures, desc="Filtering QA", total=len(futures)):
                    if future.result():
                        filtered_list.append(future.result())
                # for generated_qa in generated_qa_list:
                #     future = executor.submit(self.filter_qa, generated_qa)
                #     futures.append(future)
                # for future in futures:
                #     if future.result():
                #         filtered_list.append(generated_qa)
        elif isinstance(self.llm, VllmLLM):
            for generated_qa in tqdm(generated_qa_list, desc="Filtering QA"):
                if self.filter_qa(generated_qa):
                    filtered_list.append(generated_qa)
        else:
            raise ValueError("Unsupported LLM type")
        filtered_list = [qa for qa in filtered_list if qa is not None]
        filtered_list = [qa for qa in filtered_list if qa['context_utilization']]
        return filtered_list
    
    
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
        response = self.llm.chat(messages)
        return 'yes' in response.lower()
    
    def gen_answer(self, question, context):
        prompt = constants.GENERATE_ANSWER_PROMPT.format(
            question=question,
            context=context
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.chat(messages)
        return response