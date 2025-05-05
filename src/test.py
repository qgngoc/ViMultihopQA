from dotenv import load_dotenv
load_dotenv()
import json
import os
import re

from data_preprocess.wiki_parser import WikiParser
from generate_qa.context_filter import ContextFilter
from generate_qa.qa_generator import QAGenerator
from llm_manage.vllm_llm import VllmLLM
from llm_manage.api_llm import APILLM
from data_postprocess.dificulty_evaluator import DifficultyEvaluator
from data_postprocess.utilization_filter import UtilizationFilter
from evaluate.evaluator import Evaluator
# Load environment variables

def extract_links(text):
    pattern = r'&lt;a href=\"(.*?)\"?&gt;'
    links = re.findall(pattern, text)
    return links

if __name__ == "__main__":
    # Initialize the WikiParser
    # wiki_parser = WikiParser()

    # # Define the input file path and output file name
    # input_file_path = "../data/wikidumps/viwiki-20250320-pages-articles-multistream.xml.bz2"
    # output_file_name = "wiki_extracted.jsonl"

    # # Parse the wiki dump file and save the extracted content
    # wiki_parser.parse(input_file_path, output_file_name)
    
    # output_file_name = "wiki_extracted_with_links.jsonl"
    # wiki_parser.parse(input_file_path, output_file_name, with_links=True)
    
    # with open('../data/wiki_extracted/wiki_extracted_with_links.jsonl', 'r', encoding='utf-8') as f:
    #     ds = [json.loads(line) for line in f]
    
    # with open('../data/wiki_extracted/wiki_extracted.jsonl', 'r', encoding='utf-8') as f:
    #     ds_no_links = [json.loads(line) for line in f]
    
    # for d1, d2 in zip(ds, ds_no_links):
    #     assert d1['title'] == d2['title']
    
    # from data_preprocess.pairs_extractor import PairsExtractor

    # pairs_extractor = PairsExtractor()

    # pairs_extractor.extract_pairs(dataset_no_links=ds_no_links, dataset_with_links=ds)

    # with open('../data/context_pairs.jsonl', 'r', encoding='utf-8') as f:
    #     context_pairs = [json.loads(line) for line in f]
    #     print(len(context_pairs))
    
    # context_filter = ContextFilter()
    # filtered_context_pairs = context_filter.filter_context_pairs(context_pairs)
    # print(len(filtered_context_pairs))

    # with open('../data/context_pairs_filtered.jsonl', 'w', encoding='utf-8') as f:
    #     for item in filtered_context_pairs:
    #         f.write(json.dumps(item, ensure_ascii=False))
    #         f.write('\n')

    # with open('../data/context_pairs_filtered.jsonl', 'r', encoding='utf-8') as f:
    #     context_pairs = [json.loads(line) for line in f]
    
    # llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8000/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=0.3)
    # # llm = VllmLLM(model="/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct", tensor_parallel_size=8)

    # qa_generator = QAGenerator(llm)

    # qa_generator.generate_qa_list(context_pairs)
    # llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8000/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=0.3)

    # with open("../data/generated_multihop_qa.jsonl", "r", encoding="utf-8") as f:
    #     generated_qa_list = [json.loads(line) for line in f]
    
    # utilization_filter = UtilizationFilter(llm)

    # utilization_filtered_qa_list = utilization_filter.filter_qa_list(generated_qa_list)
    # print(len(utilization_filtered_qa_list))
    # with open("../data/generated_multihop_qa_utilization_filtered.jsonl", "w", encoding="utf-8") as f:
    #     for item in utilization_filtered_qa_list:
    #         f.write(json.dumps(item, ensure_ascii=False))
    #         f.write('\n')
        
    # llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8001/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=1)

    # judge_llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8001/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=0.01)
    
    # with open("../data/generated_multihop_qa_utilization_filtered.jsonl", "r", encoding="utf-8") as f:
    #     utilization_filtered_qa_list = [json.loads(line) for line in f]
    
    # difficulty_evaluator = DifficultyEvaluator(llm, judge_llm)
    # evaluated_qa_list = difficulty_evaluator.evaluate_difficulty_qa_list(utilization_filtered_qa_list)

    # print(len(evaluated_qa_list))
    # with open("../data/generated_multihop_qa_difficulty_evaluated.jsonl", "w", encoding="utf-8") as f:
    #     for item in evaluated_qa_list:
    #         f.write(json.dumps(item, ensure_ascii=False))
    #         f.write('\n')

    llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8001/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=0.01)

    judge_llm = APILLM(model='/data/lhtm-opt3/hub/Llama-3.3-70B-Instruct', base_url='http://localhost:8001/v1/', api_key=os.getenv('OPENAI_API_KEY'), temperature=0.01)
    
    
    evaluator = Evaluator(llm, judge_llm)
    with open("../data/generated_multihop_qa_difficulty_evaluated.jsonl", "r", encoding="utf-8") as f:
        evaluated_qa_list = [json.loads(line) for line in f]
    
    evaluated_qa_list = evaluator.evaluate_qa_list(evaluated_qa_list)
    
    folder_path = "../data/results/Llama-3.3-70B-Instruct"
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "samples.jsonl"), "w", encoding="utf-8") as f:
        for item in evaluated_qa_list:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')
            
    aggregated_results = evaluator.aggregate_results(evaluated_qa_list)
    with open(os.path.join(folder_path, "aggregated_results.json"), "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, ensure_ascii=False, indent=2)