
import re
import os
import json
from tqdm import tqdm
from retry import retry
from concurrent.futures import ThreadPoolExecutor

from llm_manage.api_llm import APILLM
from llm_manage.vllm_llm import VllmLLM

class QAGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_qa_list(self, context_pairs):
        batch_size = 250
        output_file = '../data/generated_multihop_qa.jsonl'
        i = 0
        print(f"Data generated: {i}")
        pbar = tqdm(total=len(context_pairs))
        pbar.update(i)
        # existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = [json.loads(line) for line in f]
                i = len(existing_data)

        while i < len(context_pairs):
            batch = context_pairs[i:i + batch_size]
            if isinstance(self.llm, APILLM):
                with ThreadPoolExecutor(max_workers=25) as executor:
                    futures = [executor.submit(self.generate_qa, context_pair) for context_pair in batch]
                    results = [future.result() for future in tqdm(futures, desc=f"Processing from {i} to {i + batch_size}")]
            elif isinstance(self.llm, VllmLLM):
                results = [self.generate_qa(context_pair) for context_pair in tqdm(batch, desc=f"Processing from {i} to {i + batch_size}")]
            else:
                raise ValueError("Unsupported LLM type")

            results = [result for result in results if result is not None]
            with open(output_file, 'a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            pbar.update(batch_size)
            i += batch_size

    def generate_qa(self, context_pair):
        try:
            # Placeholder for the actual implementation
            question1 = self.gen_q1(context_pair)
            question2, answer2 = self.gen_q2a2(context_pair)
            question = self.get_2hop_q(question1, context_pair['title_2'], question2, answer2)
            # ground_truth_label = answer2
            full_sentence_answer = self.gen_full_sentence_answer(question, answer2)
            reasoning_answer = self.gen_reasoning_answer(question, answer2, context_pair['context_1'], context_pair['context_2'])
            return {
                'question': question,
                'ground_truth_label': answer2,
                'full_sentence_answer': full_sentence_answer,
                'reasoning_answer': reasoning_answer,
                'question1': question1,
                'question2': question2,
                'context_1': context_pair['context_1'],
                'context_2': context_pair['context_2'],
                'title_1': context_pair['title_1'],
                'title_2': context_pair['title_2'],
            }
        except Exception as e:
            print(f"Error generating QA for context pair {context_pair['id']}: {e}")
            return None

    def gen_q1(self, context_pair):
        context_1 = context_pair['context_1']
        title_2 = context_pair['title_2']
        example = """Paragraph: Hồ Chí Minh (19 tháng 5 năm 1890 – 2 tháng 9 năm 1969), tên khai sinh là Nguyễn Sinh Cung, còn được biết với tên gọi Bác Hồ, là một nhà cách mạng và chính khách người Việt Nam. Ông là người sáng lập Đảng Cộng sản Việt Nam, từng là Chủ tịch nước Việt Nam Dân chủ Cộng hòa từ 1945–1969, Thủ tướng Việt Nam Dân chủ Cộng hòa trong những năm 1945–1955, Tổng Bí thư Ban Chấp hành Trung ương Đảng Lao động Việt Nam từ 1956–1960, Chủ tịch Ban Chấp hành Trung ương Đảng Lao động Việt Nam từ năm 1951 cho đến khi qua đời.
Trong quãng thời gian sinh sống và hoạt động trước khi lên nắm quyền, Hồ Chí Minh đã đi qua nhiều quốc gia và châu lục, ông được cho là đã sử dụng 50 đến 200 bí danh khác nhau.[2] Về mặt tư tưởng chính trị, Hồ Chí Minh là một người theo chủ nghĩa Marx–Lenin. Ông là nhà lãnh đạo phong trào độc lập Việt Minh tiến hành Cách mạng Tháng Tám năm 1945. Ông cũng là người đã soạn thảo, đọc bản Tuyên ngôn độc lập thành lập nước Việt Nam Dân chủ Cộng hòa, và trở thành Chủ tịch nước sau cuộc tổng tuyển cử năm 1946.
Entity: Đảng Cộng sản Việt Nam
Question: Hồ Chí Minh đã sáng lập đảng nào?"""

        system_prompt = f"""Given a paragraph and an entity, generate a question (in Vietnamese) which has the answer is the given entity.
Example:
{example}"""

        prompt = f"""Paragraph: {context_1}
Entity: {title_2}
Question:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        resp = self.llm.chat(messages)

        return resp

    @retry(tries=10, delay=1)
    def gen_q2a2(self, context_pair):
        title_2 = context_pair['title_2']
        context_2 = context_pair['context_2']
        title_1 = context_pair['title_1']

        example = """Paragraph: Đảng Cộng sản Việt Nam là đảng cầm quyền và là chính đảng duy nhất được phép hoạt động tại Việt Nam theo Hiến pháp. Theo Cương lĩnh và Điều lệ chính thức hiện nay, Đảng là đại diện của giai cấp công nhân, nhân dân lao động và của cả dân tộc, lấy Chủ nghĩa Marx-Lenin và Tư tưởng Hồ Chí Minh làm kim chỉ nam cho mọi hoạt động.[3] Trong ngữ cảnh không chính thức cũng dùng từ "Đảng" (hoặc "Đảng ta") để nói về Đảng Cộng sản Việt Nam.
Cơ quan cao nhất của Đảng là Đại hội Đại biểu toàn quốc, nơi sẽ bầu ra Ban Chấp hành Trung ương. Giữa các kỳ Đại hội Đảng, Ban Chấp hành Trung ương Đảng là cơ quan tối cao quyết định các vấn đề của Đảng. Sau Đại hội, Trung ương sẽ bầu ra Bộ Chính trị, Ban Bí thư và Ủy ban Kiểm tra Trung ương và bầu một Ủy viên Bộ Chính trị làm Tổng Bí thư.
Entity: Đảng Cộng sản Việt Nam
Question answer pair: <question>Cơ quan cao nhất của Đảng Cộng Sản Việt Nam là gì?<answer>Đại hội Đại biểu toàn quốc"""

        system_prompt = f"""Given a paragraph and an entity, generate one question answer pair (in Vietnamese) about the entity in the format <question>question<answer>answer.
Example:
{example}

Generate only one question answer pair in one line."""

        prompt = f"""Paragraph: {context_2}
Entity: {title_2}
**Note**: Do not generate the question which the answer is {title_1}
Question answer pair:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        resp = self.llm.chat(messages)
        q, a = self.extract_qa(resp)
        if title_1.lower() in a.lower() or a.lower() in title_1.lower():
            raise Exception(f"Answer {a} contains title {title_1}")
        # print(f"Question: {q}")
        # print(f"Answer: {a}")
        return q, a


    def get_2hop_q(self, q1, a1, q2, a2):
        example = """Question 1: Hồ Chí Minh đã sáng lập đảng nào?
Answer 1: Đảng Cộng sản Việt Nam
Question 2: Cơ quan cao nhất của Đảng Cộng Sản Việt Nam là gì?
Answer 2: Đại hội Đại biểu toàn quốc
Question: Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là gì?"""

        system_prompt = f"""Given 2 question answer pair, which the answer (answer 1) of question 1 is the main topic of question 2. Formulate a multi-hop question by combining question 1 and question 2 through answer 1 as the bridge entity, do not include the answer 1 in the question. The answer of the formulated question must be answer 2.
Note: The question must be in Vietnamese, do not translate to English, do not generate the answer.
Example:
{example}"""

        prompt = f"""Question 1: {q1}
Answer 1: {a1}
Question 2: {q2}
Answer 2: {a2}
Question:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        resp = self.llm.chat(messages)
        return resp


    def gen_full_sentence_answer(self, q, a):
        example = """Question: Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là gì?
Answer: Đại hội Đại biểu toàn quốc
Rewritten answer: Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là Đại hội Đại biểu toàn quốc。"""
        system_prompt = f"""Given a question and its answer, rewrite the answer to be more clear and understandable.
Example:
{example}"""

        prompt = f"""Question: {q}
Answer: {a}
Rewritten answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        resp = self.llm.chat(messages)
        return resp


    def gen_reasoning_answer(self, question, answer, context1, context2):
        example = """Question: Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là gì?
Answer: Đại hội Đại biểu toàn quốc

List of context: 
Context 1: Hồ Chí Minh (chữ Nho: 胡志明; 19 tháng 5 năm 1890 – 2 tháng 9 năm 1969), tên khai sinh là Nguyễn Sinh Cung (chữ Nho: 阮生恭), còn được biết với tên gọi Bác Hồ, là một nhà cách mạng và chính khách người Việt Nam. Ông là người sáng lập Đảng Cộng sản Việt Nam, từng là Chủ tịch nước Việt Nam Dân chủ Cộng hòa từ 1945–1969, Thủ tướng Việt Nam Dân chủ Cộng hòa trong những năm 1945–1955, Tổng Bí thư Ban Chấp hành Trung ương Đảng Lao động Việt Nam từ 1956–1960, Chủ tịch Ban Chấp hành Trung ương Đảng Lao động Việt Nam từ năm 1951 cho đến khi qua đời.
Trong quãng thời gian sinh sống và hoạt động trước khi lên nắm quyền, Hồ Chí Minh đã đi qua nhiều quốc gia và châu lục, ông được cho là đã sử dụng 50[1] đến 200 bí danh khác nhau.[2] Về mặt tư tưởng chính trị, Hồ Chí Minh là một người theo chủ nghĩa Marx–Lenin. Ông là nhà lãnh đạo phong trào độc lập Việt Minh tiến hành Cách mạng Tháng Tám năm 1945. Ông cũng là người đã soạn thảo, đọc bản Tuyên ngôn độc lập thành lập nước Việt Nam Dân chủ Cộng hòa, và trở thành Chủ tịch nước sau cuộc tổng tuyển cử năm 1946.
Context 2: Đảng Cộng sản Việt Nam là đảng cầm quyền và là chính đảng duy nhất được phép hoạt động tại Việt Nam theo Hiến pháp. Theo Cương lĩnh và Điều lệ chính thức hiện nay, Đảng là đại diện của giai cấp công nhân, nhân dân lao động và của cả dân tộc, lấy Chủ nghĩa Marx-Lenin và Tư tưởng Hồ Chí Minh làm kim chỉ nam cho mọi hoạt động.[3] Trong ngữ cảnh không chính thức cũng dùng từ "Đảng" (hoặc "Đảng ta") để nói về Đảng Cộng sản Việt Nam.
Cơ quan cao nhất của Đảng là Đại hội Đại biểu toàn quốc, nơi sẽ bầu ra Ban Chấp hành Trung ương. Giữa các kỳ Đại hội Đảng, Ban Chấp hành Trung ương Đảng là cơ quan tối cao quyết định các vấn đề của Đảng. Sau Đại hội, Trung ương sẽ bầu ra Bộ Chính trị, Ban Bí thư và Ủy ban Kiểm tra Trung ương và bầu một Ủy viên Bộ Chính trị làm Tổng Bí thư.

Chain of thought: 1. Đầu tiên, từ câu hỏi "Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là gì?", ta có thể xác định rằng câu hỏi đang tìm kiếm một tổ chức hoặc cơ quan liên quan đến Hồ Chí Minh.
2. Context 1 chứa thông tin về Hồ Chí Minh đã người sáng lập Đảng Cộng sản Việt Nam.
3. Context 2 cho biết cơ quan cao nhất của Đảng Cộng sản Việt Nam là Đại hội Đại biểu toàn quốc.
4. Từ đó, có thể kết luận rằng cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là Đại hội Đại biểu toàn quốc.
Đáp án: Cơ quan cao nhất của đảng do Hồ Chí Minh sáng lập là Đại hội Đại biểu toàn quốc。
"""
        system_prompt = f"""You will be provided a question, a list of contexts, and a correct answer. Create a chain of thought (must be in Vietnamese) that leads from the question to the answer using information in the list of contexts.
Example:
{example}"""

        ctx = "\n".join([f"Context {i + 1}: {c}" for i, c in enumerate([context1, context2])])

        prompt = f"""Question: {question}
Correct answer: {answer}
List of context: 
{ctx}
Chain of thought:"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        resp = self.llm.chat(messages)
        return resp
    
    @staticmethod
    def extract_qa(text):
        pattern = r'<question>(.*?)<answer>(.*?)$'
        match = re.match(pattern, text, re.DOTALL)
        if match:
            question = match.group(1).strip().replace('</answer>', '').replace('</question>', '')
            answer = match.group(2).strip().replace('</answer>', '').replace('</question>', '')
            return question, answer
        else:
            raise ValueError("No match found")