

JUDGE_PROMPT = """Given a question, an answer of the model and a ground truth label, determine if the answer is correct or not.
Question: {question}
Answer: {answer}
Label: {ground_truth_label}
Is the answer determined to be correct or not? (yes/no only without any explanation)"""


GENERATE_ANSWER_PROMPT = """Answer the question only based on the document provided.
Document: 
{context}

Question: {question}
"""

N_CONCURENTS = 50