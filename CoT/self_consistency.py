from collections import Counter
import re

from question import QUESTION
from utilities import llm

PROMPT = """
Solve the following problem step by step.

{premise_block}

Let's think step by step.
{question_sentence}
""".strip()


def build_prompt(premise_sentences: list[str], question_sentence: str) -> str:
    premise_block = "\n".join(premise_sentences)
    return PROMPT.format(premise_block=premise_block, question_sentence=question_sentence)


def extract_final_answer(text) -> str | None:
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return numbers[-1]


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def identify_question_index(sentences: list[str]) -> int:
    numbered = []
    for i, s in enumerate(sentences, 1):
        numbered.append(f"{i}. {s}")

    numbered_text = "\n".join(numbered)

    prompt = f"""
Which sentence is the main question the user wants answered?
Return ONLY the number. Do not write anything else. Do not repeat the sentence.

{numbered_text}

Answer with a single number (1, 2, ..., or {len(numbered)}):
"""

    output = llm(prompt)
    match = re.search(r"\d+", output)
    if not match:
        raise ValueError(f"LLM failed to output a number: {output}")
    return int(match.group())


def solve_problem(prompt, n=10) -> str:
    answers = []
    for i in range(n):
        output = llm(prompt, temperature=0.8)
        answer = extract_final_answer(output)
        if answer is not None:
            answers.append(answer)

    if not answers:
        raise ValueError("No valid answers from LLM.")

    counts = Counter(answers)
    final_answer = counts.most_common(1)[0][0]
    return final_answer


def self_consistency(question: str) -> str:
    sentences = split_sentences(question)
    question_index = identify_question_index(sentences)
    question_sentence = sentences[question_index - 1]

    premise_sentences = [s for i, s in enumerate(sentences, 1) if i != question_index]
    prompt = build_prompt(premise_sentences, question_sentence)
    return solve_problem(prompt)


if __name__ == "__main__":
    answer = self_consistency(QUESTION)
    print(f"Self-Consistency\n\nQuestion:\n{QUESTION}\n\nAnswer:\n{answer}")
