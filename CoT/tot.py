import json
import re

from question import QUESTION
from utilities import llm

COT_PROMPT = """
Here are some examples of input-output pairs for solving simple reasoning problems.

Input:
John has 4 apples. He buys 3 more. How many apples does he have?
Output:
John has a total of 7 apples.

Input:
Alice has 10 candies. She gives 4 to Bob. How many candies does Alice have left?
Output:
Alice has 6 candies left.

Input:
Tom has 8 pencils. He loses 3 and buys 2 more. How many pencils does he have now?
Output:
Tom has 7 pencils now.

Now solve the following problem.

Input:
{input}
""".strip()

PROPOSE_PROMPT = """
{input}

Please propose {k} different possible outputs (answers) for the last Input.
Return them as a JSON list of strings, for example:
["...", "...", ...]
""".strip()

PROPOSE_WITH_ATTEMPT_PROMPT = """
{input}

Here is the current attempt at solving the problem:

{current_attempt}

You may refine, correct, or provide alternative outputs based on this attempt.

Please propose {k} different possible outputs (answers) for the last Input.
Return them as a JSON list of strings, for example:
["...", "...", ...]
""".strip()

VALUE_PROMPT = """
Evaluate how correct the following output is for the given input.

Input:
{input}

Output:
{output}

Give a score between 0 and 1, where:
- 1 means "definitely correct and consistent with the input"
- 0 means "definitely incorrect or inconsistent"

Return ONLY this JSON:
{{"value": <number>}}
""".strip()


def propose_prompt_wrap(question: str, current_attempt: str | None, k: int) -> str:
    cot_question = COT_PROMPT.format(input=question)
    if current_attempt:
        prompt = PROPOSE_WITH_ATTEMPT_PROMPT.format(input=cot_question, current_attempt=current_attempt, k=k)
    else:
        prompt = PROPOSE_PROMPT.format(input=cot_question, k=k)
    return prompt


def propose_thoughts(question: str, current_attempt: str | None, k: int) -> list[str]:
    prompt = propose_prompt_wrap(question, current_attempt, k)
    output = llm(prompt)
    match = re.search(r"\[.*]", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON list found in LLM output:\n" + output)
    json_array = json.loads(match.group())
    return [thought.strip() for thought in json_array if isinstance(thought, str) and thought.strip()]


def evaluate_thought(question: str, thought: str) -> float:
    prompt = VALUE_PROMPT.format(input=question, output=thought)
    output = llm(prompt)
    match = re.search(r"\{.*}", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output:\n" + output)
    obj = json.loads(match.group())
    value = float(obj["value"])
    return max(0.0, min(1.0, value))


def tot_bfs(question: str, step: int = 3, size: int = 3, breadth: int = 3) -> str:
    current_attempts: list[str | None] = [None]

    for d in range(step):
        next_attempts = []

        for attempt in current_attempts:
            thoughts = propose_thoughts(question=question, current_attempt=attempt, k=size)

            scored = []
            for thought in thoughts:
                score = evaluate_thought(question, thought)
                scored.append((score, thought))

            scored.sort(key=lambda x: x[0], reverse=True)
            next_attempts.extend([t for _, t in scored[:breadth]])

        current_attempts = next_attempts[:breadth]

    return current_attempts[0]


if __name__ == "__main__":
    answer = tot_bfs(QUESTION)
    print(f"Tree-of-Thought, ToT\n\nQuestion:\n{QUESTION}\n\nAnswer:\n{answer}")
