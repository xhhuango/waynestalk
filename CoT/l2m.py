import re

from question import QUESTION
from utilities import llm

SUBPROBLEM_GENERATOR_PROMPT = """
Break the following problem into simple subproblems.
Each subproblem should be a small step required to solve the final question.

Return ONLY subproblems per each line. No explanation. Not solution.

Problem:
{input}

Output format:
SP 1: "subproblem 1"
SP 2: "subproblem 2"
...
"""

SUBPROBLEM_PROMPT = """
You are solving a complex problem by breaking it into smaller subproblems.

Original problem:
{input}

Previously solved subproblems:
{solved_subproblems}

Now solve the next subproblem:
{next_subproblem}

Return ONLY the short answer.
"""

FINAL_PROMPT = """
Use the solved subproblems to answer the final question.

Original problem:
{input}

Solved subproblems:
{solve_subproblems}

Based on the above, give the final answer to the original question.
Return ONLY the short answer.
"""


def generate_subproblems(question: str) -> list[str]:
    prompt = SUBPROBLEM_GENERATOR_PROMPT.format(input=question)
    output = llm(prompt)
    lines = output.strip().splitlines()
    subproblems = []
    for line in lines:
        match = re.match(r"\s*SP\s*\d+\s*:\s*(.+)", line.strip())
        if match:
            subproblems.append(match.group(1).strip())
    return subproblems


def solve_subproblem(question: str, next_subproblem: str, solved: list[tuple[str, str]]):
    solved_block = "\n".join(f"{i + 1}. {p} -> {a}" for i, (p, a) in enumerate(solved)) if solved else "(none)"
    prompt = SUBPROBLEM_PROMPT.format(input=question, solved_subproblems=solved_block, next_subproblem=next_subproblem)
    return llm(prompt)


def solve_final_problem(question: str, subproblems, answers):
    solved_subproblems = []
    for i, (sp, ans) in enumerate(zip(subproblems, answers), 1):
        solved_subproblems.append(f"{i}. Subproblem: {sp}")
        solved_subproblems.append(f"   Answer: {ans}")
    sp_block = "\n".join(solved_subproblems)

    prompt = FINAL_PROMPT.format(input=question, solve_subproblems=sp_block)
    return llm(prompt)


def l2m(question: str) -> str:
    subproblems = generate_subproblems(question)

    solved_pairs = []
    answers = []
    for i in range(len(subproblems)):
        answer_i = solve_subproblem(question, subproblems[i], solved_pairs)
        solved_pairs.append((subproblems[i], answer_i))
        answers.append(answer_i)

    return solve_final_problem(question, subproblems, answers)


if __name__ == "__main__":
    answer = l2m(QUESTION)
    print(f"Least-To-Most Prompting, L2M\n\nQuestion:\n{QUESTION}\n\nAnswer:\n{answer}")
