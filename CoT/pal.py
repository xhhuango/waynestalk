import re

from question import QUESTION
from utilities import llm

PROMPT = """
Write a Python function solve() that solves the problem below.
The function must return ONLY the final answer as an integer.

Problem:
{input}

Return ONLY the python code.

The Python code must define a function:

def solve():
    # compute the answer
    return <integer>
"""


def generate_code(question: str) -> str:
    prompt = PROMPT.format(input=question)
    output = llm(prompt)
    match = re.search(r"```(.*?)```", output, re.DOTALL)
    return match.group(1).strip()


def execute_code(code: str) -> int:
    local_vars = {}
    global_vars = {"__builtins__": {"range": range, "len": len}}
    exec(code, global_vars, local_vars)

    if "solve" not in local_vars:
        raise ValueError("solve() function not defined in code")
    return local_vars["solve"]()


def pal(question: str) -> int:
    code = generate_code(question)
    return execute_code(code)


if __name__ == "__main__":
    answer = pal(QUESTION)
    print(f"Program-aided Language Model, PAL\n\nQuestion:\n{QUESTION}\n\nAnswer:\n{answer}")
