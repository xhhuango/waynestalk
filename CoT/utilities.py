from ollama import ChatResponse, chat


def llm(prompt: str, temperature: float = 0.8) -> str:
    response: ChatResponse = chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response.message.content
