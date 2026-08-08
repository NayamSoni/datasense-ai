import ollama


MODEL = "llama3.2:3b"


def ask_llm(system_prompt, user_prompt, json_mode=False):
    """
    Generic Ollama function.

    json_mode=True forces Ollama's structured-output mode, which
    makes the model emit syntactically valid JSON. This does NOT
    guarantee your specific schema/shape, so you still need to
    validate/parse defensively downstream — but it kills the most
    common failure mode (the model wrapping JSON in prose or
    markdown fences).
    """

    kwargs = {}

    if json_mode:
        kwargs["format"] = "json"

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        **kwargs
    )

    return response["message"]["content"]
