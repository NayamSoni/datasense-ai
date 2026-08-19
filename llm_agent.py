import os

from ollama import Client


LOCAL_OLLAMA_HOST = "http://localhost:11434"
LOCAL_OLLAMA_MODEL = "llama3.2:3b"


def get_setting(name, default=""):
    """
    Read a setting from an environment variable or Streamlit Secrets.

    Environment variables work in normal Python execution.
    Streamlit Secrets work in the deployed Streamlit application.
    """

    environment_value = os.getenv(name)

    if environment_value:
        return environment_value

    try:
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


def create_ollama_client():
    """
    Create an Ollama client for either local or cloud execution.
    """

    host = get_setting("OLLAMA_HOST", LOCAL_OLLAMA_HOST)

    configured_model = get_setting("OLLAMA_MODEL")
    model = configured_model or LOCAL_OLLAMA_MODEL

    api_key = get_setting("OLLAMA_API_KEY")

    using_ollama_cloud = host.rstrip("/") == "https://ollama.com"

    if using_ollama_cloud and not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is required when using Ollama Cloud."
        )

    if using_ollama_cloud and not configured_model:
        raise RuntimeError(
            "OLLAMA_MODEL must be configured when using Ollama Cloud."
        )

    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = Client(
        host=host,
        headers=headers,
        timeout=120.0,
    )

    return client, model, using_ollama_cloud


def ask_llm(system_prompt, user_prompt, json_mode=False):
    """
    Send a prompt to the configured Ollama model.

    Locally:
        Uses http://localhost:11434 and llama3.2:3b by default.

    After deployment:
        Uses OLLAMA_HOST, OLLAMA_MODEL, and OLLAMA_API_KEY
        from Streamlit Secrets.
    """

    client, model, using_ollama_cloud = create_ollama_client()

    request_options = {}

    if json_mode and not using_ollama_cloud:
        request_options["format"] = "json"

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        **request_options,
    )

    return response["message"]["content"]