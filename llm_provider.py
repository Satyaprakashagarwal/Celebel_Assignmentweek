"""
Returns the chat LLM used to answer questions.

Uses Groq's free, cloud-hosted API for extremely fast inference. Get a free
API key at https://console.groq.com/keys and set it as the GROQ_API_KEY
environment variable (in a .env file) or paste it into the Streamlit sidebar.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_llm(groq_api_key: str | None = None):
    """
    groq_api_key: lets the caller (e.g. a Streamlit sidebar field) pass a key
                  explicitly instead of relying on the environment variable.
    """
    key = groq_api_key or os.environ.get("GROQ_API_KEY")

    if not key:
        raise ValueError(
            "No GROQ_API_KEY was provided. "
            "Get a free key at https://console.groq.com/keys and set it "
            "in your .env file (GROQ_API_KEY=...) or paste it into the "
            "Streamlit sidebar."
        )

    from langchain_groq import ChatGroq

    return ChatGroq(model=GROQ_MODEL, api_key=key, temperature=0.2)
