"""
claude_client.py — Streaming Claude wrapper.

Yields raw text strings consumed by the Flask SSE generator.
"""

import os
import anthropic
from dotenv import load_dotenv
from config import CLAUDE_MODEL, MAX_TOKENS, SYSTEM_PROMPT

_client = None


def _get_client():
    global _client
    load_dotenv(override=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
    if _client is None or _client.api_key != api_key:
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def stream_response(messages: list[dict]):
    """
    Stream a response from Claude.

    messages: list of {"role": "user"|"assistant", "content": str}

    Yields text strings as they arrive. Thinking blocks are dropped.
    """
    client = _get_client()

    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield delta.text
