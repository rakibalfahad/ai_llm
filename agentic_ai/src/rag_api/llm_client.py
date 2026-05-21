"""
LLM client — forwards augmented requests to the llama-cpp-python server
and streams back the response.
"""
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Persistent async HTTP client — one pool shared across all FastAPI workers.
# max_connections matches LLM_PARALLEL_REQUESTS (3) × RAG_API_WORKERS (4) with headroom.
_client = httpx.AsyncClient(
    base_url=settings.llm_server_url,
    timeout=httpx.Timeout(settings.llm_request_timeout),
    limits=httpx.Limits(
        max_keepalive_connections=settings.llm_parallel_requests * 2,
        max_connections=settings.llm_parallel_requests * 4,
    ),
)


async def stream_chat_completion(
    messages: list[dict],
    temperature: float = settings.llm_default_temperature,
    max_tokens: int = settings.llm_default_max_tokens,
    top_p: float = 0.95,
) -> AsyncIterator[str]:
    """
    Yields raw SSE lines from the LLM server.
    Each yielded value is a complete 'data: ...' SSE line (or 'data: [DONE]').
    """
    payload = {
        "model": settings.llm_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True,
    }

    async with _client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line:
                yield line + "\n"
        yield "data: [DONE]\n\n"


async def chat_completion(
    messages: list[dict],
    temperature: float = settings.llm_default_temperature,
    max_tokens: int = settings.llm_default_max_tokens,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Non-streaming chat completion — returns the full response dict."""
    payload = {
        "model": settings.llm_model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }
    response = await _client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()


async def health_check() -> bool:
    """Returns True if the LLM server is responsive."""
    try:
        resp = await _client.get("/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
