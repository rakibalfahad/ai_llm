"""
Web Search — fetches live results from the internet to augment LLM responses.

Providers (selected by WEB_SEARCH_PROVIDER in .env):
  duckduckgo  — free, no API key required (duckduckgo-search package)
  tavily      — high-quality, LLM-optimised results (requires TAVILY_API_KEY)

Both return a list of:
  {'title': str, 'url': str, 'content': str, 'score': float, 'source': str}

Auto-fallback: if Tavily is configured but fails, DuckDuckGo is tried next.

Network note:
  The rag-api container sits on the Docker `frontend` network (not internal),
  so outbound internet access is available without any extra configuration.
"""
import asyncio
import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: Optional[int] = None) -> list[dict]:
    """
    Run a web search and return result dicts.
    Returns an empty list if web search is disabled or all providers fail.
    """
    if not settings.web_search_enabled:
        return []

    n = max_results or settings.web_search_max_results
    provider = settings.web_search_provider.lower()

    if provider == "tavily":
        results = await _tavily_search(query, n)
        if results:
            return results
        logger.warning("Tavily search returned no results, falling back to DuckDuckGo")
        return await _ddg_search(query, n)

    # default: duckduckgo
    return await _ddg_search(query, n)


async def _tavily_search(query: str, max_results: int) -> list[dict]:
    """Search via Tavily API (premium, purpose-built for LLMs)."""
    if not settings.tavily_api_key:
        logger.warning("WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set")
        return []
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]

        client = TavilyClient(api_key=settings.tavily_api_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            ),
        )
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": round(float(r.get("score", 0.0)), 4),
                    "source": "web:tavily",
                }
            )
        logger.info("Tavily returned %d results for query: %.60s", len(results), query)
        return results
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []


async def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Search via DuckDuckGo (free, no API key required)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        loop = asyncio.get_event_loop()

        def _sync_search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await loop.run_in_executor(None, _sync_search)
        results = []
        for i, r in enumerate(raw):
            # Assign a rank-decay score (1.0 → 0.1) since DuckDuckGo gives no scores
            score = round(max(1.0 - i * 0.1, 0.1), 4)
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "score": score,
                    "source": "web:duckduckgo",
                }
            )
        logger.info("DuckDuckGo returned %d results for query: %.60s", len(results), query)
        return results
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
