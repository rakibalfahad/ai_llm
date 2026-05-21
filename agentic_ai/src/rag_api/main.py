"""
RAG API — FastAPI application
OpenAI-compatible /v1/chat/completions endpoint with retrieval augmentation.

Queueing design
───────────────
Request queuing is handled by Redis (see queue.py).

  Redis LIST  llm:slot_pool  holds N tokens (one per LLM parallel slot).
  acquire()   BLPOP — blocks until a token is available (FIFO, server-side block).
  release()   RPUSH — returns the token, waking the longest-waiting caller.

Benefits over asyncio.Semaphore:
  • Works across multiple rag-api replicas (horizontal scaling).
  • Survives a replica crash — the next replica's init() resets the pool.
  • FIFO ordering guaranteed by Redis BLPOP semantics.
  • No polling — zero CPU overhead while waiting.

This server still runs as a single uvicorn worker (workers=1).
asyncio handles all concurrent I/O (ChromaDB queries, LLM streams) efficiently
in one event loop. Increase workers only if you add more hardware and Redis
is already in place.

Web search
──────────
Set WEB_SEARCH_ENABLED=true in config/.env to let the assistant fetch live
internet data. Per-request opt-in via "use_web_search": true in the JSON body.
Set WEB_SEARCH_AUTO=true to trigger search automatically when local RAG context
is sparse. Providers: duckduckgo (free) or tavily (requires TAVILY_API_KEY).

Endpoints:
  GET  /health                      — liveness / queue depth / dependency status
  GET  /v1/models                   — list available models (OpenAI compat)
  POST /v1/chat/completions         — main chat endpoint (stream or not)
  POST /v1/rag/search               — debug: raw RAG search results
  GET  /v1/rag/stats                — document counts per collection
  POST /v1/web/search               — debug: raw web search results
"""
import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import llm_client
from auth import verify_api_key
from config import settings
from llm_queue import LLMSlotQueue
from rag import RAGPipeline
import web_search as web_search_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global singletons (initialised at startup) ────────────────────────────────
_rag:   Optional[RAGPipeline]  = None
_queue: Optional[LLMSlotQueue] = None
_redis: Optional[aioredis.Redis] = None


app = FastAPI(
    title="Coding Assistant RAG API",
    description="GPU-backed LLM with RAG over your codebase",
    version="1.0.0",
    docs_url=None,      # disable Swagger UI in production
    redoc_url=None,
    openapi_url=None,   # hide schema endpoint
)

# Allow Open WebUI (same Docker network) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://open-webui:8080", "https://localhost"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.on_event("startup")
async def startup():
    global _rag, _queue, _redis
    logger.info("Connecting to Redis at %s", settings.redis_url)
    _redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,   # we decode manually in queue.py
        socket_connect_timeout=5,
        socket_timeout=settings.llm_queue_timeout + 10,
        health_check_interval=30,
    )
    _queue = LLMSlotQueue(
        redis=_redis,
        slots=settings.llm_parallel_requests,
        queue_timeout=settings.llm_queue_timeout,
    )
    await _queue.init()
    logger.info(
        "LLM slot queue ready: %d slots, %ds timeout",
        settings.llm_parallel_requests,
        settings.llm_queue_timeout,
    )
    logger.info("Loading RAG pipeline...")
    _rag = RAGPipeline()
    logger.info("RAG pipeline ready.")


@app.on_event("shutdown")
async def shutdown():
    if _redis:
        await _redis.aclose()


# ── Pydantic models ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = settings.llm_model_name
    messages: list[Message]
    stream: bool = False
    temperature: float = Field(default=settings.llm_default_temperature, ge=0.0, le=2.0)
    max_tokens: int = Field(default=settings.llm_default_max_tokens, ge=1, le=131072)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    # Custom extension: restrict RAG to one collection
    rag_collection: Optional[str] = Field(
        default=None,
        description="'codebase' or 'enterprise'. None searches both.",
    )
    # Web search: fetch live internet results to augment the response
    use_web_search: bool = Field(
        default=False,
        description=(
            "Fetch live internet results for this request. "
            "Requires WEB_SEARCH_ENABLED=true in the server config."
        ),
    )


class RAGSearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    collection: Optional[str] = None


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    llm_ok = await llm_client.health_check()
    redis_ok = await _queue.health_check() if _queue else False
    try:
        rag_stats = _rag.get_stats() if _rag else {}
        chroma_ok = True
    except Exception:
        rag_stats = {}
        chroma_ok = False

    overall = "ok" if llm_ok and chroma_ok and redis_ok else "degraded"
    return {
        "status": overall,
        "llm":      "connected" if llm_ok    else "unreachable",
        "chromadb": "connected" if chroma_ok else "unreachable",
        "redis":    "connected" if redis_ok  else "unreachable",
        # Live queue visibility
        "llm_slots_total":  settings.llm_parallel_requests,
        "llm_slots_active": await _queue.active_count() if _queue else -1,
        "llm_slots_queued": await _queue.queue_depth()  if _queue else -1,
        "collections": rag_stats,
    }


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """OpenAI-compatible model list."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: ChatCompletionRequest):
    if _rag is None or _queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    req_id = uuid.uuid4().hex[:8]
    t_start = time.monotonic()

    # ── Acquire an LLM slot from Redis ────────────────────────────────────────
    # BLPOP blocks the Redis connection (not the asyncio event loop).
    # The event loop stays free to handle other requests while this one waits.
    queued = await _queue.queue_depth()
    if queued > 0:
        logger.info("[%s] QUEUED  position=%d", req_id, queued + 1)

    token = await _queue.acquire()

    if token is None:
        waited = int(time.monotonic() - t_start)
        logger.warning("[%s] TIMEOUT  waited=%ds", req_id, waited)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Your request waited {waited}s in the queue but no slot became free. "
                "The server is very busy — please try again in a moment."
            ),
            headers={"Retry-After": "15"},
        )

    t_acquired = time.monotonic()
    logger.info(
        "[%s] START  slot=%s  waited=%.1fs  active=%d/%d",
        req_id, token, t_acquired - t_start,
        await _queue.active_count(), settings.llm_parallel_requests,
    )

    try:
        # ── RAG retrieval ─────────────────────────────────────────────────────
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message in request",
            )

        last_query = user_messages[-1].content
        context_docs = _rag.retrieve(last_query, collection=request.rag_collection)
        logger.info("[%s] RAG  %d chunks  query=%.60s", req_id, len(context_docs), last_query)

        # ── Web search (explicit opt-in or auto when RAG is sparse) ───────────
        web_results: list[dict] = []
        should_search = request.use_web_search or (
            settings.web_search_auto
            and len(context_docs) < max(settings.rag_top_k // 2, 1)
        )
        if should_search:
            if settings.web_search_enabled:
                web_results = await web_search_module.web_search(last_query)
                logger.info(
                    "[%s] WEB  %d results  query=%.60s", req_id, len(web_results), last_query
                )
            else:
                logger.warning(
                    "[%s] use_web_search=true but WEB_SEARCH_ENABLED=false — skipping", req_id
                )

        # ── Build augmented prompt ────────────────────────────────────────────
        raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        augmented = _rag.build_augmented_messages(raw_messages, context_docs, web_results)

        # ── Streaming response ────────────────────────────────────────────────
        if request.stream:
            async def event_stream() -> AsyncGenerator[str, None]:
                try:
                    async for line in llm_client.stream_chat_completion(
                        messages=augmented,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                    ):
                        yield line
                finally:
                    # Always release — even if the client disconnects mid-stream.
                    # RPUSH returns the token to Redis so the next waiter wakes up.
                    await _queue.release(token)
                    elapsed = time.monotonic() - t_start
                    logger.info("[%s] DONE(stream)  slot=%s  %.1fs", req_id, token, elapsed)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ── Non-streaming response ────────────────────────────────────────────
        result = await llm_client.chat_completion(
            messages=augmented,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
        result["model"] = settings.llm_model_name
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] ERROR  %s", req_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if not request.stream:
            await _queue.release(token)
            elapsed = time.monotonic() - t_start
            logger.info("[%s] DONE  slot=%s  %.1fs", req_id, token, elapsed)


@app.post("/v1/rag/search", dependencies=[Depends(verify_api_key)])
async def rag_search(request: RAGSearchRequest):
    """Debug endpoint: see raw RAG results for a query."""
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    docs = _rag.retrieve(request.query, k=request.k, collection=request.collection)
    return {"query": request.query, "results": docs}


@app.get("/v1/rag/stats", dependencies=[Depends(verify_api_key)])
async def rag_stats():
    """Returns document counts per ChromaDB collection."""
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready")
    return _rag.get_stats()


@app.post("/v1/web/search", dependencies=[Depends(verify_api_key)])
async def web_search_debug(request: WebSearchRequest):
    """Debug endpoint: run a live web search and return raw results."""
    if not settings.web_search_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search is disabled. Set WEB_SEARCH_ENABLED=true in config/.env.",
        )
    results = await web_search_module.web_search(
        request.query, max_results=request.max_results
    )
    return {
        "query": request.query,
        "provider": settings.web_search_provider,
        "results": results,
    }
