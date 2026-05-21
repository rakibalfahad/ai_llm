"""Configuration — loaded from environment variables (Docker .env file)."""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── LLM Server ────────────────────────────────────────────────────────────
    llm_server_url: str = Field(
        default="http://llm-server:8000",
        description="Base URL of the llama-cpp-python OpenAI-compatible server",
    )
    llm_model_name: str = Field(
        default="coding-assistant",
        description="Model name returned in API responses",
    )
    llm_default_max_tokens: int = Field(default=2048)
    llm_default_temperature: float = Field(default=0.1)
    llm_request_timeout: int = Field(
        default=300,
        description="Seconds to wait for a streaming response from the LLM server",
    )
    llm_parallel_requests: int = Field(
        default=3,
        description="Must match LLM_PARALLEL_REQUESTS in docker-compose (sets semaphore size)",
    )
    llm_queue_timeout: int = Field(
        default=120,
        description=(
            "Seconds a request may wait in the Redis queue before getting a 503. "
            "Set to 0 to reject immediately when all slots are busy (no queuing)."
        ),
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(
        description=(
            "Redis connection URL. Format: redis://:<password>@<host>:<port>/<db>. "
            "Set automatically by docker-compose from REDIS_PASSWORD env var."
        ),
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chromadb_host: str = Field(default="chromadb")
    chromadb_port: int = Field(default=8000)
    chroma_token: str = Field(description="ChromaDB auth token from .env")
    chroma_code_collection: str = Field(default="codebase")
    chroma_docs_collection: str = Field(default="enterprise")

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace sentence-transformers model for embedding",
    )
    embedding_device: str = Field(
        default="cpu",
        description="'cpu' or 'cuda'. Use cpu to save VRAM for the LLM.",
    )

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    rag_top_k: int = Field(
        default=5,
        description="Number of document chunks to retrieve per query",
    )
    rag_score_threshold: float = Field(
        default=0.3,
        description="Minimum cosine similarity score (0–1) to include a chunk",
    )

    # ── Web Search ────────────────────────────────────────────────────────────
    web_search_enabled: bool = Field(
        default=False,
        description=(
            "Set to true to allow the assistant to search the internet. "
            "Requires outbound internet access from the rag-api container "
            "(already available via the Docker frontend network)."
        ),
    )
    web_search_provider: str = Field(
        default="duckduckgo",
        description="'duckduckgo' (free, no key) or 'tavily' (requires TAVILY_API_KEY)",
    )
    web_search_max_results: int = Field(
        default=5,
        description="Maximum web results to fetch per query",
    )
    web_search_auto: bool = Field(
        default=False,
        description=(
            "Automatically add web search when RAG returns fewer chunks than "
            "rag_top_k // 2. Requires web_search_enabled=true."
        ),
    )
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key (https://tavily.com). Only needed when web_search_provider=tavily.",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    rag_api_key: str = Field(description="Bearer token required for all API requests")

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt_file: str = Field(default="/app/system_prompt.txt")

    class Config:
        env_file = "/app/../config/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
