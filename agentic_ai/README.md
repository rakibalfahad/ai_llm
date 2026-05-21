# GPU-Accelerated Coding Assistant — V100 + RAG

A self-hosted, **ChatGPT-like coding assistant** that runs entirely on your server with:
- **Tesla V100-PCIE-16GB** GPU (via your `deeplearning:v100-llm` container)
- **RAG** over your codebase and enterprise documents
- **Secure browser interface** accessible from your laptop
- **No cloud — model and data never leave your server**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR LAPTOP                                                    │
│  Browser  ──── HTTPS (port 443) ──────────────────────────────► │
└─────────────────────────────────────────────────────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  NGINX  (TLS proxy)│  ← server:443
                              └─────────┬──────────┘
                                        │ internal HTTP
                              ┌─────────▼──────────┐
                              │   Open WebUI        │  ← ChatGPT-like UI
                              │   (port 3000)       │
                              └─────────┬──────────┘
                                        │ /v1/chat/completions + API key
                              ┌─────────▼──────────┐
                              │   RAG API           │  ← FastAPI (port 8080)
                              │   (FastAPI)         │    context retrieval
                              └────┬──────┬─────┬───┘
                                   │      │     │
                      ┌────────────▼──┐  ┌▼──────────┐  ┌──────────────────┐
                      │  LLM Server   │  │ ChromaDB   │  │  🌐 Internet      │
                      │  llama-cpp    │  │(vector DB) │  │  (web search)    │
                      │  GPU: V100    │  │ port 8001  │  │  DuckDuckGo /    │
                      │  port 8000    │  └────────────┘  │  Tavily API      │
                      └───────────────┘       ▲          └──────────────────┘
                                              │           (optional, opt-in)
                              ┌───────────────┴────────┐
                              │  Your Code + Docs       │
                              │  (ingested on demand)   │
                              └────────────────────────┘

  All services run in isolated Docker networks.
  The RAG API has outbound internet access (via the frontend network)
  only when WEB_SEARCH_ENABLED=true is set in config/.env.
```

---

## Quick Start (5 steps)

```bash
# 1. Clone / enter this directory
cd /home/ralfahad/projects/ai_llm/agentic_ai

# 2. Generate secrets and SSL certificate
bash setup/03_generate_secrets.sh

# 3. Download the coding LLM (Qwen2.5-Coder-7B, ~4.4 GB GGUF)
bash setup/02_download_model.sh

# 4. Start all services
bash setup/04_start_services.sh

# 5. Ingest your codebase
bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm
```

Then open **https://YOUR_SERVER_IP** from your laptop.

---

## Directory Structure

```
agentic_ai/
├── README.md                    ← this file
├── TUTORIAL.md                  ← full step-by-step tutorial
│
├── docker/
│   ├── docker-compose.yml       ← orchestrates all services
│   ├── Dockerfile.llm-server    ← LLM server (extends deeplearning:v100-llm)
│   ├── Dockerfile.rag-api       ← RAG API server
│   └── nginx/
│       ├── nginx.conf           ← TLS reverse proxy config
│       └── generate_ssl.sh      ← self-signed cert generator
│
├── src/
│   ├── rag_api/
│   │   ├── main.py              ← FastAPI app (OpenAI-compatible endpoint)
│   │   ├── auth.py              ← API key authentication
│   │   ├── rag.py               ← RAG pipeline (ChromaDB + LangChain)
│   │   ├── web_search.py        ← internet retrieval (DuckDuckGo / Tavily)
│   │   ├── llm_client.py        ← streams responses from LLM server
│   │   ├── config.py            ← settings from environment
│   │   └── requirements.txt
│   └── scripts/
│       ├── ingest.py            ← indexes code + docs into ChromaDB
│       └── generate_keys.py     ← secure random key generator
│
├── config/
│   └── .env.example             ← copy to .env and fill in
│
└── setup/
    ├── 01_check_prerequisites.sh
    ├── 02_download_model.sh
    ├── 03_generate_secrets.sh
    ├── 04_start_services.sh
    └── 05_ingest_code.sh
```

---

## Web Search

The assistant can fetch live data from the internet to answer questions about
current events, latest library versions, documentation, etc.

| Setting | Default | Description |
|---------|---------|-------------|
| `WEB_SEARCH_ENABLED` | `false` | Master switch — must be `true` to enable |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` (free) or `tavily` (premium) |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max results per query |
| `WEB_SEARCH_AUTO` | `false` | Auto-search when local RAG context is sparse |
| `TAVILY_API_KEY` | *(empty)* | Required only for the `tavily` provider |

**Enable web search:**
```bash
# Edit config/.env:
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo   # no key needed
# or:
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
```

**Per-request opt-in** (pass in the JSON body):
```json
{ "use_web_search": true }
```

**Test it:**
```bash
source config/.env
curl -s -X POST http://localhost:8080/v1/web/search \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "latest PyTorch release", "max_results": 3}' \
| python3 -m json.tool
```

---

## Security Model

| Layer | Mechanism |
|-------|----------|
| Transport | TLS 1.3 (self-signed cert, HSTS) |
| UI access | Open WebUI user accounts (bcrypt passwords) |
| API access | API key (Bearer token, 32-byte random) |
| Vector DB | ChromaDB auth token (internal network only) |
| LLM server | Internal Docker network only (never exposed) |
| Data | Named Docker volumes, host-path optional |
| Network | `backend` network is `internal: true` (no external routing) |
| Web search | Disabled by default; no key stored in code; opt-in per request |

---

## Prerequisites

- Docker >= 24 with `nvidia-container-toolkit`
- `deeplearning:v100-llm` image built (see `../create_docker.sh`)
- `nvidia-smi` showing V100

---

## Read the Full Tutorial

→ [TUTORIAL.md](TUTORIAL.md)
