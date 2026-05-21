# Step-by-Step Tutorial: GPU Coding Assistant with RAG

**Tesla V100-PCIE-16GB · deeplearning:v100-llm · Secure Browser Interface**

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [How It All Works](#2-how-it-all-works)
3. [Step 1 — Prerequisites Check](#step-1--prerequisites-check)
4. [Step 2 — Choosing Your LLM Model](#step-2--choosing-your-llm-model)
5. [Step 3 — Download the Model](#step-3--download-the-model)
6. [Step 4 — Generate Secrets & SSL](#step-4--generate-secrets--ssl)
7. [Step 5 — Configure the Stack](#step-5--configure-the-stack)
8. [Step 6 — Build & Start All Services](#step-6--build--start-all-services)
9. [Step 7 — Ingest Your Code & Enterprise Data](#step-7--ingest-your-code--enterprise-data)
10. [Step 8 — Connect from Your Laptop](#step-8--connect-from-your-laptop)
11. [Step 9 — Using the Coding Assistant](#step-9--using-the-coding-assistant)
12. [Step 10 — Enable Web Search](#step-10--enable-web-search)
13. [Step 11 — Security Hardening](#step-11--security-hardening)
14. [Monitoring & Maintenance](#monitoring--maintenance)
15. [Troubleshooting](#troubleshooting)
16. [Advanced: Switching to vLLM](#advanced-switching-to-vllm)

---

## 1. Overview & Goals

By the end of this tutorial you will have a **fully local, GPU-accelerated AI coding assistant** that:

- **Runs entirely on your server** — model weights and data never leave your machine
- Serves a **ChatGPT-like browser interface** accessible from any device on your network
- Uses **Retrieval-Augmented Generation (RAG)** to answer questions grounded in:
  - Your actual Python/code files
  - Enterprise documents (PDFs, Markdown, text)
- Is **secured** end-to-end: TLS, user authentication, API keys, network isolation

The stack uses your existing `deeplearning:v100-llm` Docker image and adds:

| Component | Role | Image |
|-----------|------|-------|
| **llama-cpp-python** server | Run the LLM on GPU | `deeplearning:v100-llm` |
| **ChromaDB** | Store & search embedded vectors | `chromadb/chroma` |
| **Redis** | Distributed request queue (FIFO slot management) | `redis:7-alpine` |
| **RAG API** (FastAPI) | Augment queries with context, manage queue, web search | custom (Python 3.11) |
| **Open WebUI** | ChatGPT-like browser frontend | `ghcr.io/open-webui/open-webui` |
| **Nginx** | TLS termination, reverse proxy | `nginx:alpine` |

---

## 2. How It All Works

### The RAG Loop

When you type a question in the browser:

```
1.  Browser    ─► Open WebUI
2.  Open WebUI ─► POST /v1/chat/completions  (API key in header)
                  ─► RAG API (FastAPI)
3.  RAG API    ─► BLPOP llm:slot_pool  (Redis)
                      • If a slot is free  → pops a token immediately
                      • If all slots busy  → waits in FIFO queue until one frees
4.  RAG API    ─► Embed your question with a local embedding model
5.  RAG API    ─► Query ChromaDB: find the top-5 most relevant code/doc chunks
6.  RAG API    ─► [Optional] Web search: fetch live internet results
                      • Triggered by "use_web_search": true in the request
                      • Or automatically when RAG finds fewer than rag_top_k/2 chunks
                        (requires WEB_SEARCH_AUTO=true)
7.  RAG API    ─► Build an augmented prompt:
                      [SYSTEM: You are a coding assistant...]
                      [CONTEXT: <retrieved code chunks>]          ← from ChromaDB
                      [WEB: <live search results>]               ← from internet (optional)
                      [USER: <your question>]
8.  RAG API    ─► Forward augmented prompt to LLM server (llama-cpp)
9.  LLM server ─► Run inference on V100 GPU, stream tokens
10. RAG API    ─► RPUSH llm:slot_pool  (Redis) — release slot, wake next waiter
11. RAG API    ─► Stream tokens back to Open WebUI
12. Open WebUI ─► Stream to your browser
```

### How Redis Queues Requests

The GPU can only run a fixed number of inference jobs at once (`LLM_PARALLEL_REQUESTS`, default 3). Redis enforces this limit fairly:

```
Redis LIST  llm:slot_pool = ["slot:0", "slot:1", "slot:2"]   (3 tokens)

User A ─► BLPOP ─► gets "slot:0" ─► GPU inference ─► RPUSH "slot:0"
User B ─► BLPOP ─► gets "slot:1" ─► GPU inference ─► RPUSH "slot:1"
User C ─► BLPOP ─► gets "slot:2" ─► GPU inference ─► RPUSH "slot:2"

User D ─► BLPOP ─► list empty → waits  (no CPU spin, blocked server-side)
User E ─► BLPOP ─► list empty → waits

User A finishes ─► RPUSH "slot:0" → Redis wakes User D  (FIFO)
User B finishes ─► RPUSH "slot:1" → Redis wakes User E
```

If a user's browser tab is closed mid-stream, the `finally` block in the streaming generator still calls `RPUSH` — so no slot is ever permanently lost.

### Network Isolation

```
[Internet / your laptop]
        │
        │ HTTPS:443
        ▼
    ┌──────────────────────────────────────────────┐
    │ EXTERNAL Docker network                      │
    │   ┌──────────┐     ┌──────────────────────┐ │
    │   │  Nginx   │────►│   Open WebUI :3000   │ │
    │   └──────────┘     └──────────┬───────────┘ │
    └────────────────────────────────────────────-─┘
                                    │
                     ┌──────────────▼─────────────────────────────┐
                     │ INTERNAL Docker network (no egress)         │
                     │                                             │
                     │  ┌──────────────────┐                      │
                     │  │  RAG API :8080   │                      │
                     │  └───┬─────┬────┬───┘                      │
                     │      │     │    │                           │
                     │  ┌───▼──┐ ┌▼────────┐ ┌─────────────────┐ │
                     │  │ LLM  │ │ChromaDB │ │  Redis  :6379   │ │
                     │  │:8000 │ │  :8000  │ │  (slot queue)   │ │
                     │  └──────┘ └─────────┘ └─────────────────┘ │
                     └───────────────────────────────────────────-┘
```

Redis, the LLM server, and ChromaDB have **zero** external exposure — internal network only.

---

## Step 1 — Prerequisites Check

Run the check script:

```bash
cd /home/ralfahad/projects/ai_llm/agentic_ai
bash setup/01_check_prerequisites.sh
```

What it verifies:
- Docker daemon is running
- `nvidia-container-toolkit` is installed
- Your `deeplearning:v100-llm` image exists
- V100 GPU is visible (`nvidia-smi`)
- At least 20 GB free disk space (for model + data)
- Ports 80 and 443 are available

**Manual check:**

```bash
# Verify GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Verify Docker image
docker images deeplearning:v100-llm

# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Expected output for GPU check:
```
Tesla V100-PCIE-16GB, 16160 MiB, 580.105.08
```

---

## Step 2 — Choosing Your LLM Model

Your V100 has **16 GB** of VRAM. Here are recommended models with their VRAM requirements:

| Model | Format | VRAM | Strengths | HuggingFace Link |
|-------|--------|------|-----------|------------------|
| **Qwen2.5-Coder-7B-Instruct Q4_K_M** ⭐ | GGUF | ~4.4 GB | Best code quality, fast | [bartowski/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF) |
| DeepSeek-Coder-6.7B-Instruct Q4_K_M | GGUF | ~4.0 GB | Excellent code, multilingual | [TheBloke/deepseek-coder-6.7B-instruct-GGUF](https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF) |
| CodeLlama-13B-Instruct Q4_K_M | GGUF | ~7.4 GB | Large context, explanations | [TheBloke/CodeLlama-13B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF) |
| Phi-3.5-mini-instruct Q4_K_M | GGUF | ~2.4 GB | Fastest inference | [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

**Recommendation:** Start with **Qwen2.5-Coder-7B-Instruct Q4_K_M**. It leaves ~12 GB free for KV cache, giving you long context windows (32K tokens) and fast inference.

**Why GGUF / llama-cpp?**
- Already installed in `deeplearning:v100-llm`
- 4-bit quantization: models 4× smaller with <5% quality loss
- Offloads all layers to GPU (`-ngl -1`)
- Built-in OpenAI-compatible server

---

## Step 3 — Download the Model

All models are downloaded from **[HuggingFace Hub](https://huggingface.co)** — the standard public repository for open-source AI models.

> **Direct link to the recommended model:**
> **https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF**
> File: `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (~4.4 GB)

The automated script downloads it for you:

```bash
bash setup/02_download_model.sh
```

This script:
1. Creates `./models/` directory (mounted into the LLM container)
2. Downloads the GGUF model from HuggingFace Hub via `huggingface_hub` Python client
3. Falls back to `wget` if the Python client is unavailable

**Manual download (if behind proxy or prefer direct control):**

```bash
mkdir -p ./models

# Option A: huggingface-cli Python client (respects proxy env vars)
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='bartowski/Qwen2.5-Coder-7B-Instruct-GGUF',
    filename='Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf',
    local_dir='./models'
)
print('Download complete')
"

# Option B: direct wget
wget -P ./models \
  "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
```

**All model direct download URLs:**

| Model | Direct download URL |
|-------|---------------------|
| Qwen2.5-Coder-7B Q4_K_M ⭐ | https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf |
| DeepSeek-Coder-6.7B Q4_K_M | https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf |
| CodeLlama-13B Q4_K_M | https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf |
| Phi-3.5-mini Q4_K_M | https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf |

**Behind Intel proxy?** The `http_proxy` and `https_proxy` variables in `config/.env` are passed into the containers, so downloads inside Docker will use the proxy automatically.

**Directory structure after download:**
```
agentic_ai/
└── models/
    └── Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf   (~4.4 GB)
```

> **Security note:** Verify the model file before use. Never run GGUF files from untrusted sources — they contain executable compute graphs.
> The [`bartowski`](https://huggingface.co/bartowski) HuggingFace account is a well-known, community-trusted quantizer. You can inspect all files and their checksums directly on the model page before downloading.

---

## Step 4 — Generate Secrets & SSL

```bash
bash setup/03_generate_secrets.sh
```

This generates:
1. A **32-byte random API key** for the RAG API (`RAG_API_KEY`)
2. A **32-byte random token** for ChromaDB auth (`CHROMA_TOKEN`)
3. A **32-byte Open WebUI secret** (`WEBUI_SECRET_KEY`)
4. A **32-byte Redis password** (`REDIS_PASSWORD`) — used by both the Redis server and the RAG API client
5. A **self-signed TLS certificate** for Nginx (valid 10 years for internal use)
6. Writes all secrets into `config/.env` (git-ignored)

**Why self-signed?** You are on an internal network. If you have a domain name pointing to this server, replace the self-signed cert with a Let's Encrypt cert (instructions in Step 10).

**Security:** The `config/.env` file is created with `chmod 600`. Never commit it to version control. The `.gitignore` is already configured to exclude it.

**Manual inspection:**
```bash
cat config/.env        # review generated secrets
ls -la docker/nginx/ssl/   # review cert
openssl x509 -text -noout -in docker/nginx/ssl/server.crt | grep -E "Subject|Not (Before|After)"
```

---

## Step 5 — Configure the Stack

Copy and edit the environment file:

```bash
# Already done by 03_generate_secrets.sh, but review these settings:
nano config/.env
```

**Key settings to verify:**

```bash
# ── Model ──────────────────────────────────────────────────
# Filename inside ./models/ directory
LLM_MODEL_FILE=Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

# Context window (tokens). 32768 for Qwen2.5-Coder, 4096 for CodeLlama
LLM_CONTEXT_SIZE=32768

# GPU layers to offload (-1 = all layers, maximizes GPU use)
LLM_GPU_LAYERS=-1

# ── Embedding model (for RAG) ───────────────────────────────
# Runs inside the RAG API container; loaded into GPU memory
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# ── Server ─────────────────────────────────────────────────
# IP or hostname that your laptop will use to reach the server
SERVER_HOST=0.0.0.0

# ── Security (auto-generated, do not change) ───────────────
RAG_API_KEY=<auto-generated>
CHROMA_TOKEN=<auto-generated>
WEBUI_SECRET_KEY=<auto-generated>
```

**Optional: Custom system prompt**

Edit `src/rag_api/system_prompt.txt` to give the assistant context about your company/project:

```
You are an expert Python and AI/ML engineer assistant for [YOUR COMPANY].
You have deep knowledge of PyTorch, TensorFlow, HuggingFace, and LangChain.
When answering questions about code, always reference the provided context first.
Be concise, accurate, and proactive about pointing out potential bugs or improvements.
```

---

## Step 6 — Build & Start All Services

```bash
bash setup/04_start_services.sh
```

This runs `docker compose up -d --build` which:

1. **Builds** `Dockerfile.llm-server` (adds model serving deps on top of `deeplearning:v100-llm`)
2. **Builds** `Dockerfile.rag-api` (lightweight FastAPI container)
3. **Pulls** `chromadb/chroma`, `redis:7-alpine`, `ghcr.io/open-webui/open-webui`, `nginx:alpine`
4. **Starts** all six services in the correct order
5. Runs a **health-check** loop until all services are ready

**First boot takes ~3-5 minutes** (embedding model download, GGUF model load into GPU, Redis slot pool initialisation).

**Verify all services are healthy:**

```bash
docker compose -f docker/docker-compose.yml ps
```

Expected output:
```
NAME              IMAGE                    STATUS
llm-server        deeplearning:v100-llm    Up (healthy)
chromadb          chromadb/chroma          Up (healthy)
redis             redis:7-alpine           Up (healthy)
rag-api           coding-assistant-api     Up (healthy)
open-webui        open-webui               Up (healthy)
nginx             nginx:alpine             Up
```

**Check GPU is being used:**
```bash
watch -n 2 nvidia-smi
```
You should see the `llama-cpp` process using ~4-5 GB of GPU memory.

**Check LLM server is responding:**
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# Expected: {"status": "ok"}

curl -s http://localhost:8000/v1/models | python3 -m json.tool
# Expected: list with your model
```

**Check RAG API:**
```bash
# Replace with your actual RAG_API_KEY from config/.env
source config/.env
curl -s -H "Authorization: Bearer $RAG_API_KEY" http://localhost:8080/health | python3 -m json.tool
```

Expected response (all services healthy, no one queued):
```json
{
  "status": "ok",
  "llm": "connected",
  "chromadb": "connected",
  "redis": "connected",
  "llm_slots_total": 3,
  "llm_slots_active": 0,
  "llm_slots_queued": 0
}
```

`llm_slots_active` shows how many users are currently generating. `llm_slots_queued` shows how many are waiting in the Redis queue.

---

## Step 7 — Ingest Your Code & Enterprise Data

This is what makes the assistant actually useful for your specific codebase.

### 7.1 Ingest Your Python Code

```bash
# Ingest the entire ai_llm project (Python files, notebooks, scripts)
bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm

# Or ingest a specific directory
bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm/deeplearning/scripts
```

The script:
- Finds all `.py`, `.ipynb`, `.md`, `.sh` files
- Splits them into overlapping 1000-token chunks
- Embeds each chunk using `BAAI/bge-small-en-v1.5` (GPU-accelerated)
- Stores vectors + metadata in ChromaDB

### 7.2 Ingest Enterprise Documents

```bash
# Ingest PDFs, text files, Markdown documentation
docker compose -f docker/docker-compose.yml exec rag-api \
  python3 /app/scripts/ingest.py \
  --source /app/data/enterprise_docs \
  --collection enterprise

# Copy your docs to the data volume first:
docker cp /path/to/your/docs/. \
  $(docker compose -f docker/docker-compose.yml ps -q rag-api):/app/data/enterprise_docs/
```

**Supported file types:**
- `.py` `.js` `.ts` `.java` `.cpp` `.go` `.rs` — Source code (language-aware splitting)
- `.md` `.rst` `.txt` — Documentation
- `.pdf` — Research papers, manuals (requires `pypdf2`)
- `.ipynb` — Jupyter notebooks (extracts code + markdown cells)

### 7.3 Verify Ingestion

```bash
source config/.env
curl -s -X POST http://localhost:8080/v1/rag/search \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "how to train a model with PyTorch", "k": 3}' \
| python3 -m json.tool
```

You should see 3 relevant code chunks from your codebase in the response.

### 7.4 Re-ingest When Code Changes

Set up a cron job or a file watcher to re-ingest when files change:

```bash
# Add to crontab (re-ingest every night at 2 AM)
crontab -e
# Add: 0 2 * * * /home/ralfahad/projects/ai_llm/agentic_ai/setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm
```

---

## Step 8 — Connect from Your Laptop

### 8.1 Find Your Server IP

```bash
ip route get 1.1.1.1 | awk '{print $7; exit}'
# or
hostname -I | awk '{print $1}'
```

Example: `192.168.1.50`

### 8.2 Open the Interface

On your laptop, open your browser and navigate to:

```
https://192.168.1.50
```

> **TLS warning:** Since we use a self-signed certificate, your browser will show a warning.
> - Chrome/Edge: Click "Advanced" → "Proceed to 192.168.1.50 (unsafe)"
> - Firefox: Click "Advanced" → "Accept the Risk and Continue"
> - Safari: Click "Show Details" → "visit this website"
>
> This is expected for internal/private deployments. See Step 10 for a proper cert.

### 8.3 Create Your Account

1. On first visit, Open WebUI shows a **Sign Up** page
2. Create an admin account (first account is automatically admin)
3. You are now logged in

### 8.4 Configure the API Connection

Open WebUI needs to know where the RAG API is:

1. Click the gear icon (⚙) → **Settings**
2. Go to **Connections**
3. Under "OpenAI API":
   - **API Base URL**: `http://rag-api:8080/v1` (Docker internal) — already set via env vars
   - **API Key**: your `RAG_API_KEY` from `config/.env` — already set via env vars
4. Click **Save**
5. Go to **Models** — you should see `coding-assistant` in the list

If the model does not appear, check `docker compose logs rag-api`.

---

## Step 9 — Using the Coding Assistant

### Starting a Conversation

1. Click **New Chat**
2. Select model: `coding-assistant`
3. Type your question

### Example Prompts

**Code explanation:**
```
Explain how the CIFAR-10 ResNet training script works.
What is the purpose of the learning rate scheduler in train_cifar10_resnet.py?
```

**Debugging:**
```
My training loss is not decreasing after epoch 5. Here is my training loop:
[paste your code]
What might be wrong?
```

**Code generation:**
```
Write a PyTorch DataLoader for a custom dataset that reads images from a directory.
Use the same pattern as the existing CIFAR-10 code in this project.
```

**RAG-aware questions:**
```
What models have already been trained and saved in this project?
How does the BPE tokenizer in this codebase work?
```

**Web search (real-time internet data):**
```
What is the latest stable version of PyTorch? [use_web_search]
Is there a known bug in transformers 4.40 with LoRA? [use_web_search]
What does the HuggingFace Qwen2.5-Coder model card say about fine-tuning? [use_web_search]
```
> To trigger web search from Open WebUI, start your message with `[use_web_search]`
> or enable `WEB_SEARCH_AUTO=true` in `config/.env` (auto-triggers when local RAG
> context is thin).

### Understanding RAG Responses

The assistant will often say things like:
> "Based on the code in `train_cifar10_resnet.py`, I can see that..."

This means the RAG system found relevant context in your codebase and the LLM is using it to give you a grounded, accurate answer rather than a generic one.

### Conversation Tips

- **Be specific** about file names or function names if you know them
- **Paste code snippets** directly — the model handles up to 32K tokens
- **Ask follow-up questions** — the conversation history is maintained
- Use **"@collection enterprise"** prefix to search only enterprise docs

---

## Step 10 — Enable Web Search

The assistant can search the internet for real-time information such as
library changelogs, documentation, CVEs, or anything not in your codebase.

### 10.1 How It Works

Web search is **disabled by default** and completely opt-in. When enabled:
- Each chat request can include `"use_web_search": true` to trigger a search.
- You can also set `WEB_SEARCH_AUTO=true` to automatically search when local
  RAG results are sparse (fewer than `rag_top_k / 2` chunks found).
- Results are injected into the prompt in a separate `<web_search_results>`
  block, alongside the local `<retrieved_context>` block.
- The LLM cites URLs from web results and prefers local codebase context
  over web results when both are relevant.

**Network note:** The `rag-api` container is already on the Docker `frontend`
network (which has outbound internet access). No changes to `docker-compose.yml`
are needed.

### 10.2 Option A — DuckDuckGo (Free, No API Key)

```bash
# Edit config/.env and add:
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_AUTO=false       # set true to auto-search when RAG is sparse

# Restart the RAG API to pick up the new settings:
docker compose -f docker/docker-compose.yml restart rag-api
```

### 10.3 Option B — Tavily (Premium, Better Quality)

Tavily is purpose-built for LLM use: results are pre-cleaned, ranked by
relevance, and designed to fit into prompts efficiently.

1. Create a free account at https://tavily.com and copy your API key.

```bash
# Edit config/.env and add:
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
WEB_SEARCH_MAX_RESULTS=5

# Restart:
docker compose -f docker/docker-compose.yml restart rag-api
```

If Tavily fails (e.g., quota exceeded), the system automatically falls back
to DuckDuckGo.

### 10.4 Test Web Search

```bash
source config/.env

# Raw search results
curl -s -X POST http://localhost:8080/v1/web/search \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "PyTorch 2.4 release notes", "max_results": 3}' \
| python3 -m json.tool

# Full chat request with web search enabled
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding-assistant",
    "messages": [{"role": "user", "content": "What is the latest stable PyTorch version?"}],
    "use_web_search": true
  }' | python3 -m json.tool
```

### 10.5 Security Considerations

- Web search is disabled by default — no outbound requests are made unless you
  explicitly set `WEB_SEARCH_ENABLED=true`.
- The `TAVILY_API_KEY` is stored in `config/.env` (git-ignored, `chmod 600`).
  Never commit it to version control.
- DuckDuckGo does not require any credentials and sends only the search query.
- All web traffic goes through the rag-api container; the LLM server and
  ChromaDB remain fully isolated on the internal Docker network.

---

## Step 11 — Security Hardening

### 11.1 Change Default Ports (optional)

Edit `config/.env`:
```bash
NGINX_HTTP_PORT=8080
NGINX_HTTPS_PORT=8443
```
Then restart: `docker compose restart nginx`

### 11.2 Replace Self-Signed Cert with Let's Encrypt

If your server has a domain name and is reachable from the internet:

```bash
# Install certbot
apt install certbot

# Get certificate (replace with your domain)
certbot certonly --standalone -d assistant.yourdomain.com

# Update nginx config to use Let's Encrypt paths
sed -i 's|/etc/nginx/ssl/server.crt|/etc/letsencrypt/live/assistant.yourdomain.com/fullchain.pem|' \
  docker/nginx/nginx.conf
sed -i 's|/etc/nginx/ssl/server.key|/etc/letsencrypt/live/assistant.yourdomain.com/privkey.pem|' \
  docker/nginx/nginx.conf

# Mount certs into Nginx container (update docker-compose.yml volumes section)
# Add: - /etc/letsencrypt:/etc/letsencrypt:ro

# Auto-renew
echo "0 3 * * 0 certbot renew --quiet && docker compose restart nginx" | crontab -
```

### 11.3 Restrict Access by IP (optional)

Edit `docker/nginx/nginx.conf` and add inside the `server` block:

```nginx
# Allow only your laptop's IP
allow 192.168.1.100;   # replace with your laptop's IP
deny all;
```

### 11.4 Enable Rate Limiting

Already configured in `nginx.conf`:
- 10 requests/second per IP to the API
- 2 requests/second per IP to the UI login

### 11.5 Rotate API Keys

```bash
# Generate new keys (including a new Redis password)
python3 src/scripts/generate_keys.py

# Update .env
nano config/.env

# Restart affected services
# Redis must restart first to pick up the new password,
# then rag-api to pick up the new REDIS_URL
docker compose -f docker/docker-compose.yml restart redis
docker compose -f docker/docker-compose.yml restart rag-api open-webui
```

### 11.6 Audit Logs

```bash
# View who is making API calls
docker compose -f docker/docker-compose.yml logs rag-api | grep "POST /v1"

# View Nginx access log
docker compose -f docker/docker-compose.yml logs nginx | grep " 200 \| 401 \| 403 "
```

### 11.7 Data Backup

```bash
# Backup ChromaDB (your vector store — all ingested data)
docker run --rm \
  -v coding-assistant_chromadb_data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/chromadb_$(date +%Y%m%d).tar.gz -C /source .

# Backup Redis (slot queue state — lightweight, optional)
docker run --rm \
  -v coding-assistant_redis_data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/redis_$(date +%Y%m%d).tar.gz -C /source .

# Schedule daily backups
crontab -e
# Add: 0 1 * * * docker run --rm -v coding-assistant_chromadb_data:/source:ro -v /backup:/backup alpine tar czf /backup/chromadb_$(date +\%Y\%m\%d).tar.gz -C /source .
```

---

## Monitoring & Maintenance

### Real-time GPU Monitoring

```bash
# Watch GPU usage every 2 seconds
watch -n 2 nvidia-smi

# More detailed: memory, utilization, temp
nvidia-smi dmon -s mu -d 2
```

### Service Health

```bash
# All services status
docker compose -f docker/docker-compose.yml ps

# Last 100 lines of each service
docker compose -f docker/docker-compose.yml logs --tail=100 llm-server
docker compose -f docker/docker-compose.yml logs --tail=100 rag-api
docker compose -f docker/docker-compose.yml logs --tail=100 chromadb
docker compose -f docker/docker-compose.yml logs --tail=100 redis

# Follow logs in real-time
docker compose -f docker/docker-compose.yml logs -f rag-api
```

### Monitor the Redis Queue

The `/health` endpoint shows live queue depth:
```bash
source config/.env
watch -n 2 'curl -s -H "Authorization: Bearer $RAG_API_KEY" \
  http://localhost:8080/health | python3 -m json.tool'
# Watch llm_slots_active and llm_slots_queued update in real time
```

Inspect the Redis slot pool directly:
```bash
source config/.env

# Tokens currently in the pool (= free slots)
docker exec redis redis-cli -a "$REDIS_PASSWORD" LRANGE llm:slot_pool 0 -1

# Number of requests waiting in the queue
docker exec redis redis-cli -a "$REDIS_PASSWORD" GET llm:waiting_count

# All llm:* keys at a glance
docker exec redis redis-cli -a "$REDIS_PASSWORD" KEYS 'llm:*'
```

### Performance Tuning

**Increase throughput** (for multiple simultaneous users):
```bash
# Edit config/.env
LLM_PARALLEL_REQUESTS=4    # process 4 requests in parallel
LLM_BATCH_SIZE=512         # larger batch = better GPU utilization
```

**Reduce latency** (for single user):
```bash
LLM_PARALLEL_REQUESTS=1
LLM_CONTEXT_SIZE=8192      # smaller context = faster prefill
```

### Stopping the Stack

```bash
# Stop all services (keeps data)
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes (WARNING: deletes ChromaDB data!)
docker compose -f docker/docker-compose.yml down -v
```

### Upgrading

```bash
# Pull latest Open WebUI
docker compose -f docker/docker-compose.yml pull open-webui
docker compose -f docker/docker-compose.yml up -d open-webui

# Re-ingest code after adding new files
bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm
```

---

## Troubleshooting

### LLM Server won't start

```bash
docker compose -f docker/docker-compose.yml logs llm-server
```

Common causes:
- **"CUDA out of memory"**: Reduce `LLM_CONTEXT_SIZE` in `.env` to `4096`
- **"model file not found"**: Check `./models/` contains your `.gguf` file; verify `LLM_MODEL_FILE` in `.env`
- **"CUDA not available"**: Run `docker run --rm --gpus all deeplearning:v100-llm nvidia-smi` — if this fails, reinstall `nvidia-container-toolkit`

### ChromaDB auth failures

```bash
# Verify token matches between .env and running container
source config/.env
curl -H "Authorization: Bearer $CHROMA_TOKEN" http://localhost:8001/api/v1/heartbeat
```

### Redis connection errors

```bash
# Check Redis is running and healthy
docker compose -f docker/docker-compose.yml ps redis
docker compose -f docker/docker-compose.yml logs --tail=50 redis

# Ping Redis manually (should return PONG)
source config/.env
docker exec redis redis-cli -a "$REDIS_PASSWORD" ping

# Check rag-api can see Redis
docker compose -f docker/docker-compose.yml logs rag-api | grep -i redis

# Slot pool stuck (all slots checked out after a crash):
# Reset the pool — rag-api does this automatically on restart:
docker compose -f docker/docker-compose.yml restart rag-api

# Or reset manually:
source config/.env
docker exec redis redis-cli -a "$REDIS_PASSWORD" DEL llm:slot_pool llm:waiting_count
# Then restart rag-api to re-initialise:
docker compose -f docker/docker-compose.yml restart rag-api
```

### Open WebUI shows "Connection failed"

```bash
# Check if RAG API is healthy
docker compose -f docker/docker-compose.yml exec open-webui \
  wget -qO- http://rag-api:8080/health
```

### RAG returns empty results

```bash
# Check collection document count
source config/.env
curl -s -H "Authorization: Bearer $RAG_API_KEY" \
  http://localhost:8080/v1/rag/stats | python3 -m json.tool
# Look for "document_count" > 0
# If 0, re-run the ingest script
```

### Slow inference

- Check GPU utilization with `nvidia-smi` — should show ~90%+ during inference
- Reduce `LLM_CONTEXT_SIZE` — KV cache grows linearly with context
- Check if embedding model is sharing GPU memory: set `EMBEDDING_DEVICE=cpu` in `.env` to free VRAM for the LLM

### Web search not working

```bash
# 1. Confirm it is enabled
grep WEB_SEARCH config/.env

# 2. Test the search endpoint directly
source config/.env
curl -s -X POST http://localhost:8080/v1/web/search \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "python requests library", "max_results": 2}' \
| python3 -m json.tool

# 3. Check rag-api logs for provider errors
docker compose -f docker/docker-compose.yml logs rag-api | grep -i "web\|search\|duck\|tavily"
```

Common causes:
- **503 "Web search is disabled"**: `WEB_SEARCH_ENABLED=true` is missing from `.env` — restart rag-api after adding it.
- **DuckDuckGo returns no results**: the container may be rate-limited. Add a brief retry delay or switch to Tavily.
- **Tavily "Invalid API key"**: double-check `TAVILY_API_KEY` in `.env` (no extra spaces/quotes).
- **Timeout / connection refused**: verify the server has outbound internet access (`curl -s https://api.duckduckgo.com` from the host).

### TLS/HTTPS not working from laptop

```bash
# Test from server first
curl -k https://localhost/health

# Check nginx config
docker compose -f docker/docker-compose.yml exec nginx nginx -t

# View nginx errors
docker compose -f docker/docker-compose.yml logs nginx | grep error
```

---

## Advanced: Switching to vLLM

vLLM provides higher throughput (useful if multiple people use the assistant simultaneously). V100 is supported (compute capability 7.0).

```bash
# Edit config/.env
LLM_BACKEND=vllm
LLM_MODEL_HF=Qwen/Qwen2.5-Coder-7B-Instruct  # HuggingFace model ID (not GGUF)
LLM_QUANTIZATION=awq                           # use AWQ 4-bit quantization

# Rebuild the LLM server container (vLLM is installed there)
docker compose -f docker/docker-compose.yml build llm-server
docker compose -f docker/docker-compose.yml up -d llm-server
```

**vLLM requirements on V100:**
- Use `--dtype float16` (V100 does not support bfloat16)
- Use AWQ or GPTQ quantization for 7B+ models
- Set `--max-model-len 8192` if running into memory issues

**Recommended vLLM models for V100 16GB:**
- `TheBloke/deepseek-coder-6.7B-instruct-AWQ` (~4 GB, fast)
- `TheBloke/CodeLlama-13B-Instruct-AWQ` (~7 GB, higher quality)
