#!/usr/bin/env bash
###############################################################################
# 02_download_model.sh
# Download the recommended GGUF model for the coding assistant.
# Default: Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.4 GB)
###############################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── Model options (uncomment the one you want) ────────────────────────────────
REPO_ID="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
MODEL_FILE="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"

# Alternative: DeepSeek-Coder-6.7B (also excellent for code)
# REPO_ID="TheBloke/deepseek-coder-6.7B-instruct-GGUF"
# MODEL_FILE="deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

# Alternative: CodeLlama-13B (bigger, more explanation capability)
# REPO_ID="TheBloke/CodeLlama-13B-Instruct-GGUF"
# MODEL_FILE="codellama-13b-instruct.Q4_K_M.gguf"
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

TARGET="$MODELS_DIR/$MODEL_FILE"

echo -e "${YELLOW}=== Model Download ===${NC}"
echo "  Repo     : $REPO_ID"
echo "  File     : $MODEL_FILE"
echo "  Save to  : $TARGET"
echo ""

if [ -f "$TARGET" ]; then
    SIZE=$(du -sh "$TARGET" | cut -f1)
    echo -e "${GREEN}Model already exists ($SIZE). Skipping download.${NC}"
    echo "  Delete $TARGET to re-download."
else
    echo -e "${YELLOW}Downloading... (this may take several minutes)${NC}"

    # Try huggingface_hub Python client first (respects HF_HUB_VERBOSITY and proxy)
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

# Support Intel proxy from environment
proxies = {}
if os.getenv("https_proxy") or os.getenv("HTTPS_PROXY"):
    p = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    proxies = {"https": p, "http": p}

path = hf_hub_download(
    repo_id="$REPO_ID",
    filename="$MODEL_FILE",
    local_dir="$MODELS_DIR",
    force_download=False,
)
print(f"Downloaded to: {path}")
PYEOF
    else
        # Fallback: wget
        echo "huggingface_hub not found — using wget"
        wget --progress=bar:force \
            "https://huggingface.co/${REPO_ID}/resolve/main/${MODEL_FILE}" \
            -O "$TARGET"
    fi
fi

echo ""
echo -e "${GREEN}Model ready: $TARGET${NC}"
echo ""

# ── Update .env with model filename ──────────────────────────────────────────
ENV_FILE="config/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -q "^LLM_MODEL_FILE=" "$ENV_FILE"; then
        sed -i "s|^LLM_MODEL_FILE=.*|LLM_MODEL_FILE=${MODEL_FILE}|" "$ENV_FILE"
    else
        echo "LLM_MODEL_FILE=${MODEL_FILE}" >> "$ENV_FILE"
    fi
    echo "Updated config/.env: LLM_MODEL_FILE=${MODEL_FILE}"
else
    echo -e "${YELLOW}config/.env not found — run setup/03_generate_secrets.sh first${NC}"
fi

echo ""
echo "Next step: bash setup/03_generate_secrets.sh"
