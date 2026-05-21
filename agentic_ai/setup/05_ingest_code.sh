#!/usr/bin/env bash
###############################################################################
# 05_ingest_code.sh
# Index a code directory into ChromaDB so the assistant can answer
# questions grounded in your actual code.
#
# Usage:
#   bash setup/05_ingest_code.sh /path/to/code [collection_name]
#
# Examples:
#   bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm
#   bash setup/05_ingest_code.sh /home/ralfahad/docs enterprise
###############################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

SOURCE_DIR="${1:-}"
COLLECTION="${2:-codebase}"
COMPOSE="docker compose -f docker/docker-compose.yml --env-file config/.env"

if [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 /path/to/source/code [collection]"
    echo "  collection: 'codebase' (default) or 'enterprise'"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

if [ ! -f "config/.env" ]; then
    echo -e "${RED}config/.env not found — run setup/03_generate_secrets.sh first${NC}"
    exit 1
fi

source config/.env

echo -e "${YELLOW}=== Ingesting into collection: ${COLLECTION} ===${NC}"
echo "  Source    : $SOURCE_DIR"
echo "  Collection: $COLLECTION"
echo ""

# ── Copy source into rag-api container's data volume ─────────────────────────
# We mount the source directory into the running container temporarily
echo -e "${YELLOW}Running ingestion inside rag-api container...${NC}"

docker run --rm \
    --network coding-assistant_backend \
    -e CHROMADB_HOST=chromadb \
    -e CHROMADB_PORT=8000 \
    -e CHROMA_TOKEN="${CHROMA_TOKEN}" \
    -e EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}" \
    -e EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}" \
    -v "${SOURCE_DIR}:/ingest_source:ro" \
    -v "$(pwd)/src/scripts:/app/scripts:ro" \
    coding-assistant-api:latest \
    python3 /app/scripts/ingest.py \
        --source /ingest_source \
        --collection "$COLLECTION" \
        --chromadb-host chromadb \
        --chromadb-port 8000 \
        --chroma-token "${CHROMA_TOKEN}" \
        --embedding-model "${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}" \
        --embedding-device "${EMBEDDING_DEVICE:-cpu}" \
        --chunk-size 1000 \
        --chunk-overlap 200

echo ""
echo -e "${GREEN}=== Ingestion complete ===${NC}"
echo ""

# ── Verify with a test query ──────────────────────────────────────────────────
echo -e "${YELLOW}Verifying RAG with a test query...${NC}"
RESULT=$(curl -sf \
    -X POST http://localhost:8080/v1/rag/search \
    -H "Authorization: Bearer ${RAG_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"pytorch training loop\", \"k\": 2}" 2>/dev/null || echo '{"results":[]}')

COUNT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('results',[])))" 2>/dev/null || echo "0")

if [ "$COUNT" -gt 0 ]; then
    echo -e "${GREEN}RAG search working — retrieved $COUNT chunks for test query${NC}"
else
    echo -e "${YELLOW}Test query returned 0 results. This may be normal if the code doesn't contain 'pytorch training loop'.${NC}"
    echo "  Check stats: curl -s -H 'Authorization: Bearer ${RAG_API_KEY}' http://localhost:8080/v1/rag/stats"
fi
