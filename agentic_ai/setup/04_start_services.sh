#!/usr/bin/env bash
###############################################################################
# 04_start_services.sh
# Build and start all stack services, then wait for them to become healthy.
###############################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

COMPOSE="docker compose -f docker/docker-compose.yml --env-file config/.env"

echo -e "${YELLOW}=== Starting Coding Assistant Stack ===${NC}"
echo ""

# ── Preflight ────────────────────────────────────────────────────────────────
if [ ! -f "config/.env" ]; then
    echo -e "${RED}config/.env not found — run setup/03_generate_secrets.sh first${NC}"
    exit 1
fi

if [ ! -f "docker/nginx/ssl/server.crt" ]; then
    echo -e "${RED}TLS cert not found — run setup/03_generate_secrets.sh first${NC}"
    exit 1
fi

source config/.env
MODEL_PATH="models/${LLM_MODEL_FILE:-}"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Model file not found: $MODEL_PATH${NC}"
    echo "  Run setup/02_download_model.sh first"
    exit 1
fi

# ── Build & start ─────────────────────────────────────────────────────────────
echo -e "${YELLOW}Building images (first run takes ~5 minutes)...${NC}"
$COMPOSE build --parallel

echo ""
echo -e "${YELLOW}Starting services...${NC}"
$COMPOSE up -d

# ── Wait for health checks ────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Waiting for services to become healthy...${NC}"

SERVICES=("chromadb" "llm-server" "rag-api" "open-webui" "nginx")
TIMEOUT=300   # 5 minutes total
INTERVAL=5
ELAPSED=0

while true; do
    ALL_HEALTHY=true
    STATUS_LINE=""

    for svc in "${SERVICES[@]}"; do
        STATE=$(docker inspect --format='{{.State.Health.Status}}' "$svc" 2>/dev/null || echo "no-healthcheck")
        case "$STATE" in
            healthy)      STATUS_LINE+="  ${GREEN}✓${NC} $svc  " ;;
            no-healthcheck) STATUS_LINE+="  ${GREEN}✓${NC} $svc  " ;;
            starting)     STATUS_LINE+="  ${YELLOW}…${NC} $svc  "; ALL_HEALTHY=false ;;
            *)            STATUS_LINE+="  ${RED}✗${NC} $svc  "; ALL_HEALTHY=false ;;
        esac
    done

    echo -ne "\r$STATUS_LINE"

    if $ALL_HEALTHY; then
        echo ""
        break
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo ""
        echo -e "${RED}Timeout waiting for services. Check logs:${NC}"
        echo "  $COMPOSE logs --tail=50"
        exit 1
    fi

    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

# ── Status summary ────────────────────────────────────────────────────────────
echo ""
$COMPOSE ps
echo ""

# ── Verify LLM server ────────────────────────────────────────────────────────
echo -n "LLM server health: "
if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}not yet ready (may still be loading the model)${NC}"
fi

# ── Print access information ─────────────────────────────────────────────────
SERVER_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || hostname -I | awk '{print $1}')
HTTPS_PORT=$(grep "^NGINX_HTTPS_PORT=" config/.env 2>/dev/null | cut -d= -f2 || echo "443")

echo ""
echo -e "${GREEN}=== Stack is running ===${NC}"
echo ""
echo "  Browser interface : https://${SERVER_IP}:${HTTPS_PORT}"
echo "  (Accept the self-signed cert warning on first visit)"
echo ""
echo "  RAG API key       : $(grep RAG_API_KEY config/.env | cut -d= -f2)"
echo ""
echo "  GPU usage         : watch -n 2 nvidia-smi"
echo "  Service logs      : $COMPOSE logs -f"
echo "  Stop services     : $COMPOSE down"
echo ""
echo "Next step: bash setup/05_ingest_code.sh /home/ralfahad/projects/ai_llm"
