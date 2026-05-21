#!/usr/bin/env bash
###############################################################################
# 03_generate_secrets.sh
# Create config/.env with secure random keys and a self-signed TLS cert.
# Safe to re-run — will NOT overwrite existing secrets.
###############################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

ENV_FILE="config/.env"
mkdir -p config

echo -e "${YELLOW}=== Generating Secrets ===${NC}"

# ── Generate secrets (only if .env doesn't exist) ────────────────────────────
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}config/.env already exists — skipping key generation${NC}"
    echo "  Delete config/.env to regenerate all secrets."
else
    echo "Generating cryptographic keys..."

    # Generate the three secrets
    RAG_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    CHROMA_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    WEBUI_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    REDIS_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

    # Write .env from template
    cp config/.env.example "$ENV_FILE"
    sed -i "s|RAG_API_KEY=.*|RAG_API_KEY=${RAG_API_KEY}|"                 "$ENV_FILE"
    sed -i "s|CHROMA_TOKEN=.*|CHROMA_TOKEN=${CHROMA_TOKEN}|"               "$ENV_FILE"
    sed -i "s|WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY}|"   "$ENV_FILE"
    sed -i "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=${REDIS_PASSWORD}|"         "$ENV_FILE"

    # Secure the file
    chmod 600 "$ENV_FILE"

    echo -e "${GREEN}  config/.env created (chmod 600)${NC}"
    echo -e "  ${YELLOW}Keep this file private — never commit it to version control${NC}"
fi

# ── Generate TLS certificate ─────────────────────────────────────────────────
echo ""
echo "TLS certificate:"
if [ -f "docker/nginx/ssl/server.crt" ]; then
    echo -e "${YELLOW}  SSL certificate already exists — skipping${NC}"
    echo "  Delete docker/nginx/ssl/ to regenerate."
else
    bash docker/nginx/generate_ssl.sh
fi

# ── Ensure .gitignore protects secrets ───────────────────────────────────────
GITIGNORE_ROOT="../.gitignore"
if [ -f "$GITIGNORE_ROOT" ]; then
    for entry in "agentic_ai/config/.env" "agentic_ai/docker/nginx/ssl/" "agentic_ai/models/"; do
        if ! grep -qF "$entry" "$GITIGNORE_ROOT"; then
            echo "$entry" >> "$GITIGNORE_ROOT"
        fi
    done
fi

# Create a local .gitignore as well
cat > .gitignore << 'EOF'
config/.env
docker/nginx/ssl/
models/
__pycache__/
*.pyc
*.pyo
.DS_Store
EOF

echo ""
echo -e "${GREEN}=== Secrets & TLS ready ===${NC}"
echo ""
echo "Summary:"
echo "  config/.env           — API keys and settings"
echo "  docker/nginx/ssl/     — TLS certificate (self-signed)"
echo "  .gitignore            — protects secrets from git"
echo ""
echo "Next step: bash setup/04_start_services.sh"
