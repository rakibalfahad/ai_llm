#!/usr/bin/env bash
###############################################################################
# 01_check_prerequisites.sh
# Verify all dependencies are in place before starting the stack.
###############################################################################
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

check() {
    local label="$1"; shift
    if eval "$@" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC}  $label"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC}  $label"
        FAIL=$((FAIL + 1))
    fi
}

echo -e "${YELLOW}=== Coding Assistant — Prerequisites Check ===${NC}"
echo ""

echo "Docker:"
check "Docker daemon running"         "docker info"
check "Docker Compose available"      "docker compose version"
check "nvidia-container-toolkit"      "docker run --rm --gpus all --entrypoint nvidia-smi nvidia/cuda:12.4.1-base-ubuntu22.04 -L"

echo ""
echo "GPU:"
check "nvidia-smi available"          "command -v nvidia-smi"
check "V100 GPU detected"             "nvidia-smi --query-gpu=name --format=csv,noheader | grep -i V100"
check "16 GB VRAM"                    "nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{print \$1}' | awk '\$1 >= 15000'"

echo ""
echo "Docker image:"
check "deeplearning:v100-llm exists"  "docker image inspect deeplearning:v100-llm"

echo ""
echo "Disk space:"
check "≥ 20 GB free in /home"         "[ \$(df /home --output=avail | tail -1) -ge 20971520 ]"

echo ""
echo "Ports:"
check "Port 80 available"             "! ss -tlnp | grep -q ':80 '"
check "Port 443 available"            "! ss -tlnp | grep -q ':443 '"

echo ""
echo "Tools:"
check "openssl available"             "command -v openssl"
check "curl available"                "command -v curl"
check "python3 available"             "command -v python3"

echo ""
echo "────────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All checks passed ($PASS/$((PASS+FAIL)))${NC}"
    echo "Ready to run: bash setup/02_download_model.sh"
else
    echo -e "${RED}$FAIL check(s) failed${NC} (${PASS} passed)"
    echo ""
    echo "Common fixes:"
    echo "  nvidia-container-toolkit missing:"
    echo "    apt install -y nvidia-container-toolkit && systemctl restart docker"
    echo "  deeplearning:v100-llm missing:"
    echo "    cd ../deeplearning && bash create_docker.sh"
    echo "  Port conflict:"
    echo "    Set NGINX_HTTP_PORT and NGINX_HTTPS_PORT in config/.env"
    exit 1
fi
