#!/usr/bin/env bash
###############################################################################
# Generate a self-signed TLS certificate for the Nginx proxy
# Valid for 10 years — suitable for private / internal server use
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSL_DIR="$SCRIPT_DIR/ssl"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

mkdir -p "$SSL_DIR"

# ── Determine server IP / hostname ──────────────────────────────────────────
SERVER_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || hostname -I | awk '{print $1}')
HOSTNAME=$(hostname -f 2>/dev/null || hostname)

echo -e "${YELLOW}Generating self-signed TLS certificate...${NC}"
echo -e "  Server IP   : ${SERVER_IP}"
echo -e "  Hostname    : ${HOSTNAME}"

# ── OpenSSL config with Subject Alternative Names ───────────────────────────
cat > "$SSL_DIR/openssl.cnf" << EOF
[req]
default_bits       = 4096
prompt             = no
default_md         = sha256
distinguished_name = dn
x509_extensions    = v3_req

[dn]
C  = US
ST = Private
L  = Server
O  = CodingAssistant
OU = SelfSigned
CN = ${HOSTNAME}

[v3_req]
subjectAltName = @alt_names
keyUsage       = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = localhost
DNS.2 = ${HOSTNAME}
IP.1  = 127.0.0.1
IP.2  = ${SERVER_IP}
EOF

# ── Generate private key and certificate ────────────────────────────────────
openssl req -x509 -newkey rsa:4096 -sha256 \
    -days 3650 \
    -nodes \
    -keyout "$SSL_DIR/server.key" \
    -out   "$SSL_DIR/server.crt" \
    -config "$SSL_DIR/openssl.cnf" 2>/dev/null

# Secure the private key
chmod 600 "$SSL_DIR/server.key"
chmod 644 "$SSL_DIR/server.crt"

echo -e "${GREEN}Certificate generated:${NC}"
echo -e "  ${SSL_DIR}/server.crt"
echo -e "  ${SSL_DIR}/server.key"
echo ""
openssl x509 -text -noout -in "$SSL_DIR/server.crt" \
    | grep -E "Subject:|Not (Before|After)|IP Address|DNS:"
echo ""
echo -e "${YELLOW}Note:${NC} Browsers will show a security warning for self-signed certs."
echo -e "      Click 'Advanced' → 'Proceed' to accept it on first visit."
echo -e "      For a trusted cert, use Let's Encrypt (see TUTORIAL.md Step 10)."
