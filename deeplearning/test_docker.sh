#!/usr/bin/bash
###############################################################################
# Deep Learning Docker — Comprehensive GPU Capability Test
#
# Runs inside the deeplearning:v100-llm container and validates:
#   1. NVIDIA driver / nvidia-smi access
#   2. PyTorch CUDA — detection, compute, memory, mixed precision
#   3. TensorFlow GPU — detection and compute
#   4. Hugging Face stack — transformers, accelerate, peft, datasets
#   5. Quantization libs — bitsandbytes, auto-gptq, autoawq
#   6. xformers memory-efficient attention
#   7. LLM / RAG libs — langchain, tiktoken, faiss, chromadb
#   8. Data science libs — sklearn, pandas, matplotlib, scipy, opencv
#   9. Jupyter availability
###############################################################################

set -e

IMAGE="deeplearning:v100-llm"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

check() {
    local label="$1"; shift
    if output=$(docker run --rm --gpus all "$IMAGE" bash -c "$*" 2>&1); then
        echo -e "  ${GREEN}PASS${NC}  $label"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC}  $label"
        echo "$output" | head -5 | sed 's/^/         /'
        FAIL=$((FAIL + 1))
    fi
}

echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}  Deep Learning Docker — GPU Capability Test${NC}"
echo -e "${YELLOW}============================================================${NC}"
echo ""

# ── 1. NVIDIA Driver ────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/9] NVIDIA Driver${NC}"
check "nvidia-smi accessible" "nvidia-smi > /dev/null"
check "GPU detected" "nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi tesla"
echo ""

# ── 2. PyTorch CUDA ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/9] PyTorch CUDA${NC}"
check "torch.cuda.is_available()" \
    "python3 -c \"import torch; assert torch.cuda.is_available()\""
check "GPU device name" \
    "python3 -c \"import torch; print(torch.cuda.get_device_name(0))\" | grep -qi v100"
check "CUDA version is 12.4" \
    "python3 -c \"import torch; assert torch.version.cuda.startswith('12.4')\""
check "Matrix multiply on GPU" \
    "python3 -c \"
import torch
a = torch.randn(2048, 2048, device='cuda')
b = torch.randn(2048, 2048, device='cuda')
c = torch.mm(a, b)
assert c.device.type == 'cuda'
\""
check "GPU memory allocation" \
    "python3 -c \"
import torch
x = torch.zeros(1024, 1024, 4, device='cuda')  # ~16 MB
assert torch.cuda.memory_allocated() > 0
del x; torch.cuda.empty_cache()
\""
check "Mixed precision (float16)" \
    "python3 -c \"
import torch
with torch.cuda.amp.autocast():
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.mm(a, a)
    assert b.dtype == torch.float16
\""
echo ""

# ── 3. TensorFlow GPU ───────────────────────────────────────────────────────
echo -e "${YELLOW}[3/9] TensorFlow GPU${NC}"
check "TF sees GPU" \
    "python3 -c \"import tensorflow as tf; assert len(tf.config.list_physical_devices('GPU')) > 0\""
check "TF GPU compute" \
    "python3 -c \"
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([1024, 1024])
    b = tf.matmul(a, a)
assert 'GPU' in b.device
\""
echo ""

# ── 4. Hugging Face Stack ───────────────────────────────────────────────────
echo -e "${YELLOW}[4/9] Hugging Face Stack${NC}"
for lib in transformers datasets accelerate peft trl evaluate tokenizers sentencepiece safetensors huggingface_hub; do
    check "$lib" "python3 -c \"import $lib\""
done
echo ""

# ── 5. Quantization ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[5/9] Quantization Libraries${NC}"
check "bitsandbytes" "python3 -c \"import bitsandbytes\""
check "bitsandbytes CUDA" \
    "python3 -c \"import bitsandbytes; import torch; x = torch.zeros(64, device='cuda', dtype=torch.int8); assert x.device.type == 'cuda'\""
check "auto-gptq installed" "pip3 show auto-gptq > /dev/null"
check "autoawq" "python3 -c \"import awq\""
check "optimum" "python3 -c \"import optimum\""
echo ""

# ── 6. xformers ─────────────────────────────────────────────────────────────
echo -e "${YELLOW}[6/9] xformers${NC}"
check "xformers import" "python3 -c \"import xformers\""
check "xformers memory-efficient attention" \
    "python3 -c \"
import torch, xformers.ops as xops
q = torch.randn(1, 8, 128, 64, device='cuda')
k = torch.randn(1, 8, 128, 64, device='cuda')
v = torch.randn(1, 8, 128, 64, device='cuda')
out = xops.memory_efficient_attention(q, k, v)
assert out.shape == q.shape
\""
echo ""

# ── 7. LLM / RAG ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[7/9] LLM / RAG Libraries${NC}"
for lib in langchain langchain_community langchain_core openai tiktoken chromadb; do
    check "$lib" "python3 -c \"import $lib\""
done
check "faiss-cpu" "python3 -c \"import faiss; idx = faiss.IndexFlatL2(64); assert idx.d == 64\""
check "llama-cpp-python" "python3 -c \"from llama_cpp import Llama\""
echo ""

# ── 8. Data Science ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[8/9] Data Science Libraries${NC}"
for lib in sklearn pandas matplotlib seaborn plotly scipy sympy cv2; do
    check "$lib" "python3 -c \"import $lib\""
done
check "onnxruntime-gpu" "python3 -c \"import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()\""
echo ""

# ── 9. Jupyter & Serving ────────────────────────────────────────────────────
echo -e "${YELLOW}[9/9] Jupyter & Serving${NC}"
check "jupyter lab" "python3 -c \"import jupyterlab\""
check "gradio" "python3 -c \"import gradio\""
check "wandb" "python3 -c \"import wandb\""
check "tensorboard" "python3 -c \"import tensorboard\""
check "mlflow" "python3 -c \"import mlflow\""
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "${YELLOW}============================================================${NC}"
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}All $TOTAL tests passed${NC}"
else
    echo -e "  ${GREEN}$PASS passed${NC}  /  ${RED}$FAIL failed${NC}  (out of $TOTAL)"
fi
echo -e "${YELLOW}============================================================${NC}"

exit "$FAIL"
