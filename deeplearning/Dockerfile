###############################################################################
# Deep Learning & LLM Image
# CUDA 12.4 on Ubuntu 22.04 — Tesla V100 (sm_70)
###############################################################################
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# ── GPU ──────────────────────────────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="7.0"

# ── Network proxy (Intel) ───────────────────────────────────────────────────
ENV http_proxy=http://proxy-us.intel.com:912 \
    https_proxy=http://proxy-us.intel.com:912 \
    no_proxy=localhost,127.0.0.1

# ── System packages ─────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential cmake ninja-build \
        curl wget git git-lfs vim \
        libssl-dev libffi-dev libsndfile1 ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 default ─────────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

# ── pip proxy ────────────────────────────────────────────────────────────────
RUN mkdir -p /root/.pip && printf '\
[global]\n\
proxy = http://proxy-us.intel.com:912\n\
trusted-host = pypi.org pypi.python.org files.pythonhosted.org download.pytorch.org\n' \
    > /root/.pip/pip.conf

# ── PyTorch 2.6 + CUDA 12.4 ─────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    "torch==2.6.*" "torchvision==0.21.*" "torchaudio==2.6.*" \
    --index-url https://download.pytorch.org/whl/cu124

# ── TensorFlow (bundles its own CUDA libs) ───────────────────────────────────
RUN pip3 install --no-cache-dir "tensorflow[and-cuda]"

# ── Hugging Face ecosystem ───────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    transformers datasets accelerate peft trl \
    evaluate tokenizers sentencepiece safetensors huggingface_hub

# ── Quantization (essential for V100 16 GB) ──────────────────────────────────
RUN pip3 install --no-cache-dir \
    bitsandbytes optimum auto-gptq autoawq

# ── xformers (memory-efficient attention, sm_70) ─────────────────────────────
RUN pip3 install --no-cache-dir \
    "xformers>=0.0.29,<0.0.30" \
    --index-url https://download.pytorch.org/whl/cu124

# ── LLM / RAG ───────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    langchain langchain-community langchain-core \
    openai tiktoken faiss-cpu chromadb
RUN pip3 install --no-cache-dir llama-cpp-python

# ── Data Science & ML ────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    scikit-learn pandas matplotlib seaborn plotly \
    scipy sympy pillow opencv-python-headless

# ── ONNX ─────────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir onnx onnxruntime-gpu

# ── Experiment tracking ──────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir wandb tensorboard mlflow

# ── Jupyter ──────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir jupyter jupyterlab ipywidgets nbformat

# ── Utilities ────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir einops gradio tqdm rich protobuf

# ── Re-pin PyTorch (guard against dependency overwrites) ─────────────────────
RUN pip3 install --no-cache-dir \
    "torch==2.6.*" "torchvision==0.21.*" "torchaudio==2.6.*" \
    --index-url https://download.pytorch.org/whl/cu124

# ── Fix cuDNN: TF 2.21 needs >=9.3, PyTorch cu124 ships 9.1 ─────────────────
# Pin to 9.3.x — newer versions (9.20+) drop V100 sm_70 conv kernels
RUN pip3 install --no-cache-dir "nvidia-cudnn-cu12>=9.3,<9.4"

# ── Test script ──────────────────────────────────────────────────────────────
COPY <<'TESTSCRIPT' /test_gpu.py
import sys, importlib

print("=" * 60)

# PyTorch
import torch
print(f"PyTorch        : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  GPU count      : {torch.cuda.device_count()}")
    print(f"  CUDA version   : {torch.version.cuda}")
    x = torch.randn(1024, 1024, device="cuda")
    _ = torch.mm(x, x)
    print(f"  Compute test   : OK  (1024x1024 matmul)")

print("-" * 60)

# TensorFlow
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow     : {tf.__version__}")
print(f"  GPUs found     : {len(gpus)}")
if gpus:
    print(f"  GPU            : {gpus[0]}")

print("-" * 60)

# Key libraries
for lib in ["transformers", "accelerate", "peft", "bitsandbytes",
            "datasets", "langchain", "xformers"]:
    try:
        mod = importlib.import_module(lib)
        ver = getattr(mod, "__version__", "installed")
        print(f"{lib:15s}: {ver}")
    except ImportError:
        print(f"{lib:15s}: NOT FOUND")

print("=" * 60)
TESTSCRIPT

WORKDIR /workspace
EXPOSE 8888 7860
CMD ["bash"]
