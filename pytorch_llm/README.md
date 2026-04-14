# PyTorch LLM — Build Large Language Models from Scratch

A hands-on tutorial series for advanced Python developers with deep learning knowledge.
You will build every component of a modern LLM (LLaMA-style) from scratch in PyTorch,
then fine-tune a real open-weight LLaMA model on your own data.

---

## Prerequisites

- Python proficiency (decorators, generators, dataclasses)
- Deep learning fundamentals (backprop, loss, gradient descent)
- Basic PyTorch (tensors, `nn.Module`, `DataLoader`)
- Access to the `deeplearning:v100-llm` Docker image

---

## How to Run Any Tutorial

### Start an interactive container (recommended for exploration)

```bash
cd /home/ralfahad/projects/ai_llm/deeplearning

docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \
  -it deeplearning:v100-llm
```

Inside the container:
```bash
python3 /workspace/pytorch_llm/tutorials/01_tokenizer_bpe.py
python3 /workspace/pytorch_llm/tutorials/02_attention_mechanism.py
# ... etc
```

### Run a single tutorial directly

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \
  deeplearning:v100-llm \
  python3 /workspace/pytorch_llm/tutorials/01_tokenizer_bpe.py
```

### Run with TensorBoard

```bash
docker run --rm --gpus all \
  -p 7777:7777 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \
  deeplearning:v100-llm \
  bash -c "python3 /workspace/pytorch_llm/tutorials/04_gpt_mini.py & \
           tensorboard --logdir /workspace/data/llm_runs --host 0.0.0.0 --port 7777"
```

Open: `http://<your-host-ip>:7777`

---

## Learning Path

```
00_pytorch_refresher.py
    ↓  Tensors · autograd · nn.Module · training loop · GPU patterns
01_tokenizer_bpe.py
    ↓  How text becomes numbers
02_attention_mechanism.py
    ↓  The core of every transformer
03_transformer_block.py
    ↓  RMSNorm · RoPE · SwiGLU (exactly what LLaMA uses)
04_gpt_mini.py
    ↓  Your first LLM — train & generate text
05_llama_architecture.py
    ↓  Full LLaMA-2 / LLaMA-3 architecture with KV cache
06_finetune_llama_lora.py
    ↓  Fine-tune a real LLaMA model with QLoRA on custom data
```

---

## Tutorial Overview

### 00 — PyTorch Refresher
**Concepts:** tensors, autograd, `nn.Module`, training loop, `nn.Embedding`, `Dataset`/`DataLoader`, GPU transfer, mixed precision, checkpointing, debug hooks  
**What you build:** custom linear layer, MLP, sliding-window text dataset, full mixed-precision training step  
**Key insight:** review all the PyTorch primitives you'll use constantly in every subsequent tutorial — shapes, grads, devices, and saving/loading  

### 01 — Tokenizer & BPE
**Concepts:** tokenization, vocabulary, byte-pair encoding, encode/decode  
**What you build:** character tokenizer → BPE tokenizer from scratch → compare with HuggingFace  
**Key insight:** LLMs don't see words, they see token IDs

### 02 — Attention Mechanism
**Concepts:** Q/K/V, scaled dot-product attention, masking, multi-head attention  
**What you build:** single-head → multi-head → causal masked attention, all from `torch` ops  
**Key insight:** attention is just a learned weighted average; masking makes it causal

### 03 — Transformer Block (LLaMA-style)
**Concepts:** RMSNorm, Rotary Positional Encoding (RoPE), SwiGLU FFN  
**What you build:** each component step by step, then assemble into a full block  
**Key insight:** LLaMA replaced LayerNorm → RMSNorm, absolute PE → RoPE, GELU → SwiGLU

### 04 — Mini-GPT: Train a Language Model
**Concepts:** autoregressive training, cross-entropy loss, perplexity, text generation, sampling strategies  
**What you build:** complete ~10M-param LLM, train on Shakespeare, generate text  
**Key insight:** next-token prediction on unlabeled text is self-supervised pre-training

### 05 — LLaMA Architecture from Scratch
**Concepts:** Grouped Query Attention (GQA), KV cache, model parallelism hints, memory math  
**What you build:** production-quality LLaMA-2/3-style model class, analyzable and extensible  
**Key insight:** KV cache is why inference is fast; GQA is why large models fit in memory

### 06 — Fine-tune LLaMA with QLoRA
**Concepts:** LoRA, QLoRA (4-bit), PEFT, instruction tuning, chat templates, adapter merging  
**What you build:** fine-tuning pipeline for TinyLlama-1.1B on a custom instruction dataset  
**Key insight:** you don't need to train from scratch; 4-bit + LoRA fits a 7B model on 16 GB

---

## Directory Structure

```
pytorch_llm/
├── README.md                    ← you are here
└── tutorials/
    ├── 01_tokenizer_bpe.py      ← Tokenization & BPE from scratch
    ├── 02_attention_mechanism.py ← Self-attention & multi-head attention
    ├── 03_transformer_block.py  ← RMSNorm, RoPE, SwiGLU, full block
    ├── 04_gpt_mini.py           ← Complete LLM, train on Shakespeare
    ├── 05_llama_architecture.py ← LLaMA-2/3 style model + KV cache
    └── 06_finetune_llama_lora.py ← QLoRA fine-tuning on custom data
```

Data outputs are written to the mounted `/workspace/data/` and persist on your host under `deeplearning/data/`.

---

## Key LLM Concepts at a Glance

| Concept | Tutorial | One-line explanation |
|---------|----------|----------------------|
| Tokenization | 01 | Split text into integer IDs using a learned vocabulary |
| Self-attention | 02 | Each token attends to all others; score = Q·Kᵀ / √d |
| Causal mask | 02 | Future tokens are masked so generation is left-to-right |
| Multi-head attention | 02 | Run attention in parallel in `h` subspaces, concat results |
| RMSNorm | 03 | Faster norm: no mean subtraction, just RMS scaling |
| RoPE | 03 | Encode position by rotating Q/K vectors — generalises to long contexts |
| SwiGLU | 03 | FFN activation: `Swish(xW) ⊗ (xV)` — better than ReLU/GELU |
| Autoregressive | 04 | Generate one token at a time, feed it back as next input |
| Perplexity | 04 | `exp(cross_entropy_loss)` — lower is better |
| KV cache | 05 | Cache past K/V during inference so we don't recompute them |
| GQA | 05 | Share K/V heads across Q heads — saves memory at large scale |
| LoRA | 06 | Freeze base weights, add low-rank trainable matrices ΔW = AB |
| QLoRA | 06 | LoRA on top of 4-bit quantized base model |

---

## Recommended Reading (alongside tutorials)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original transformer paper
- [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) — architecture details
- [LoRA paper](https://arxiv.org/abs/2106.09685) — low-rank adaptation
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — 4-bit fine-tuning
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT reference
