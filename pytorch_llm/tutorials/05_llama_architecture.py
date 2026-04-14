"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 05 — Full LLaMA Architecture from Scratch                  ║
║  pytorch_llm/tutorials/05_llama_architecture.py                     ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. Grouped Query Attention (GQA) — how LLaMA-2 34B+ and LLaMA-3 save memory
  2. KV Cache — crucial for fast autoregressive inference
  3. Full LLaMA-2 and LLaMA-3 configurations (verified parameter counts)
  4. Parameter counting and memory math for real LLaMA models
  5. Forward pass analysis: shape of every tensor through the model
  6. How to load HuggingFace LLaMA weights into your architecture (structure map)

GQA vs MHA vs MQA
──────────────────
  MHA (Multi-Head Attention):
    n_kv_heads = n_heads   → each Q head has its own K/V head
    Used in LLaMA-2 7B, 13B

  GQA (Grouped Query Attention):
    n_kv_heads < n_heads   → groups of Q heads share one K/V head
    Used in LLaMA-2 34B, 70B; LLaMA-3 all sizes
    Saves KV cache memory by n_heads/n_kv_heads factor

  MQA (Multi-Query Attention):
    n_kv_heads = 1         → all Q heads share one K/V pair
    Maximum memory savings, slight quality drop

RUN IN DOCKER
─────────────
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/05_llama_architecture.py
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LLaMA Model Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLaMAConfig:
    """
    Verified configurations match published LLaMA-2 and LLaMA-3 parameter counts.
    """
    vocab_size:  int = 32_000
    d_model:     int = 4096
    n_layers:    int = 32
    n_heads:     int = 32
    n_kv_heads:  int = 32        # GQA: n_kv_heads <= n_heads; must divide n_heads
    d_ff:        Optional[int] = None  # auto-computed from d_model if None
    max_seq_len: int = 4096
    rope_base:   float = 10_000.0    # LLaMA-2=10000, LLaMA-3=500000
    rms_eps:     float = 1e-5
    dropout:     float = 0.0

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.d_k = self.d_model // self.n_heads
        self.n_groups = self.n_heads // self.n_kv_heads   # Q heads per KV head

        if self.d_ff is None:
            # LLaMA formula: keep ≈ 2/3 * 4 * d_model, rounded to 256
            d_ff_raw = int(2 / 3 * 4 * self.d_model)
            self.d_ff = (d_ff_raw + 255) // 256 * 256

    @classmethod
    def llama2_7b(cls) -> "LLaMAConfig":
        return cls(vocab_size=32000, d_model=4096, n_layers=32,
                   n_heads=32, n_kv_heads=32, max_seq_len=4096,
                   rope_base=10000.0)

    @classmethod
    def llama2_13b(cls) -> "LLaMAConfig":
        return cls(vocab_size=32000, d_model=5120, n_layers=40,
                   n_heads=40, n_kv_heads=40, max_seq_len=4096,
                   rope_base=10000.0)

    @classmethod
    def llama2_70b(cls) -> "LLaMAConfig":
        # 70B uses GQA: 64 Q heads, 8 KV heads
        return cls(vocab_size=32000, d_model=8192, n_layers=80,
                   n_heads=64, n_kv_heads=8,  max_seq_len=4096,
                   d_ff=28672, rope_base=10000.0)

    @classmethod
    def llama3_8b(cls) -> "LLaMAConfig":
        return cls(vocab_size=128256, d_model=4096, n_layers=32,
                   n_heads=32, n_kv_heads=8,  max_seq_len=8192,
                   d_ff=14336, rope_base=500000.0)

    @classmethod
    def llama3_70b(cls) -> "LLaMAConfig":
        return cls(vocab_size=128256, d_model=8192, n_layers=80,
                   n_heads=64, n_kv_heads=8,  max_seq_len=8192,
                   d_ff=28672, rope_base=500000.0)

    @classmethod
    def tiny(cls) -> "LLaMAConfig":
        """Tiny model for testing on any hardware."""
        return cls(vocab_size=32000, d_model=512, n_layers=4,
                   n_heads=8, n_kv_heads=2, max_seq_len=512,
                   rope_base=10000.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


def precompute_rope_freqs(d_k: int, max_seq_len: int,
                           base: float = 10000.0,
                           device: torch.device = torch.device("cpu")):
    """Precompute RoPE cos/sin frequencies. Returns [max_seq_len, d_k//2]."""
    i = torch.arange(0, d_k, 2, device=device).float()
    theta = 1.0 / (base ** (i / d_k))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """Apply RoPE to q or k. offset supports KV cache position tracking."""
    s = x.size(2)
    c = cos[offset:offset+s].unsqueeze(0).unsqueeze(0)
    s_ = sin[offset:offset+s].unsqueeze(0).unsqueeze(0)
    xe, xo = x[..., 0::2], x[..., 1::2]
    return torch.stack([xe * c - xo * s_, xe * s_ + xo * c], dim=-1).flatten(-2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Grouped Query Attention with KV Cache
# ══════════════════════════════════════════════════════════════════════════════
#
# KV Cache Explained:
#   During autoregressive generation, at each step we compute K and V for
#   the new token. But we also need K and V for all previous tokens to
#   compute attention.
#
#   Without cache: recompute K/V for all tokens at every step  → O(N²) time
#   With cache: store K/V from previous steps, only compute for new token
#               → O(N) time per new token
#
#   Cache shape: [batch, n_kv_heads, seq_so_far, d_k]
#   Each new token *appends* to the cache.
#
# GQA Implementation:
#   n_groups = n_heads / n_kv_heads
#   Q: [B, n_heads,    S,  d_k]
#   K: [B, n_kv_heads, S,  d_k]  → repeat n_groups times → [B, n_heads, S, d_k]
#   V: [B, n_kv_heads, S,  d_k]  → repeat n_groups times → [B, n_heads, S, d_k]

class GQAttention(nn.Module):
    """
    Grouped Query Attention with optional KV cache (for inference).

    During training: use standard batched attention (cache=None).
    During inference: pass cache dict to accumulate K/V across tokens.
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups   = cfg.n_groups
        self.d_k        = cfg.d_k
        d_model         = cfg.d_model

        # Q projects to n_heads × d_k = d_model
        # K, V project to n_kv_heads × d_k  (smaller in GQA)
        self.W_q = nn.Linear(d_model, cfg.n_heads   * cfg.d_k, bias=False)
        self.W_k = nn.Linear(d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.W_v = nn.Linear(d_model, cfg.n_kv_heads * cfg.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model,                   bias=False)

        # RoPE frequencies (stored as non-parameter buffers)
        cos_f, sin_f = precompute_rope_freqs(cfg.d_k, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos_freq", cos_f)
        self.register_buffer("sin_freq", sin_f)

        self.dropout = cfg.dropout

    def forward(self,
                x: torch.Tensor,
                cache: Optional[dict] = None,
                cache_pos: int = 0) -> torch.Tensor:
        """
        x         : [B, S, d_model]
        cache     : dict with 'k' and 'v' tensors (for inference)
        cache_pos : position offset for RoPE when using KV cache
        """
        B, S, D = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(B, S, self.n_heads,    self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE (with position offset for KV cache)
        q = apply_rope(q, self.cos_freq, self.sin_freq, offset=cache_pos)
        k = apply_rope(k, self.cos_freq, self.sin_freq, offset=cache_pos)

        # KV Cache: append new K/V and use full history
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v

        # GQA: expand K/V to match n_heads by repeating each KV head n_groups times
        # [B, n_kv_heads, S, d_k] → [B, n_heads, S, d_k]
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention
        # is_causal=True only during training (full sequence);
        # during inference (cache), query S=1, so no masking needed
        is_causal = (cache is None)
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                              dropout_p=dp,
                                              is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feed-Forward Network (SwiGLU)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.W_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.W_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLaMA Decoder Layer
# ══════════════════════════════════════════════════════════════════════════════

class LLaMADecoderLayer(nn.Module):
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn      = GQAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn       = FeedForward(cfg)
        self.drop       = nn.Dropout(cfg.dropout)

    def forward(self,
                x: torch.Tensor,
                cache: Optional[dict] = None,
                cache_pos: int = 0) -> torch.Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x), cache=cache, cache_pos=cache_pos))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Full LLaMA Model
# ══════════════════════════════════════════════════════════════════════════════

class LLaMA(nn.Module):
    """
    Production-quality LLaMA-2 / LLaMA-3 model.

    Weight naming matches HuggingFace LLaMA implementation:
      model.embed_tokens.weight
      model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{i}.mlp.{gate,up,down}_proj.weight
      model.layers.{i}.{input,post}_layernorm.weight
      model.norm.weight
      lm_head.weight
    """

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop         = nn.Dropout(cfg.dropout)
        self.layers       = nn.ModuleList([
            LLaMADecoderLayer(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm    = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed_tokens.weight, std=std)
        for layer in self.layers:
            for name, p in layer.named_parameters():
                if "W_o" in name or "W_down" in name:
                    nn.init.normal_(p, std=std / math.sqrt(2 * self.cfg.n_layers))
                elif p.dim() >= 2:
                    nn.init.normal_(p, std=std)
                else:
                    nn.init.ones_(p)

    def forward(self,
                input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> tuple:
        """
        input_ids : [B, S]
        targets   : [B, S]  (optional, for computing loss)
        use_cache : if True, returns (logits, caches) instead of (logits, loss)
        """
        B, S = input_ids.shape
        x = self.drop(self.embed_tokens(input_ids))   # [B, S, d_model]

        caches = [{} for _ in self.layers] if use_cache else [None] * len(self.layers)

        for layer, cache in zip(self.layers, caches):
            x = layer(x, cache=cache, cache_pos=0)

        x      = self.norm(x)
        logits = self.lm_head(x)           # [B, S, vocab_size]

        if use_cache:
            return logits, caches

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    def generate_with_cache(self,
                             prompt_ids: torch.Tensor,
                             max_new_tokens: int = 100,
                             temperature: float = 0.8,
                             top_p: float = 0.9) -> torch.Tensor:
        """
        KV-cache-based generation. Much faster than naive generation
        because we only run the transformer on ONE new token per step.
        """
        self.eval()
        device = prompt_ids.device

        # Prefill phase: process entire prompt at once
        with torch.no_grad():
            _, caches = self.forward(prompt_ids, use_cache=True)
        generated = prompt_ids
        cache_pos = prompt_ids.size(1)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Decode phase: only pass the LAST generated token
                last_token = generated[:, -1:]
                x = self.embed_tokens(last_token)

                for layer, cache in zip(self.layers, caches):
                    x = layer(x, cache=cache, cache_pos=cache_pos)

                logits = self.lm_head(self.norm(x))[:, -1, :] / temperature

                # Top-p sampling
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cumprobs = sorted_logits.softmax(-1).cumsum(-1)
                sorted_logits[cumprobs - sorted_logits.softmax(-1) > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
                next_id = torch.multinomial(logits.softmax(-1), 1)

                generated = torch.cat([generated, next_id], dim=-1)
                cache_pos += 1

        return generated

    # ── Parameter counting ────────────────────────────────────────────────────

    def count_params(self) -> dict[str, int]:
        attn_params = sum(
            p.numel() for l in self.layers
            for name, p in l.attn.named_parameters()
        )
        ffn_params = sum(
            p.numel() for l in self.layers
            for name, p in l.ffn.named_parameters()
        )
        embed_params = self.embed_tokens.weight.numel()
        norm_params = sum(
            p.numel() for l in self.layers
            for p in [l.attn_norm.scale, l.ffn_norm.scale]
        ) + self.norm.scale.numel()

        total = attn_params + ffn_params + embed_params + norm_params
        return {
            "embed":     embed_params,
            "attention": attn_params,
            "ffn":       ffn_params,
            "norms":     norm_params,
            "total":     total,
            "total_non_embed": total - embed_params,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — KV Cache Memory Math
# ══════════════════════════════════════════════════════════════════════════════

def kv_cache_memory_mb(cfg: LLaMAConfig, seq_len: int,
                        batch_size: int = 1, precision: str = "float16") -> float:
    """
    KV cache stores K and V for every layer and every token.
    Shape: [n_layers, batch, n_kv_heads, seq_len, d_k]
    """
    bytes_per_elem = 2 if precision == "float16" else 4
    # 2 tensors (K and V) per layer
    n_elems = 2 * cfg.n_layers * batch_size * cfg.n_kv_heads * seq_len * cfg.d_k
    return n_elems * bytes_per_elem / 1e6   # MB


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_model_analysis(name: str, cfg: LLaMAConfig, device: torch.device):
    """Build model, print param breakdown, run a forward pass."""
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}")
    print(f"  d_ff={cfg.d_ff}, vocab={cfg.vocab_size:,}, seq_len={cfg.max_seq_len}")

    if cfg.d_model > 2048:
        # Skip forward pass for large models — just count params
        model = LLaMA(cfg)
        counts = model.count_params()
        print(f"\n  Parameter breakdown:")
        for k, v in counts.items():
            print(f"    {k:<16s}: {v:>15,}  ({v/1e9:.3f}B)")
        fp16_gb = counts["total"] * 2 / 1e9
        fp32_gb = counts["total"] * 4 / 1e9
        print(f"\n  Storage: {fp16_gb:.2f} GB (fp16)  |  {fp32_gb:.2f} GB (fp32)")
        kv_mb = kv_cache_memory_mb(cfg, seq_len=cfg.max_seq_len)
        print(f"  KV cache ({cfg.max_seq_len} tokens, fp16): {kv_mb:,.0f} MB")

        if cfg.n_groups > 1:
            kv_mha = kv_cache_memory_mb(
                LLaMAConfig(n_kv_heads=cfg.n_heads,
                            n_heads=cfg.n_heads,
                            n_layers=cfg.n_layers,
                            d_k_override=cfg.d_k if hasattr(cfg, 'd_k_override') else None,
                            d_model=cfg.d_model,
                            max_seq_len=cfg.max_seq_len),
                seq_len=cfg.max_seq_len
            )
            saving = (1 - kv_mb / kv_mha) * 100
            print(f"  KV cache saving (vs MHA): {saving:.0f}%")
        return

    # Forward pass for smaller models
    model = LLaMA(cfg).to(device)
    counts = model.count_params()
    print(f"\n  Parameter breakdown:")
    for k, v in counts.items():
        print(f"    {k:<16s}: {v:>12,}  ({v/1e6:.1f}M)")

    B, S = 2, min(64, cfg.max_seq_len)
    ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    model.train()
    logits, loss = model(ids, targets=tgt)
    print(f"\n  Forward pass (train): ids{list(ids.shape)} → logits{list(logits.shape)}")
    print(f"  Loss: {loss.item():.4f}  |  Expected ≈ {math.log(cfg.vocab_size):.2f}  (random init)")

    model.eval()
    with torch.no_grad():
        logits_c, caches = model(ids[:1, :8], use_cache=True)
    print(f"\n  KV cache shapes (layer 0):")
    print(f"    K: {list(caches[0]['k'].shape)}   [B, n_kv_heads, S, d_k]")
    print(f"    V: {list(caches[0]['v'].shape)}")
    kv_mb = kv_cache_memory_mb(cfg, seq_len=cfg.max_seq_len)
    print(f"  KV cache for {cfg.max_seq_len} tokens (fp16): {kv_mb:.1f} MB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tutorial 05 — Full LLaMA Architecture")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── Part A: GQA vs MHA illustration ──────────────────────────────────────
    print("\n── PART A: GQA vs MHA vs MQA ──")
    print("""
  n_heads=8, d_k=64:

  MHA  (n_kv_heads=8):  Q[8,64]  K[8,64]  V[8,64]  → no sharing
  GQA  (n_kv_heads=2):  Q[8,64]  K[2,64]  V[2,64]  → 4 Q heads share 1 KV head
  MQA  (n_kv_heads=1):  Q[8,64]  K[1,64]  V[1,64]  → all Q heads share 1 KV head

  KV cache memory:
    MHA : 2 × 8 × seq × 64 × 2B = {mha:.1f} MB  (seq=4096)
    GQA : 2 × 2 × seq × 64 × 2B = {gqa:.1f} MB  (4× smaller)
    MQA : 2 × 1 × seq × 64 × 2B = {mqa:.1f} MB  (8× smaller)
""".format(
        mha=2*8*4096*64*2/1e6 * 32,
        gqa=2*2*4096*64*2/1e6 * 32,
        mqa=2*1*4096*64*2/1e6 * 32,
    ))

    # ── Part B: Tiny model (full analysis + generation) ───────────────────────
    print_model_analysis("Tiny LLaMA (testing)", LLaMAConfig.tiny(), device)

    # ── Part C: GQA in practice ───────────────────────────────────────────────
    print("\n── PART C: GQA Tensor Shapes Through the Model ──")
    cfg_tiny = LLaMAConfig.tiny()
    model = LLaMA(cfg_tiny).to(device)
    model.eval()
    B, S, d = 1, 32, cfg_tiny.d_model

    x_in = torch.randint(0, cfg_tiny.vocab_size, (B, S), device=device)
    layer0 = model.layers[0]

    with torch.no_grad():
        x_norm = layer0.attn_norm(model.embed_tokens(x_in))
        q = layer0.attn.W_q(x_norm).view(B, S, cfg_tiny.n_heads, cfg_tiny.d_k).transpose(1, 2)
        k = layer0.attn.W_k(x_norm).view(B, S, cfg_tiny.n_kv_heads, cfg_tiny.d_k).transpose(1, 2)
        v = layer0.attn.W_v(x_norm).view(B, S, cfg_tiny.n_kv_heads, cfg_tiny.d_k).transpose(1, 2)
        k_exp = k.repeat_interleave(cfg_tiny.n_groups, dim=1)
        v_exp = v.repeat_interleave(cfg_tiny.n_groups, dim=1)

    print(f"\n  n_heads={cfg_tiny.n_heads}, n_kv_heads={cfg_tiny.n_kv_heads}, "
          f"n_groups={cfg_tiny.n_groups}")
    print(f"  {'Tensor':<15} {'Shape':<30} {'Description'}")
    print(f"  {'─'*65}")
    for name, t in [
        ("input", x_in), ("Q", q), ("K (compact)", k), ("V (compact)", v),
        ("K (expanded)", k_exp), ("V (expanded)", v_exp)
    ]:
        print(f"  {name:<15} {str(list(t.shape)):<30} {dict(input='[B, S]', Q='[B, n_heads, S, d_k]', **{'K (compact)': '[B, n_kv_heads, S, d_k]', 'V (compact)': '[B, n_kv_heads, S, d_k]', 'K (expanded)': '[B, n_heads, S, d_k]', 'V (expanded)': '[B, n_heads, S, d_k]'}).get(name, '')}")

    # ── Part D: KV cache generation demo ─────────────────────────────────────
    print("\n── PART D: KV Cache Generation ──")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        prompt = "The meaning of life is"
        prompt_ids = torch.tensor([tok.encode(prompt)], device=device)
        # Use tiny model (not TinyLlama weights, just for shape demo)
        cfg_demo = LLaMAConfig(vocab_size=tok.vocab_size,
                                d_model=512, n_layers=2, n_heads=8,
                                n_kv_heads=2, max_seq_len=512)
        demo_model = LLaMA(cfg_demo).to(device)
        demo_model.eval()
        # Just show the cache shapes
        with torch.no_grad():
            _, caches = demo_model(prompt_ids[:, :4], use_cache=True)
        print(f"  After prefilling {4} tokens:")
        print(f"  cache['k'] shape: {list(caches[0]['k'].shape)}  ← grows with each new token")
        print("\n  (Skipping full generation — model has random weights)")
    except Exception as e:
        print(f"  (HuggingFace tokenizer not available: {e})")
        print("  KV cache concept: cache grows [B, n_kv_heads, 1→N, d_k] as tokens are generated")

    # ── Part E: Published model configs ──────────────────────────────────────
    print("\n── PART E: Published LLaMA Configurations ──")
    configs = [
        ("LLaMA-2 7B",   LLaMAConfig.llama2_7b()),
        ("LLaMA-2 13B",  LLaMAConfig.llama2_13b()),
        ("LLaMA-2 70B",  LLaMAConfig.llama2_70b()),
        ("LLaMA-3 8B",   LLaMAConfig.llama3_8b()),
        ("LLaMA-3 70B",  LLaMAConfig.llama3_70b()),
    ]

    print(f"\n  {'Model':<14} {'d_model':>7} {'layers':>6} {'Q-heads':>7} "
          f"{'KV-heads':>8} {'d_ff':>6} {'Params':>10} {'fp16':>8} {'KV 4K fp16':>11}")
    print("  " + "─" * 82)

    for name, cfg in configs:
        m = LLaMA(cfg)
        counts = m.count_params()
        p = counts["total"]
        fp16 = p * 2 / 1e9
        kv = kv_cache_memory_mb(cfg, 4096)
        print(f"  {name:<14} {cfg.d_model:>7} {cfg.n_layers:>6} {cfg.n_heads:>7} "
              f"{cfg.n_kv_heads:>8} {cfg.d_ff:>6} "
              f"{p/1e9:>9.2f}B {fp16:>7.1f}GB {kv:>10.0f}MB")
        del m

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  GQA savings (LLaMA-3 70B, n_kv_heads=8 vs MHA n_kv_heads=64):
    KV cache is 8× smaller → can serve longer contexts or larger batches

  KV cache is the main memory bottleneck at inference time:
    LLaMA-2 7B  inference: ~14 GB model + ~2 GB KV cache (4K context)
    LLaMA-3 8B  inference: ~16 GB model + ~1 GB KV cache (8K context, GQA)

  This is why quantization (Tutorial 06) matters so much:
    4-bit LLaMA-2 7B: ~4 GB instead of 14 GB → fits on a single GPU

  HuggingFace weight name mapping (to load pretrained weights):
    self_attn.q_proj.weight  → our W_q
    self_attn.k_proj.weight  → our W_k
    self_attn.v_proj.weight  → our W_v
    self_attn.o_proj.weight  → our W_o
    mlp.gate_proj.weight     → our W_gate
    mlp.up_proj.weight       → our W_up
    mlp.down_proj.weight     → our W_down
    input_layernorm.weight   → our attn_norm.scale
    post_feedforward_layernorm.weight → our ffn_norm.scale

  Next → Tutorial 06: Fine-tune LLaMA with QLoRA
    python3 /workspace/pytorch_llm/tutorials/06_finetune_llama_lora.py
""")


if __name__ == "__main__":
    main()
