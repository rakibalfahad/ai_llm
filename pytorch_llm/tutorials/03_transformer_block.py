"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 03 — LLaMA-Style Transformer Block                         ║
║  pytorch_llm/tutorials/03_transformer_block.py                      ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. RMSNorm — LLaMA's normalisation (faster than LayerNorm)
  2. Rotary Positional Encoding (RoPE) — positional info in attention itself
  3. SwiGLU — LLaMA's feed-forward activation (better than ReLU/GELU)
  4. Assemble a complete LLaMA-style transformer block
  5. Stack blocks into a minimal LLaMA model skeleton
  6. Count parameters and estimate memory requirements

CHANGES FROM ORIGINAL TRANSFORMER → LLAMA
──────────────────────────────────────────
  Original (GPT-2)          LLaMA-2 / LLaMA-3
  ─────────────────         ──────────────────
  LayerNorm (after block)   RMSNorm (before each sub-layer = "pre-norm")
  Absolute positional PE    Rotary PE (RoPE) applied inside attention
  GELU activation           SwiGLU in feed-forward
  MHA (all heads equal)     Grouped Query Attention (LLaMA-2 34B+, LLaMA-3)

RUN IN DOCKER
─────────────
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/03_transformer_block.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RMSNorm (Root Mean Square Layer Normalization)
# ══════════════════════════════════════════════════════════════════════════════
#
# LayerNorm (original transformer):
#   y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β
#   Learnable: γ (scale), β (shift)
#
# RMSNorm (Zhang & Sennrich 2019, used in LLaMA):
#   y = x / RMS(x) * γ
#   RMS(x) = sqrt(mean(x²) + ε)
#   Learnable: γ only (no β)
#
# Why RMSNorm?
#   - ~10% faster: skip mean subtraction and β
#   - Empirically matches LayerNorm quality on LLMs
#   - Simpler gradient flow

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applied PRE-block in LLaMA ("pre-norm architecture").
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(d_model))  # learnable γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model]"""
        # Compute RMS over last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Rotary Positional Encoding (RoPE)
# ══════════════════════════════════════════════════════════════════════════════
#
# Problem with sinusoidal PE:
#   Adding PE to embeddings before attention mixes positional info with
#   content. Also, it doesn't generalise well to positions > max_seq_len.
#
# RoPE (Su et al. 2021) insight:
#   Instead of modifying the embedding, rotate the Q and K vectors inside
#   attention. The rotation angle depends on the position. This way:
#
#     <q_m, k_n> (dot product) = f(q, m) · f(k, n)
#
#   where f applies a rotation by angle θ * position. The dot product then
#   depends only on the RELATIVE distance (m - n), which is exactly what
#   we want — positional relationship, not absolute position.
#
# Implementation:
#   For each pair of dims (2i, 2i+1) in Q/K:
#     [q_2i  ]   [ cos(mθ_i)  -sin(mθ_i) ] [q_2i  ]
#     [q_2i+1] = [ sin(mθ_i)   cos(mθ_i) ] [q_2i+1]
#
#   θ_i = 1 / (base^(2i/d_k))   where base=10000 (original), or 500000 (LLaMA-3)
#
# This rotation is computed efficiently via complex multiplication:
#   (q_2i + j*q_2i+1) * (cos(mθ) + j*sin(mθ))

def precompute_rope_freqs(d_k: int, max_seq_len: int,
                           base: float = 10000.0,
                           device: torch.device = torch.device("cpu")
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin frequencies for RoPE.
    Returns (cos, sin) each of shape [max_seq_len, d_k//2].
    """
    # θ_i = 1 / (base^(2i/d_k))
    i = torch.arange(0, d_k, 2, device=device).float()   # [d_k/2]
    theta = 1.0 / (base ** (i / d_k))                    # [d_k/2]

    positions = torch.arange(max_seq_len, device=device).float()  # [S]
    # Outer product: freq[pos, i] = pos * theta_i
    freqs = torch.outer(positions, theta)                          # [S, d_k/2]

    return freqs.cos(), freqs.sin()   # [S, d_k/2]


def apply_rope(x: torch.Tensor,
               cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional encoding to Q or K.

    x   : [batch, n_heads, seq, d_k]
    cos : [seq, d_k//2]
    sin : [seq, d_k//2]

    Strategy: rotate pairs (x[..., 0::2], x[..., 1::2])
      rotated_even = x_even * cos - x_odd  * sin
      rotated_odd  = x_even * sin + x_odd  * cos
    """
    seq = x.size(2)
    cos = cos[:seq].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_k/2]
    sin = sin[:seq].unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]   # [B, H, S, d_k/2]
    x_odd  = x[..., 1::2]   # [B, H, S, d_k/2]

    x_rot_even = x_even * cos - x_odd  * sin
    x_rot_odd  = x_even * sin + x_odd  * cos

    # Interleave back: [B, H, S, d_k]
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    return x_rot


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SwiGLU Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard transformer FFN:
#   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2     (ReLU)
#
# GPT used GELU:
#   FFN(x) = GELU(xW_1)W_2
#
# LLaMA uses SwiGLU (Shazeer 2020):
#   FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down
#
#   Swish(x) = x * sigmoid(x)   ≈ smooth ReLU
#   ⊙ = elementwise multiplication (gating mechanism)
#
# Why SwiGLU?
#   The gate (W_gate) learns to selectively suppress certain neurons.
#   This is a soft version of ReLU's hard 0/1 gate.
#   Empirically outperforms GELU in language models (PaLM, LLaMA).
#
# Hidden dimension:
#   Original FFN: d_ff = 4 * d_model
#   SwiGLU uses d_ff = (2/3) * 4 * d_model  rounded to multiple of 256
#   (because we have 3 matrices instead of 2, so we reduce to match params)

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA.

    3 linear layers (no bias):
      W_gate : d_model → d_ff
      W_up   : d_model → d_ff
      W_down : d_ff    → d_model
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        if d_ff is None:
            # LLaMA formula: (2/3)*4*d_model, rounded to nearest 256
            d_ff_raw = int(2 / 3 * 4 * d_model)
            d_ff = (d_ff_raw + 255) // 256 * 256
        self.d_ff = d_ff

        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, d_model]"""
        # gate branch: Swish activation
        gate = F.silu(self.W_gate(x))   # SiLU = Swish = x * sigmoid(x)
        # up branch: linear projection
        up   = self.W_up(x)
        # elementwise gate × up, then project down
        return self.W_down(gate * up)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Multi-Head Attention with RoPE
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttentionRoPE(nn.Module):
    """
    Causal multi-head self-attention with Rotary Positional Encoding.
    This is the attention used in LLaMA-2 (without Grouped Query Attention).
    """

    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # Precompute RoPE frequencies (registered as buffer, not a parameter)
        cos_freq, sin_freq = precompute_rope_freqs(self.d_k, max_seq_len)
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K (not V — V doesn't need positional info)
        q = apply_rope(q, self.cos_freq, self.sin_freq)
        k = apply_rope(k, self.cos_freq, self.sin_freq)

        # Causal attention using PyTorch's flash attention (if CUDA available)
        # is_causal=True automatically applies the causal mask
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLaMA-Style Transformer Block
# ══════════════════════════════════════════════════════════════════════════════
#
# A LLaMA block applies operations in this order (pre-norm):
#
#   x = x + Attention(RMSNorm(x))    ← residual connection
#   x = x + FFN(RMSNorm(x))          ← residual connection
#
# The residual connections are critical: they create a "highway" for
# gradients to flow directly to early layers without vanishing.
# This is why transformers can be stacked very deep (32-80 layers).

class LLaMABlock(nn.Module):
    """
    One LLaMA-style transformer block.
    Repeating this N times gives the full LLaMA model.
    """

    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm1   = RMSNorm(d_model)
        self.attn    = CausalSelfAttentionRoPE(d_model, n_heads, max_seq_len, dropout)
        self.norm2   = RMSNorm(d_model)
        self.ffn     = SwiGLUFFN(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-norm + FFN + residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Stacked Model Skeleton (LLaMA Core)
# ══════════════════════════════════════════════════════════════════════════════

class LLaMACore(nn.Module):
    """
    Core of a LLaMA-style language model:
      Embedding → N × LLaMABlock → RMSNorm → Linear (logits)

    This is the full architecture — we'll train it in Tutorial 04 and
    build the complete LLaMA-2 variant in Tutorial 05.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 max_seq_len: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, d_model)
        self.blocks   = nn.ModuleList([
            LLaMABlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and lm_head share weights (common in LLMs)
        # The intuition: if a token has a high embedding value, it should
        # score high in the output logits too.
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.W_o.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            nn.init.normal_(block.ffn.W_down.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq]  →  logits: [batch, seq, vocab_size]"""
        x = self.embed(token_ids)      # [B, S, d_model]
        for block in self.blocks:
            x = block(x)               # [B, S, d_model]
        x = self.norm_out(x)           # [B, S, d_model]
        return self.lm_head(x)         # [B, S, vocab_size]

    def count_params(self) -> dict[str, int]:
        counts = {}
        counts["embed"]   = self.embed.weight.numel()
        counts["blocks"]  = sum(p.numel() for b in self.blocks for p in b.parameters())
        counts["lm_head"] = 0  # tied with embed — don't double count
        counts["total"]   = sum(p.numel() for p in self.parameters())
        # subtract tied parameters
        counts["total_unique"] = counts["total"] - self.lm_head.weight.numel()
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Memory Estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_memory(n_params: int, seq_len: int, batch_size: int,
                    d_model: int, n_layers: int, n_heads: int,
                    precision: str = "float16") -> None:
    """Rough estimate of GPU memory needed."""
    bytes_per_param = 2 if precision == "float16" else 4
    bytes_per_grad  = bytes_per_param    # gradients same size as params
    bytes_per_optim = 8                  # Adam: 2 × float32 state per param

    model_mb  = n_params * bytes_per_param / 1e6
    grad_mb   = n_params * bytes_per_grad  / 1e6
    optim_mb  = n_params * bytes_per_optim / 1e6

    # Activation memory (rough): batch × seq × d_model × n_layers × constant
    # (real cost includes Q/K/V, attention weights, FFN intermediates)
    act_bytes = batch_size * seq_len * d_model * n_layers * 10 * bytes_per_param
    act_mb    = act_bytes / 1e6

    total_train_mb = model_mb + grad_mb + optim_mb + act_mb
    total_infer_mb = model_mb + act_mb * 0.1   # no grads needed

    print(f"\n  Memory estimate ({precision}, batch={batch_size}, seq={seq_len}):")
    print(f"    Parameters  : {model_mb:,.0f} MB")
    print(f"    Gradients   : {grad_mb:,.0f} MB")
    print(f"    Optimizer   : {optim_mb:,.0f} MB  (Adam)")
    print(f"    Activations : {act_mb:,.0f} MB  (approx)")
    print(f"    ─────────────────────────────────────")
    print(f"    Training    : ~{total_train_mb/1024:,.1f} GB")
    print(f"    Inference   : ~{total_infer_mb/1024:,.2f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tutorial 03 — LLaMA-Style Transformer Block")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Part A: RMSNorm ──────────────────────────────────────────────────────
    print("\n── PART A: RMSNorm ──")
    x = torch.randn(2, 10, 256, device=device)
    norm = RMSNorm(256).to(device)
    out = norm(x)
    print(f"  Input  mean/std: {x.mean():.4f} / {x.std():.4f}")
    print(f"  Output mean/std: {out.mean():.4f} / {out.std():.4f}")
    print(f"  RMSNorm shape  : {out.shape}")
    print(f"  Learnable params: {sum(p.numel() for p in norm.parameters())} (just γ)")

    # ── Part B: RoPE ─────────────────────────────────────────────────────────
    print("\n── PART B: Rotary Positional Encoding (RoPE) ──")
    d_k, seq_len = 64, 16
    cos_f, sin_f = precompute_rope_freqs(d_k, max_seq_len=512, device=device)
    print(f"  Precomputed freqs: cos {cos_f.shape}, sin {sin_f.shape}")

    q_test = torch.randn(1, 4, seq_len, d_k, device=device)
    q_rot  = apply_rope(q_test, cos_f, sin_f)
    print(f"  Q before RoPE shape: {q_test.shape}")
    print(f"  Q after  RoPE shape: {q_rot.shape}")
    # Rotation preserves vector norms
    norm_before = q_test.norm(dim=-1).mean().item()
    norm_after  = q_rot.norm(dim=-1).mean().item()
    print(f"  Norm before: {norm_before:.4f}  |  after: {norm_after:.4f}  (should match)")

    # ── Part C: SwiGLU FFN ───────────────────────────────────────────────────
    print("\n── PART C: SwiGLU Feed-Forward Network ──")
    d_model = 256
    ffn = SwiGLUFFN(d_model).to(device)
    x_ffn = torch.randn(2, 10, d_model, device=device)
    out_ffn = ffn(x_ffn)
    ffn_params = sum(p.numel() for p in ffn.parameters())
    print(f"  d_model={d_model}, d_ff={ffn.d_ff}")
    print(f"  Input  shape : {x_ffn.shape}")
    print(f"  Output shape : {out_ffn.shape}")
    print(f"  FFN params   : {ffn_params:,}  (3 matrices: W_gate, W_up, W_down)")

    # ── Part D: Full LLaMA Block ─────────────────────────────────────────────
    print("\n── PART D: LLaMA Block ──")
    n_heads = 8
    block = LLaMABlock(d_model, n_heads).to(device)
    x_block = torch.randn(2, 20, d_model, device=device)
    out_block = block(x_block)
    block_params = sum(p.numel() for p in block.parameters())
    print(f"  d_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
    print(f"  Input  shape : {x_block.shape}")
    print(f"  Output shape : {out_block.shape}")
    print(f"  Block params : {block_params:,}")

    # ── Part E: LLaMA Core (stacked blocks) ─────────────────────────────────
    print("\n── PART E: Stacked LLaMA Core ──")

    configs = {
        "Tiny   (tutorial)": dict(vocab_size=32000, d_model=256,  n_layers=4,  n_heads=8),
        "Small  (~100M)":    dict(vocab_size=32000, d_model=512,  n_layers=8,  n_heads=8),
        "Medium (~1B)":      dict(vocab_size=32000, d_model=2048, n_layers=16, n_heads=16),
        "LLaMA-2 (7B)":      dict(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32),
    }

    print(f"\n  {'Config':<22} {'d_model':>7} {'layers':>6} {'heads':>5} {'Params':>12} {'fp16 size':>10}")
    print("  " + "-" * 68)

    for name, cfg in configs.items():
        model = LLaMACore(**cfg)
        counts = model.count_params()
        total_m = counts["total_unique"]
        size_gb = total_m * 2 / 1e9  # fp16
        print(f"  {name:<22} {cfg['d_model']:>7} {cfg['n_layers']:>6} "
              f"{cfg['n_heads']:>5} {total_m:>12,} {size_gb:>9.2f}GB")

    # Run a forward pass on the tiny model
    tiny_cfg = configs["Tiny   (tutorial)"]
    tiny_model = LLaMACore(**tiny_cfg, dropout=0.0).to(device)
    token_ids = torch.randint(0, 32000, (2, 64), device=device)
    logits = tiny_model(token_ids)
    print(f"\n  Tiny model forward pass:")
    print(f"    Input  : {token_ids.shape}")
    print(f"    Logits : {logits.shape}  (batch, seq, vocab_size)")

    # ── Part F: Memory estimation ─────────────────────────────────────────────
    print("\n── PART F: Memory Estimation ──")
    tiny_params = LLaMACore(**configs["Tiny   (tutorial)"]).count_params()["total_unique"]
    llama7b_params = 7_000_000_000

    print("\n  Tiny model:")
    estimate_memory(tiny_params, seq_len=512, batch_size=16,
                    d_model=256, n_layers=4, n_heads=8)

    print("\n  LLaMA-2 7B (full fine-tune, fp16):")
    estimate_memory(llama7b_params, seq_len=2048, batch_size=1,
                    d_model=4096, n_layers=32, n_heads=32)

    print("\n  LLaMA-2 7B inference only (fp16):")
    estimate_memory(llama7b_params, seq_len=2048, batch_size=1,
                    d_model=4096, n_layers=32, n_heads=32)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — LLaMA innovations over original transformer")
    print("=" * 70)
    print("""
  Component    Original (GPT-2)         LLaMA-2
  ─────────    ────────────────         ──────────────────────────
  Norm         LayerNorm (post-block)   RMSNorm (pre-block, faster)
  Position     Sinusoidal / learned     RoPE (relative, generalises)
  Activation   GELU in FFN              SwiGLU in FFN (gated, better)
  Norm params  γ + β                    γ only (50% fewer)
  Bias         Yes in many layers       No bias anywhere (cleaner)
  Weight tie   Sometimes                Always (embed ↔ lm_head)

  Pre-norm advantage:
    Post-norm: norm(x + sublayer(x))  — gradient vanishes in deep stacks
    Pre-norm:  x + sublayer(norm(x))  — stable gradients at any depth

  Next → Tutorial 04: Train a complete Mini-LLM on Shakespeare
    python3 /workspace/pytorch_llm/tutorials/04_gpt_mini.py
""")


if __name__ == "__main__":
    main()
