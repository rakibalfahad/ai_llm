"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 02 — Attention Mechanism                                   ║
║  pytorch_llm/tutorials/02_attention_mechanism.py                    ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. Scaled dot-product attention — the core formula of every transformer
  2. Why we scale by √d_k and why it matters
  3. Causal (autoregressive) masking — makes generation left-to-right
  4. Multi-head attention — parallel attention in different subspaces
  5. Compare your implementation with PyTorch's built-in version

RUN IN DOCKER
─────────────
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/02_attention_mechanism.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — The Attention Intuition
# ══════════════════════════════════════════════════════════════════════════════
#
# Imagine you're reading the sentence: "The animal didn't cross the street
# because it was too tired."
#
# When you read "it", your brain attends back to "animal" — not "street".
# That is exactly what attention computes: for each token, how much should
# it "look at" every other token?
#
# Mathematically:
#
#   Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V
#
#   Q  (Query)  — "what am I looking for?"
#   K  (Key)    — "what do I offer?"
#   V  (Value)  — "what do I actually return if selected?"
#
# The formula is a weighted average of Values, where the weights come from
# the dot-product similarity between Queries and Keys.
#
# Example with 3 tokens [q1, k1, v1], [q2, k2, v2], [q3, k3, v3]:
#
#   scores = Q @ Kᵀ          shape [3, 3]   (all pairwise similarities)
#   weights = softmax(scores / √d_k)         (normalise to sum=1 per row)
#   output = weights @ V                     (weighted sum of Values)
#
# Each output token is a blend of all value vectors. The blend is determined
# by how similar the token's query is to every key. This is differentiable,
# so the network learns what to attend to.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Scaled Dot-Product Attention
# ══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    q: torch.Tensor,          # [batch, heads, seq_q, d_k]
    k: torch.Tensor,          # [batch, heads, seq_k, d_k]
    v: torch.Tensor,          # [batch, heads, seq_k, d_v]
    mask: torch.Tensor | None = None,   # [batch, 1, seq_q, seq_k]  (optional)
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core attention computation. Returns (output, attention_weights).

    The √d_k scaling:
      Without scaling, large d_k → large dot products → softmax saturates
      (gradient ≈ 0). Dividing by √d_k keeps the variance constant:
        Var(q·k) = d_k * Var(q_i) * Var(k_i) = d_k   (if each dim ~ N(0,1))
        After scaling: Var(q·k / √d_k) = 1
    """
    d_k = q.size(-1)
    scale = math.sqrt(d_k)

    # Step 1: similarity scores
    # q: [B, H, Sq, dk]  ×  k.T: [B, H, dk, Sk]  →  [B, H, Sq, Sk]
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    # Step 2: apply causal mask (if provided)
    # Mask shape: [B, 1, Sq, Sk] — True means "ignore this position"
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 3: softmax along the key dimension (last dim)
    weights = F.softmax(scores, dim=-1)

    # Handle NaN rows (all -inf after masking in first few tokens)
    weights = torch.nan_to_num(weights, nan=0.0)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    # Step 4: weighted sum of values
    # weights: [B, H, Sq, Sk]  ×  v: [B, H, Sk, dv]  →  [B, H, Sq, dv]
    output = torch.matmul(weights, v)

    return output, weights


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Causal (autoregressive) mask. Position i can only attend to positions ≤ i.

    For seq_len=4:
      [[False, True,  True,  True ],
       [False, False, True,  True ],
       [False, False, False, True ],
       [False, False, False, False]]

    True  = MASKED (set to -inf before softmax)
    False = ALLOWED
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════════
#
# Instead of one big attention with d_model dimensions, we run h smaller
# attentions in parallel (each with d_k = d_model / h dimensions).
#
# Why multiple heads?
#   Different heads learn to attend to different aspects:
#     Head 1 might focus on syntactic dependencies (subject-verb)
#     Head 2 might focus on coreference (pronoun → noun)
#     Head 3 might focus on positional proximity
#
# Steps:
#   1. Project input X to Q, K, V with learned weight matrices W_Q, W_K, W_V
#   2. Split into h heads along the d_model dimension
#   3. Run attention independently in each head
#   4. Concatenate head outputs
#   5. Apply final output projection W_O

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention (or cross-attention if q_x and kv_x differ).

    Architecture:
        X  →  [W_Q] → Q  ─┐
        X  →  [W_K] → K  ──→ Attention(Q,K,V) →  concat  →  [W_O]  →  output
        X  →  [W_V] → V  ─┘
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # dimension per head

        # Separate projection for Q, K, V  (could fuse into one linear for speed)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        # Small initial scale prevents attention weights from saturating at init
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,                 # [batch, seq, d_model]
        mask: torch.Tensor | None = None  # [batch, 1, seq, seq]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape

        # Step 1: linear projections  [B, S, d_model]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Step 2: split into heads and reshape for batched attention
        # [B, S, d_model] → [B, S, n_heads, d_k] → [B, n_heads, S, d_k]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # Now: q/k/v are [B, n_heads, S, d_k]

        # Step 3: scaled dot-product attention per head
        dropout_p = self.dropout if self.training else 0.0
        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=dropout_p
        )
        # attn_out: [B, n_heads, S, d_k]

        # Step 4: merge heads back  [B, n_heads, S, d_k] → [B, S, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Step 5: output projection
        output = self.W_o(attn_out)

        return output, attn_weights  # [B, S, d_model], [B, n_heads, S, S]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Sinusoidal Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════
#
# Attention is permutation-invariant: if you shuffle token order, self-attention
# gives the same output (just shuffled). This is a problem — word order matters!
#
# Solution: add a position-dependent signal to each token embedding before
# feeding it to attention.
#
# Original transformer (Vaswani 2017) used fixed sinusoidal encodings:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# LLaMA uses RoPE instead (covered in Tutorial 03). We show sinusoidal here
# to understand the concept before moving to the modern version.

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) positional encoding from the original transformer.
    Each position gets a unique vector that the model can use to infer order.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build encoding matrix [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()        # [S, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                             # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)   # even dims
        pe[:, 1::2] = torch.cos(pos * div)   # odd dims

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, S, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, d_model]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Visualisation Helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_attention_weights(weights: torch.Tensor, tokens: list[str]) -> None:
    """
    Print attention weight matrix as an ASCII heatmap.
    weights: [n_heads, seq, seq] (for one batch item)
    """
    import sys
    head_weights = weights[0]   # first head
    S = min(len(tokens), head_weights.size(-1))
    tokens_short = [t[:6] for t in tokens[:S]]

    print("\n  Attention weights (Head 0) — darker = higher weight:")
    print("  " + "  ".join(f"{t:>6s}" for t in tokens_short))
    shade = " ░▒▓█"

    for i, row_tok in enumerate(tokens_short):
        row = head_weights[i, :S]
        bars = ""
        for w in row:
            idx = min(int(w.item() * (len(shade) - 1) * 4), len(shade) - 1)
            bars += shade[idx] * 2 + " "
        print(f"  {row_tok:>6s} | {bars}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tutorial 02 — Attention Mechanism")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Part A: Raw attention on made-up tensors ─────────────────────────────
    print("\n── PART A: Scaled Dot-Product Attention (single head) ──")
    B, S, d_k = 1, 6, 64
    q = torch.randn(B, 1, S, d_k, device=device)
    k = torch.randn(B, 1, S, d_k, device=device)
    v = torch.randn(B, 1, S, d_k, device=device)

    # Without mask (bidirectional — encoder-style)
    out_bidir, w_bidir = scaled_dot_product_attention(q, k, v)
    print(f"  Input shape  : {q.shape}")
    print(f"  Output shape : {out_bidir.shape}")
    print(f"  Weight shape : {w_bidir.shape}")
    print(f"  Weights sum  : {w_bidir.sum(-1)}  (must be 1.0 per row)")

    # With causal mask
    mask = make_causal_mask(S, device)
    out_causal, w_causal = scaled_dot_product_attention(q, k, v, mask=mask)
    print(f"\n  Causal mask shape : {mask.shape}")
    print(f"  Upper-triangle weights (should be ~0):")
    print(f"  {w_causal[0, 0].round(decimals=3)}")

    # ── Part B: Multi-Head Attention module ──────────────────────────────────
    print("\n── PART B: Multi-Head Attention Module ──")
    d_model, n_heads = 256, 8
    seq_len = 10

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0).to(device)

    x = torch.randn(2, seq_len, d_model, device=device)   # batch=2
    mask = make_causal_mask(seq_len, device)

    mha.eval()
    with torch.no_grad():
        out, weights = mha(x, mask=mask)

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"  d_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Weight shape : {weights.shape}  (batch, heads, seq, seq)")
    print(f"  Parameters   : {total_params:,}")

    # ── Part C: Attention pattern on a real sentence ─────────────────────────
    print("\n── PART C: Attention Pattern Visualisation ──")
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    S = len(tokens)
    d_model_small = 32
    n_heads_small = 2

    mha_small = MultiHeadAttention(d_model_small, n_heads_small, dropout=0.0).to(device)
    mha_small.eval()

    # Embed tokens with a random embedding (in practice, nn.Embedding)
    embed = nn.Embedding(100, d_model_small).to(device)
    token_ids = torch.tensor([0, 1, 2, 3, 4, 5], device=device).unsqueeze(0)
    x_small = embed(token_ids)   # [1, 6, 32]

    # Causal mask
    mask_small = make_causal_mask(S, device)

    with torch.no_grad():
        out_small, w_small = mha_small(x_small, mask=mask_small)

    print_attention_weights(w_small[0], tokens)

    # ── Part D: Positional Encoding ──────────────────────────────────────────
    print("\n── PART D: Sinusoidal Positional Encoding ──")
    pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=20).to(device)
    x_pe = torch.zeros(1, 8, 64, device=device)
    x_pe_out = pe(x_pe)
    print(f"  PE adds unique signal per position:")
    for pos in [0, 1, 3, 7]:
        v = x_pe_out[0, pos, :4].tolist()
        print(f"  Position {pos}: first 4 dims = {[f'{x:.3f}' for x in v]}")

    # ── Part E: Compare with PyTorch built-in ────────────────────────────────
    print("\n── PART E: Compare with nn.MultiheadAttention ──")
    mha_builtin = nn.MultiheadAttention(d_model, n_heads, batch_first=True).to(device)
    x_cmp = torch.randn(2, seq_len, d_model, device=device)

    # nn.MHA uses additive mask (True = ignore)
    attn_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1
    ).bool()

    with torch.no_grad():
        out_builtin, _ = mha_builtin(x_cmp, x_cmp, x_cmp, attn_mask=attn_mask)

    print(f"  Our output shape    : {out.shape}")
    print(f"  PyTorch output shape: {out_builtin.shape}")
    print("  (Values differ — different weights — but shapes match: PASS)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  Key formulas:
    Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V

  Key numbers for LLaMA-2 (7B):
    d_model  = 4096
    n_heads  = 32
    d_k      = 4096 / 32 = 128
    seq_len  = 4096 (context window)

  Attention complexity:
    Time  : O(seq² × d_model)  — quadratic in sequence length!
    Memory: O(seq² × n_heads)

  This quadratic scaling is why long-context models are hard.
  LLaMA 3 solves this partly with Grouped Query Attention (Tutorial 05).

  Key concepts:
    ✓ Scaled dot-product attention
    ✓ Causal masking (upper-triangle = -inf)
    ✓ Multi-head: parallel attention in h subspaces
    ✓ Sinusoidal positional encoding

  Next → Tutorial 03: Transformer Block (RMSNorm, RoPE, SwiGLU)
    python3 /workspace/pytorch_llm/tutorials/03_transformer_block.py
""")


if __name__ == "__main__":
    main()
