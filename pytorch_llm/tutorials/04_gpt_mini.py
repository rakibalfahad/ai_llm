"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 04 — Train a Mini-LLM from Scratch                         ║
║  pytorch_llm/tutorials/04_gpt_mini.py                               ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. Autoregressive pre-training (next-token prediction)
  2. Cross-entropy loss as a language model objective
  3. Perplexity — how to measure LM quality
  4. Complete training loop: mixed precision, gradient accumulation,
     gradient clipping, LR scheduling with warm-up
  5. Text generation: greedy, temperature, top-k, top-p (nucleus) sampling
  6. TensorBoard logging of loss, perplexity, LR, GPU stats

ARCHITECTURE
────────────
  A character-level LLaMA-style model (~10M params) trained on Shakespeare.
  Small enough to train in <10 minutes on a V100.

RUN IN DOCKER
─────────────
  # Training only
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/04_gpt_mini.py

  # With TensorBoard
  docker run --rm --gpus all \\
    -p 7777:7777 \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    bash -c "python3 /workspace/pytorch_llm/tutorials/04_gpt_mini.py & \\
             tensorboard --logdir /workspace/data/llm_runs --host 0.0.0.0 --port 7777"
"""

import os
import math
import time
import urllib.request
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Mini-LLM trainer")
    # Model
    p.add_argument("--d-model",  type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads",  type=int, default=8)
    p.add_argument("--seq-len",  type=int, default=256,  help="Context window")
    p.add_argument("--dropout",  type=float, default=0.1)
    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int,   default=200)
    p.add_argument("--grad-accum",   type=int,   default=1,  help="Gradient accumulation steps")
    p.add_argument("--max-grad-norm",type=float, default=1.0)
    p.add_argument("--val-interval", type=int,   default=200, help="Evaluate every N steps")
    # Paths
    p.add_argument("--data-dir",   type=str, default="/workspace/data")
    p.add_argument("--output-dir", type=str, default="/workspace/data/mini_llm")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data
# ══════════════════════════════════════════════════════════════════════════════

def load_shakespeare(data_dir: str) -> str:
    """Download or load the Tiny Shakespeare dataset (~1MB)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare dataset...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded to {path}")
        except Exception:
            print("Download failed. Using built-in excerpt.")
            with open(path, "w") as f:
                f.write(SHAKESPEARE_EXCERPT * 100)
    with open(path) as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")
    return text


# Built-in fallback excerpt (used if network unavailable)
SHAKESPEARE_EXCERPT = """
HAMLET: To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.

KING LEAR: Blow, winds, and crack your cheeks! rage! blow!
You cataracts and hurricanoes, spout
Till you have drench'd our steeples, drown'd the cocks!

MACBETH: Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.

ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.

OTHELLO: She loved me for the dangers I had pass'd,
And I loved her that she did pity them.
"""


class CharDataset(torch.utils.data.Dataset):
    """
    Character-level dataset. Each sample is a fixed-length sequence.

    The target is the input shifted by 1:
      input:  [t0, t1, t2, ..., t_{L-1}]
      target: [t1, t2, t3, ..., t_L]

    This trains the model to predict the next character at every position.
    This is "teacher forcing" — during training, we always feed the true
    previous token, not the model's own prediction.
    """

    def __init__(self, text: str, char_to_id: dict, seq_len: int):
        self.seq_len    = seq_len
        self.char_to_id = char_to_id
        self.data       = torch.tensor(
            [char_to_id[c] for c in text if c in char_to_id],
            dtype=torch.long
        )
        print(f"Dataset: {len(self.data):,} tokens, {len(self)} samples")

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def build_vocab(text: str) -> tuple[dict, dict]:
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    print(f"Vocabulary: {len(chars)} unique characters")
    return char_to_id, id_to_char


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model (reuses components from Tutorial 03)
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale


def precompute_rope(d_k: int, max_len: int, device):
    theta = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    pos   = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    s = x.size(2)
    c, s_ = cos[:s].unsqueeze(0).unsqueeze(0), sin[:s].unsqueeze(0).unsqueeze(0)
    xe, xo = x[..., 0::2], x[..., 1::2]
    return torch.stack([xe * c - xo * s_, xe * s_ + xo * c], dim=-1).flatten(-2)


class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048, dropout=0.0):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = dropout
        cos, sin = precompute_rope(self.dk, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, S, self.h, self.dk).transpose(1, 2) for t in qkv]
        q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        dp = self.drop if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        return self.out(o.transpose(1, 2).contiguous().view(B, S, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(2 / 3 * 4 * d_model); d_ff = (d_ff + 255) // 256 * 256
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.0):
        super().__init__()
        self.n1  = RMSNorm(d_model)
        self.attn = CausalMHA(d_model, n_heads, max_len, dropout)
        self.n2  = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class MiniLLM(nn.Module):
    """Complete mini language model."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads,
                 max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)
        ])
        self.norm   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying

        # Initialise
        nn.init.normal_(self.embed.weight, std=0.02)
        for b in self.blocks:
            nn.init.normal_(b.attn.out.weight,  std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(b.ffn.down.weight,  std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            # Flatten: [B*S, vocab] vs [B*S]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def count_params(self) -> int:
        # Subtract tied head weight (same as embed)
        total = sum(p.numel() for p in self.parameters())
        return total - self.head.weight.numel()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Learning Rate Schedule with Warm-up + Cosine Decay
# ══════════════════════════════════════════════════════════════════════════════
#
# LLMs typically use:
#   1. Linear warm-up for the first N steps (avoids large early updates)
#   2. Cosine annealing to decay LR to some minimum
#
# This is implemented as a custom LambdaLR.

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     min_lr_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)          # linear warm-up
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)                # cosine decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Text Generation
# ══════════════════════════════════════════════════════════════════════════════
#
# Generation strategies (all autoregressive — one token at a time):
#
#   Greedy    : always pick argmax(logits)   → deterministic but repetitive
#   Temperature : scale logits by 1/T before softmax
#                 T<1: more peaked (confident), T>1: more spread (creative)
#   Top-k     : zero out all but the top-k logits, then sample
#   Top-p     : zero out tokens until cumulative prob < p (nucleus sampling)
#               Used in ChatGPT, LLaMA inference

@torch.no_grad()
def generate(model: MiniLLM,
             prompt_ids: torch.Tensor,        # [1, S]
             max_new_tokens: int,
             id_to_char: dict,
             strategy: str = "top_p",
             temperature: float = 0.8,
             top_k: int = 50,
             top_p: float = 0.9,
             seq_len: int = 256) -> str:
    """Generate text autoregressively."""
    model.eval()
    generated = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Crop context to model's max sequence length
        ctx = generated[:, -seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / temperature   # [1, vocab]

        if strategy == "greedy":
            next_id = logits.argmax(dim=-1, keepdim=True)

        elif strategy == "temperature":
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        elif strategy == "top_k":
            # Zero out logits below the k-th largest
            topk_vals, _ = logits.topk(top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits    = logits.masked_fill(logits < threshold, float("-inf"))
            probs     = F.softmax(logits, dim=-1)
            next_id   = torch.multinomial(probs, num_samples=1)

        elif strategy == "top_p":
            # Nucleus sampling
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens once cumulative prob exceeds p
            sorted_logits[cumprobs - sorted_logits.softmax(dim=-1) > top_p] = float("-inf")
            # Scatter back to original order
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        generated = torch.cat([generated, next_id], dim=-1)

    return "".join(id_to_char[i.item()] for i in generated[0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, max_batches=50) -> tuple[float, float]:
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / max(1, n)
    perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow
    model.train()
    return avg_loss, perplexity


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tutorial 04 — Train a Mini-LLM (LLaMA-style Character Model)")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    text = load_shakespeare(args.data_dir)
    char_to_id, id_to_char = build_vocab(text)
    vocab_size = len(char_to_id)

    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]

    train_ds = CharDataset(train_text, char_to_id, args.seq_len)
    val_ds   = CharDataset(val_text,   char_to_id, args.seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    n_params = model.count_params()
    print(f"\nModel: {n_params:,} parameters  ({n_params/1e6:.1f}M)")
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, seq_len={args.seq_len}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler   = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler      = GradScaler(enabled=(device.type == "cuda"))

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.data_dir, "llm_runs", "mini_llm"))

    best_val_loss = float("inf")
    global_step   = 0

    print(f"\nStarting training ({args.epochs} epochs, "
          f"{len(train_loader)} batches/epoch)...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            # Gradient accumulation: update only every grad_accum steps
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum

            # Periodic validation and logging
            if global_step > 0 and global_step % args.val_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                train_loss_avg = epoch_loss / max(1, step + 1)
                current_lr = scheduler.get_last_lr()[0]

                writer.add_scalar("Loss/train", train_loss_avg, global_step)
                writer.add_scalar("Loss/val",   val_loss,       global_step)
                writer.add_scalar("Perplexity/val", val_ppl,    global_step)
                writer.add_scalar("LR",         current_lr,     global_step)

                if device.type == "cuda":
                    mem_gb = torch.cuda.memory_reserved() / 1e9
                    writer.add_scalar("GPU/memory_GB", mem_gb, global_step)

                print(f"  step {global_step:5d} | "
                      f"train_loss={train_loss_avg:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"ppl={val_ppl:.1f} | "
                      f"lr={current_lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model":      model.state_dict(),
                        "optimizer":  optimizer.state_dict(),
                        "step":       global_step,
                        "val_loss":   val_loss,
                        "char_to_id": char_to_id,
                        "id_to_char": id_to_char,
                        "config": {
                            "vocab_size": vocab_size,
                            "d_model":    args.d_model,
                            "n_layers":   args.n_layers,
                            "n_heads":    args.n_heads,
                            "seq_len":    args.seq_len,
                        }
                    }, os.path.join(args.output_dir, "best_model.pt"))

        epoch_time = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} done — {epoch_time:.1f}s\n")

    writer.close()

    # ── Generation demo ───────────────────────────────────────────────────────
    print("=" * 70)
    print("  TEXT GENERATION DEMO")
    print("=" * 70)

    model.eval()
    prompt = "HAMLET:\n"
    prompt_ids = torch.tensor(
        [[char_to_id.get(c, 0) for c in prompt]], device=device
    )

    strategies = [
        ("greedy",      dict()),
        ("temperature", dict(temperature=0.7)),
        ("top_k",       dict(temperature=0.8, top_k=40)),
        ("top_p",       dict(temperature=0.8, top_p=0.9)),
    ]

    for name, kwargs in strategies:
        print(f"\n── Strategy: {name} ──")
        generated = generate(
            model, prompt_ids, max_new_tokens=200, id_to_char=id_to_char,
            strategy=name, seq_len=args.seq_len, **kwargs
        )
        print(generated)

    print("\n" + "=" * 70)
    print(f"  Best model saved to: {args.output_dir}/best_model.pt")
    print("""
  Key takeaways:
    ✓ Next-token prediction on unlabeled text = self-supervised pre-training
    ✓ Cross-entropy loss = negative log-likelihood of correct next token
    ✓ Perplexity = exp(loss) = how many choices the model is "choosing from"
    ✓ Greedy decoding is deterministic but repetitive
    ✓ Top-p (nucleus) sampling generates more natural text

  Next → Tutorial 05: Full LLaMA Architecture with KV Cache
    python3 /workspace/pytorch_llm/tutorials/05_llama_architecture.py
""")


if __name__ == "__main__":
    args = parse_args()
    train(args)
