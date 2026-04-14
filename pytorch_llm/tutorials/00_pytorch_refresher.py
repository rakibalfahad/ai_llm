"""
00_pytorch_refresher.py — PyTorch Core Concepts (Fast Refresher)

Target: Advanced Python developer who knows deep learning theory
        but needs to shake the rust off PyTorch syntax.

Covers:
  1.  Tensors — creation, indexing, reshaping, device transfer
  2.  Autograd — computational graph, backward, grad accumulation
  3.  nn.Module — custom layers, parameter registration
  4.  Training loop — forward, loss, backward, optimizer step
  5.  nn.Linear / nn.Embedding — building blocks used in LLMs
  6.  Dataset & DataLoader — batching, shuffling, custom datasets
  7.  GPU patterns — .to(device), non_blocking, mixed precision
  8.  Saving & loading — state_dict, checkpoints
  9.  Useful debugging tools — shapes, hooks, named parameters

Run inside Docker:
    docker run --rm --gpus all \\
      -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
      deeplearning:v100-llm \\
      python3 /workspace/pytorch_llm/tutorials/00_pytorch_refresher.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ─────────────────────────────────────────────────────────────────
# 1. TENSORS
# ─────────────────────────────────────────────────────────────────
section("1. Tensors")

# Creation
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.zeros(3, 4)
c = torch.ones(2, 3)
r = torch.randn(4, 5)           # N(0,1)
i = torch.arange(10)            # [0..9]
l = torch.linspace(0, 1, 5)    # [0.0, .25, .5, .75, 1.0]

print(f"randn(4,5) shape : {r.shape}")          # torch.Size([4, 5])
print(f"dtype            : {r.dtype}")           # torch.float32
print(f"device           : {r.device}")          # cpu

# Indexing  (exactly like NumPy)
print(f"r[0]         : {r[0]}")                  # first row
print(f"r[:, 1]      : {r[:, 1]}")               # second column
print(f"r[1:3, 2:4]  shape: {r[1:3, 2:4].shape}")   # [2,2] slice

# Boolean mask
mask = r > 0
print(f"positive values: {r[mask][:4]}")

# Reshaping
x = torch.randn(2, 3, 4)
print(f"\nOriginal  : {x.shape}")               # [2, 3, 4]
print(f"flatten   : {x.flatten(1).shape}")      # [2, 12]  — keep batch dim
print(f"view      : {x.view(2, -1).shape}")     # [2, 12]  — same as flatten here
print(f"permute   : {x.permute(0, 2, 1).shape}")# [2, 4, 3]  — swap axes
print(f"unsqueeze : {x.unsqueeze(0).shape}")    # [1, 2, 3, 4]
print(f"squeeze   : {x.unsqueeze(0).squeeze(0).shape}")  # [2, 3, 4]

# Math ops
p = torch.randn(3, 4)
q = torch.randn(4, 5)
print(f"\nmatmul(3×4, 4×5) → {(p @ q).shape}")  # [3, 5]

# einsum (used heavily in attention)
# "bqd,bkd->bqk" = batched dot product
B, T, D = 2, 5, 8
queries = torch.randn(B, T, D)
keys    = torch.randn(B, T, D)
scores  = torch.einsum("bqd,bkd->bqk", queries, keys)
print(f"einsum attention scores: {scores.shape}")   # [2, 5, 5]


# ─────────────────────────────────────────────────────────────────
# 2. AUTOGRAD
# ─────────────────────────────────────────────────────────────────
section("2. Autograd — Computational Graph")

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x          # y = x³ + 2x
y.backward()
print(f"x = {x.item():.1f}")
print(f"y = x³ + 2x = {y.item():.1f}")
print(f"dy/dx = 3x² + 2 = {x.grad.item():.1f}")   # 3*4+2 = 14

# Detach — stop gradient tracking (useful for target networks, inference)
z = y.detach()
print(f"z.requires_grad: {z.requires_grad}")       # False

# torch.no_grad() — inference mode, no graph built (saves memory)
with torch.no_grad():
    z2 = x ** 2
print(f"z2.requires_grad inside no_grad: {z2.requires_grad}")  # False

# Gradient accumulation gotcha: grads ADD UP — always zero before backward
x.grad.zero_()   # or optimizer.zero_grad()


# ─────────────────────────────────────────────────────────────────
# 3. nn.Module — Custom Layers
# ─────────────────────────────────────────────────────────────────
section("3. nn.Module")

class LinearWithBias(nn.Module):
    """Manual re-implementation of nn.Linear to show registration."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # nn.Parameter — tells PyTorch this tensor has gradients & is a param
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


layer = LinearWithBias(4, 8)
x_in  = torch.randn(2, 4)
out   = layer(x_in)
print(f"LinearWithBias output: {out.shape}")   # [2, 8]
print(f"Parameters:")
for name, p in layer.named_parameters():
    print(f"  {name:10s}  shape={tuple(p.shape)}  requires_grad={p.requires_grad}")


# Composing modules
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


mlp = MLP(16, 64, 10)
print(f"\nMLP parameter count: {sum(p.numel() for p in mlp.parameters()):,}")
print(mlp)


# ─────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────
section("4. Standard Training Loop")

# Toy problem: binary classification
torch.manual_seed(0)
X = torch.randn(200, 16)
y = (X[:, 0] + X[:, 1] > 0).long()   # simple rule

model     = MLP(16, 64, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()          # 1. clear old gradients
    logits = model(X)              # 2. forward pass
    loss   = criterion(logits, y)  # 3. compute loss
    loss.backward()                # 4. backprop
    optimizer.step()               # 5. update weights

    acc = (logits.argmax(1) == y).float().mean()
    if epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}  loss={loss.item():.4f}  acc={acc.item():.4f}")


# ─────────────────────────────────────────────────────────────────
# 5. EMBEDDINGs  (critical for LLMs)
# ─────────────────────────────────────────────────────────────────
section("5. nn.Embedding — Token Embeddings")

vocab_size = 1000
d_model    = 64

embed = nn.Embedding(vocab_size, d_model)
# Input: integer token IDs  — shape [batch, seq_len]
token_ids = torch.randint(0, vocab_size, (2, 10))   # batch=2, seq=10
embeddings = embed(token_ids)
print(f"Token IDs shape   : {token_ids.shape}")     # [2, 10]
print(f"Embeddings shape  : {embeddings.shape}")    # [2, 10, 64]

# Tied embeddings: output projection shares weights with the embedding table
# (LLaMA does this — saves memory)
output_logits = embeddings @ embed.weight.T          # [2, 10, 1000]
print(f"Tied logits shape : {output_logits.shape}") # [2, 10, 1000]


# ─────────────────────────────────────────────────────────────────
# 6. DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────
section("6. Dataset & DataLoader")

class TextDataset(Dataset):
    """Sliding-window token dataset — the standard LLM pre-training setup."""
    def __init__(self, tokens: list[int], seq_len: int):
        self.tokens  = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx     : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]  # shifted by 1
        return x, y


# Fake token stream
fake_tokens = list(range(500))
dataset     = TextDataset(fake_tokens, seq_len=32)
loader      = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

x_batch, y_batch = next(iter(loader))
print(f"Input  (x) shape: {x_batch.shape}")   # [8, 32]
print(f"Target (y) shape: {y_batch.shape}")   # [8, 32]  — same, shifted by 1


# ─────────────────────────────────────────────────────────────────
# 7. GPU PATTERNS
# ─────────────────────────────────────────────────────────────────
section("7. GPU Patterns")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Move model to GPU
model = MLP(16, 64, 10).to(device)

# Move data to GPU (non_blocking=True overlaps H→D transfer with compute)
X_gpu = torch.randn(32, 16).to(device, non_blocking=True)

with torch.no_grad():
    out = model(X_gpu)
print(f"Output on {out.device}: {out.shape}")

# Mixed precision — faster on Tensor Core GPUs (V100, A100)
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
optim  = torch.optim.AdamW(model.parameters())

x_batch = torch.randn(32, 16, device=device)
y_batch = torch.randint(0, 10, (32,), device=device)

optim.zero_grad()
with autocast("cuda"):
    logits = model(x_batch)
    loss   = F.cross_entropy(logits, y_batch)

scaler.scale(loss).backward()
scaler.unscale_(optim)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optim)
scaler.update()

print(f"Mixed-precision training step done. Loss: {loss.item():.4f}")

# Memory stats
if device.type == "cuda":
    print(f"GPU memory allocated : {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"GPU memory reserved  : {torch.cuda.memory_reserved()  / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────
# 8. SAVE & LOAD
# ─────────────────────────────────────────────────────────────────
section("8. Saving & Loading Checkpoints")

import os, tempfile

model     = MLP(16, 64, 10)
optimizer = torch.optim.AdamW(model.parameters())

# state_dict — recommended; saves only weights, not the class
ckpt = {
    "epoch"    : 5,
    "model"    : model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "val_loss" : 0.123,
}

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "ckpt.pt")
    torch.save(ckpt, path)

    loaded = torch.load(path, weights_only=True)
    model2 = MLP(16, 64, 10)
    model2.load_state_dict(loaded["model"])
    print(f"Loaded checkpoint from epoch {loaded['epoch']}, val_loss={loaded['val_loss']}")


# ─────────────────────────────────────────────────────────────────
# 9. DEBUGGING TOOLS
# ─────────────────────────────────────────────────────────────────
section("9. Useful Debugging Tools")

model = MLP(16, 64, 10)

# Named parameters — shapes at a glance
print("Named parameters:")
for name, p in model.named_parameters():
    print(f"  {name:30s} {tuple(p.shape)}")

# Total param count
total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params     : {total:,}")
print(f"Trainable params : {train:,}")

# Forward hook — inspect intermediate activations
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().shape
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        layer.register_forward_hook(make_hook(name))

_ = model(torch.randn(4, 16))
print("\nActivation shapes (via hooks):")
for name, shape in activations.items():
    print(f"  {name:30s} → {shape}")

# torch.autograd.set_detect_anomaly — catches NaN/Inf in backward
# (slow, use only for debugging)
print("\n✓ Refresher complete — you're ready for Tutorial 01!")
print("  Next: python3 /workspace/pytorch_llm/tutorials/01_tokenizer_bpe.py")
