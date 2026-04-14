"""
train_cifar10_resnet.py — Train a custom ResNet on CIFAR-10 (PyTorch).

Features:
  - Custom ResNet-18 built from scratch (no torchvision models)
  - Mixed-precision training (torch.amp) for faster GPU throughput
  - OneCycleLR scheduler with warm-up
  - Gradient clipping
  - Early stopping with patience
  - Checkpoint saving (best & last)
  - TensorBoard logging (loss, accuracy, LR, GPU memory)
  - Per-class accuracy breakdown at evaluation
  - Reproducible via seed control

Usage — inside Docker:
    docker run --rm --gpus all \\
      -v $(pwd)/scripts:/workspace/scripts \\
      -v $(pwd)/data:/workspace/data \\
      deeplearning:v100-llm \\
      python3 /workspace/scripts/train_cifar10_resnet.py

    # Custom hyperparameters:
    docker run --rm --gpus all \\
      -v $(pwd)/scripts:/workspace/scripts \\
      -v $(pwd)/data:/workspace/data \\
      deeplearning:v100-llm \\
      python3 /workspace/scripts/train_cifar10_resnet.py \\
        --epochs 50 --batch-size 256 --lr 0.1 --patience 10

    # With TensorBoard:
    docker run --rm --gpus all \\
      -p 6006:6006 \\
      -v $(pwd)/scripts:/workspace/scripts \\
      -v $(pwd)/data:/workspace/data \\
      deeplearning:v100-llm \\
      bash -c "python3 /workspace/scripts/train_cifar10_resnet.py & tensorboard --logdir /workspace/data/runs --host 0.0.0.0"

Outputs:
  /workspace/data/cifar10_resnet/
    best_model.pt       — Best validation accuracy checkpoint
    last_model.pt       — Final epoch checkpoint
    runs/               — TensorBoard event files
"""

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 ResNet Trainer")
    p.add_argument("--epochs",      type=int,   default=30,    help="Max training epochs")
    p.add_argument("--batch-size",  type=int,   default=128,   help="Batch size")
    p.add_argument("--lr",          type=float, default=0.05,  help="Max learning rate (OneCycleLR)")
    p.add_argument("--weight-decay",type=float, default=5e-4,  help="L2 regularization")
    p.add_argument("--patience",    type=int,   default=7,     help="Early-stopping patience (epochs)")
    p.add_argument("--seed",        type=int,   default=42,    help="Random seed")
    p.add_argument("--workers",     type=int,   default=4,     help="DataLoader worker threads")
    p.add_argument("--val-split",   type=float, default=0.1,   help="Fraction of train set used for validation")
    p.add_argument("--output-dir",  type=str,   default="/workspace/data/cifar10_resnet")
    p.add_argument("--data-dir",    type=str,   default="/workspace/data/cifar10")
    return p.parse_args()


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Model: Custom ResNet-18 ──────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Standard residual block with optional projection shortcut."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    """
    Configurable ResNet for CIFAR-10 (32×32 inputs).
    Default: ResNet-18  [2, 2, 2, 2] → 11.2 M parameters.
    """
    def __init__(self,
                 block=BasicBlock,
                 layers=(2, 2, 2, 2),
                 num_classes: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.in_channels = 64

        # Stem: smaller kernel for 32×32 inputs (no max-pool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # Weight init (Kaiming for conv, constant for BN)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ─── Data ─────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASSES = ("airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck")

def get_dataloaders(data_dir, batch_size, val_split, workers):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(data_dir, train=True,
                                  download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(data_dir, train=False,
                                  download=True, transform=test_tf)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val])
    # Use clean transforms for validation subset
    val_set.dataset = datasets.CIFAR10(data_dir, train=True,
                                       download=False, transform=test_tf)

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size*2, shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size*2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─── Training & Evaluation ────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, training=True):
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def evaluate_per_class(model, loader, device):
    model.eval()
    class_correct = torch.zeros(10)
    class_total   = torch.zeros(10)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with autocast():
                preds = model(images).argmax(1).cpu()
            for c in range(10):
                mask = labels == c
                class_correct[c] += (preds[mask] == c).sum()
                class_total[c]   += mask.sum()
    return class_correct / class_total.clamp(min=1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"  PyTorch {torch.__version__}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}")
        print(f"  Memory : {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA   : {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.val_split, args.workers)
    print(f"  Train batches: {len(train_loader)}  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ResNet().to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet-18 (CIFAR) — {n_params:,} trainable parameters")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    scaler = GradScaler()

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir  = os.path.join(args.output_dir, "runs")
    writer  = SummaryWriter(tb_dir)
    best_ckpt = os.path.join(args.output_dir, "best_model.pt")
    last_ckpt = os.path.join(args.output_dir, "last_model.pt")

    best_val_acc = 0.0
    patience_ctr = 0

    print(f"\n  Training for up to {args.epochs} epochs (patience={args.patience})...\n")
    print(f"  {'Epoch':>5}  {'LR':>8}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>8}  {'GPU Mem':>8}")
    print("  " + "-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            scheduler=scheduler, training=True)

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            scheduler=None, training=False)

        elapsed = time.time() - t0
        gpu_mem = (torch.cuda.memory_reserved(0) / 1e9 if device.type == "cuda" else 0.0)

        print(f"  {epoch:>5}  {current_lr:>8.5f}  {train_loss:>10.4f}  "
              f"{train_acc:>9.4f}  {val_loss:>8.4f}  {val_acc:>8.4f}  "
              f"{gpu_mem:>6.2f}GB  ({elapsed:.1f}s)")

        # TensorBoard
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR",        current_lr,                             epoch)
        if device.type == "cuda":
            writer.add_scalar("GPU/memory_GB", gpu_mem, epoch)

        # Save last checkpoint
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_acc":    val_acc,
        }, last_ckpt)

        # Best checkpoint + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "val_acc": val_acc,
            }, best_ckpt)
            print(f"  ✓ New best val acc: {best_val_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    writer.close()

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"  Best epoch: {ckpt['epoch']}  |  Val acc: {ckpt['val_acc']:.4f}")

    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, optimizer, scaler, device,
        scheduler=None, training=False)
    print(f"\n  Test accuracy : {test_acc:.4f}")
    print(f"  Test loss     : {test_loss:.4f}")

    per_class = evaluate_per_class(model, test_loader, device)
    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASSES):
        bar = "█" * int(per_class[i].item() * 20)
        print(f"    {name:>12s}  {per_class[i]:.4f}  {bar}")

    print(f"\n  Checkpoints : {args.output_dir}/")
    print(f"  TensorBoard : tensorboard --logdir {tb_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
