"""
pt_mnist_gpu.py — Train a CNN on MNIST using PyTorch with GPU.

Usage:
    docker run --rm --gpus all \
      -v $(pwd)/scripts:/workspace/scripts \
      -v $(pwd)/data:/workspace/data \
      deeplearning:v100-llm \
      python3 /workspace/scripts/pt_mnist_gpu.py

Outputs:
    - Training & validation metrics per epoch
    - Test accuracy and per-class breakdown
    - Saves model to /workspace/data/mnist_model_pt/
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Device ───────────────────────────────────────────────────────────────────
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA   : {torch.version.cuda}")
print("=" * 60)

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

data_dir = "/workspace/data/mnist"
train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=256, pin_memory=True)

print(f"Train: {len(train_set)} samples  |  Test: {len(test_set)} samples")

# ── Model ────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Train ────────────────────────────────────────────────────────────────────
epochs = 5
print(f"\nTraining {epochs} epochs on {device}...")
start = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"  Epoch {epoch}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.4f}")

elapsed = time.time() - start
print(f"\nTraining completed in {elapsed:.1f}s")

# ── Evaluate ─────────────────────────────────────────────────────────────────
model.eval()
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += (preds[i] == labels[i]).item()
            class_total[label] += 1

print(f"\nTest accuracy: {correct / total:.4f}  ({correct}/{total})")
print("\nPer-class accuracy:")
for digit in range(10):
    acc = class_correct[digit] / class_total[digit]
    print(f"  Digit {digit}: {acc:.4f}  ({class_total[digit]} samples)")

# ── Save ─────────────────────────────────────────────────────────────────────
save_dir = "/workspace/data/mnist_model_pt"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "mnist_cnn.pt")
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to {save_path}")
print("=" * 60)
