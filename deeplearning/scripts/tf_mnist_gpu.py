"""
tf_mnist_gpu.py — Train a neural network on MNIST using TensorFlow with GPU.

Usage:
    docker run --rm --gpus all \
      -v $(pwd)/scripts:/workspace/scripts \
      -v $(pwd)/data:/workspace/data \
      deeplearning:v100-llm \
      python3 /workspace/scripts/tf_mnist_gpu.py

Outputs:
    - Training logs with GPU utilization
    - Saves trained model to /workspace/data/mnist_model_tf/
    - Prints test accuracy and per-class results
"""

import os
import time
import tensorflow as tf
import numpy as np

# ── Verify GPU ───────────────────────────────────────────────────────────────
print("=" * 60)
gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}")
print(f"GPUs available: {len(gpus)}")
for gpu in gpus:
    print(f"  {gpu}")
if not gpus:
    print("WARNING: No GPU found — training will be slow on CPU")
print("=" * 60)

# ── Load MNIST ───────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten to (N, 784)
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")

# ── Build model ──────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining on GPU..." if gpus else "\nTraining on CPU...")
start = time.time()

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

elapsed = time.time() - start
print(f"\nTraining completed in {elapsed:.1f}s")

# ── Evaluate ─────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Per-class accuracy
predictions = model.predict(x_test, verbose=0).argmax(axis=1)
print("\nPer-class accuracy:")
for digit in range(10):
    mask = y_test == digit
    acc = (predictions[mask] == digit).mean()
    print(f"  Digit {digit}: {acc:.4f}  ({mask.sum()} samples)")

# ── Save model ───────────────────────────────────────────────────────────────
save_dir = "/workspace/data/mnist_model_tf"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "mnist_dense.keras")
model.save(save_path)
print(f"\nModel saved to {save_path}")
print("=" * 60)
