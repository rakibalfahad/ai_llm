import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
import math
import warnings

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load text data from a file
def load_data(file_path):
    """Read the input text file and return its content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Create a character-level tokenizer
def create_tokenizer(text):
    """Create and train a character-level tokenizer."""
    # Initialize a WordLevel tokenizer (used for character-level by treating each char as a token)
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    trainer = WordLevelTrainer(special_tokens=["<pad>", "<unk>"])
    
    # Train tokenizer on the input text (character-level)
    tokenizer.train_from_iterator([text], trainer)
    
    # Enable padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"))
    
    # Wrap with PreTrainedTokenizerFast for Hugging Face compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>"
    )
    return fast_tokenizer

# Prepare the dataset by tokenizing and creating sequences
def prepare_data(text, tokenizer, seq_length):
    """Tokenize text and create input-target sequences."""
    # Split text into training (80%) and validation (20%)
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Tokenize
    train_ids = tokenizer.encode(train_text, add_special_tokens=False)
    val_ids = tokenizer.encode(val_text, add_special_tokens=False)
    
    # Create sequences
    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []
    
    for i in range(0, len(train_ids) - seq_length, 1):
        train_inputs.append(train_ids[i:i + seq_length])
        train_targets.append(train_ids[i + 1:i + seq_length + 1])
    
    for i in range(0, len(val_ids) - seq_length, 1):
        val_inputs.append(val_ids[i:i + seq_length])
        val_targets.append(val_ids[i + 1:i + seq_length + 1])
    
    return (train_inputs, train_targets), (val_inputs, val_targets)

# Generate batches for training
def get_batches(inputs, targets, batch_size):
    """Create batches of input-target pairs."""
    for i in range(0, len(inputs) - batch_size, batch_size):
        yield (torch.tensor(inputs[i:i + batch_size], dtype=torch.long).to(device),
               torch.tensor(targets[i:i + batch_size], dtype=torch.long).to(device))

# Positional encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Initialize positional encoding for Transformer."""
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, :x.size(1), :]

# Define the GPT-like Transformer model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads):
        """Initialize the GPT-like Transformer model."""
        super(SimpleGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x, tgt_mask=None):
        """Forward pass: embedding -> positional encoding -> transformer -> linear."""
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, x, tgt_mask=tgt_mask)
        x = self.fc(x)
        return x

# Generate causal mask for Transformer
def generate_square_subsequent_mask(sz):
    """Generate a causal mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask.to(device)

# Print model architecture summary
def print_model_summary(model, vocab_size, seq_length):
    """Print a custom summary of the model architecture and parameters."""
    print("\nModel Architecture:")
    print("------------------")
    print(model)
    print("\nModel Parameters:")
    print("-----------------")
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: {param.shape} ({param_count} parameters)")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Input Shape: [batch_size, {seq_length}]")
    print(f"Output Shape: [batch_size, {seq_length}, {vocab_size}]")

# Plot training loss, accuracy, and validation loss
def plot_metrics(losses, accuracies, val_losses):
    """Generate and save a plot of training metrics."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_losses, 'g-', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics_v1.png')
    plt.close()

# Training function with advanced features
def train(model, tokenizer, train_data, val_data, epochs, batch_size, lr, patience=3):
    """Train the model with gradient clipping, LR scheduling, and early stopping."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * (len(train_data[0]) // batch_size)
    if total_steps == 0:
        total_steps = 1
        warnings.warn("Training dataset is too small, resulting in zero training steps. Using total_steps=1 for scheduler. "
                      "Please provide a larger input.txt file with more text.")
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    criterion = nn.CrossEntropyLoss()
    
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    best_val_loss = float('inf')
    patience_counter = 0
    losses, accuracies, val_losses = [], [], []
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        total_loss, total_correct = 0, 0
        total_samples = 0
        
        for i, (batch_inputs, batch_targets) in enumerate(get_batches(train_inputs, train_targets, batch_size)):
            optimizer.zero_grad()
            tgt_mask = generate_square_subsequent_mask(batch_inputs.size(1))
            output = model(batch_inputs, tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
            
            # Compute accuracy
            _, predicted = torch.max(output, dim=-1)
            correct = (predicted == batch_targets).float().mean()
            accuracy = correct.item() * 100
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_correct += correct.item() * batch_inputs.numel()
            total_samples += batch_inputs.numel()
            
            if i % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {i}/{len(train_inputs)//batch_size}, "
                      f"Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%")
                losses.append(loss.item())
                accuracies.append(accuracy)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in get_batches(val_inputs, val_targets, batch_size):
                tgt_mask = generate_square_subsequent_mask(batch_inputs.size(1))
                output = model(batch_inputs, tgt_mask=tgt_mask)
                val_loss += criterion(output.view(-1, vocab_size), batch_targets.view(-1)).item()
        
        val_loss /= (len(val_inputs) // batch_size) or 1  # Avoid division by zero
        val_losses.append(val_loss)
        print(f"Epoch: {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model and tokenizer
            os.makedirs('saved_model', exist_ok=True)
            torch.save(model.state_dict(), 'saved_model/model.pt')
            tokenizer.save_pretrained('saved_model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot and save metrics
    plot_metrics(losses, accuracies, val_losses)

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    file_path = "input.txt"  # Input text file
    embed_size = 128        # Embedding dimension
    hidden_size = 256       # Transformer hidden units
    num_layers = 4          # Transformer layers
    num_heads = 4           # Attention heads
    seq_length = 100        # Sequence length
    batch_size = 32         # Batch size
    epochs = 10             # Number of training epochs
    learning_rate = 0.001   # Learning rate
    patience = 3            # Early stopping patience
    
    # Load and prepare data
    text = load_data(file_path)
    tokenizer = create_tokenizer(text)
    vocab_size = tokenizer.vocab_size
    train_data, val_data = prepare_data(text, tokenizer, seq_length)
    
    # Initialize model and move to device
    model = SimpleGPT(vocab_size, embed_size, hidden_size, num_layers, num_heads).to(device)
    
    # Print model architecture
    print_model_summary(model, vocab_size, seq_length)
    
    # Train the model
    train(model, tokenizer, train_data, val_data, epochs, batch_size, learning_rate, patience)
