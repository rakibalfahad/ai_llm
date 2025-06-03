import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

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

# Prepare the dataset by creating character mappings
def prepare_data(text):
    """Convert text to indices using a character vocabulary."""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_idx[ch] for ch in text]
    return data, char_to_idx, idx_to_char

# Generate input-target sequences for training
def get_batches(data, seq_length):
    """Create input-target pairs of sequences for training."""
    inputs = []
    targets = []
    for i in range(0, len(data) - seq_length, 1):
        inputs.append(data[i:i + seq_length])
        targets.append(data[i + 1:i + seq_length + 1])
    return inputs, targets

# Define the LSTM-based language model
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """Initialize the model with embedding, LSTM, and linear layers."""
        super(SimpleLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        """Forward pass: embedding -> LSTM -> linear."""
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for LSTM."""
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(num_layers, batch_size, hidden_size).to(device),
                  weight.new_zeros(num_layers, batch_size, hidden_size).to(device))
        return hidden

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

# Plot training loss and accuracy
def plot_metrics(losses, accuracies):
    """Generate and save a plot of training loss and accuracy."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Training function with loss and accuracy tracking
def train(model, data, epochs, seq_length, batch_size, lr):
    """Train the model, track loss and accuracy, and print progress."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    inputs, targets = get_batches(data, seq_length)
    num_batches = len(inputs) // batch_size
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        hidden = model.init_hidden(batch_size)
        
        for i in range(0, len(inputs) - batch_size, batch_size):
            batch_inputs = torch.tensor(inputs[i:i + batch_size], dtype=torch.long).to(device)
            batch_targets = torch.tensor(targets[i:i + batch_size], dtype=torch.long).to(device)
            
            hidden = tuple(h.detach() for h in hidden)
            model.zero_grad()
            
            output, hidden = model(batch_inputs, hidden)
            loss = criterion(output.view(-1, vocab_size), batch_targets.view(-1))
            
            # Compute accuracy
            _, predicted = torch.max(output, dim=-1)
            correct = (predicted == batch_targets).float().mean()
            accuracy = correct.item() * 100
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {i}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                losses.append(loss.item())
                accuracies.append(accuracy)
    
    # Plot and save metrics
    plot_metrics(losses, accuracies)

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    file_path = "input.txt"  # Input text file
    embed_size = 128        # Size of character embeddings
    hidden_size = 256       # LSTM hidden state size
    num_layers = 2          # Number of LSTM layers
    seq_length = 100        # Length of input sequences
    batch_size = 32         # Batch size for training
    epochs = 10             # Number of training epochs
    learning_rate = 0.001   # Learning rate for Adam optimizer
    
    # Load and prepare data
    text = load_data(file_path)
    data, char_to_idx, idx_to_char = prepare_data(text)
    vocab_size = len(char_to_idx)
    
    # Initialize model and move to device
    model = SimpleLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    
    # Print model architecture
    print_model_summary(model, vocab_size, seq_length)
    
    # Train the model
    train(model, data, epochs, seq_length, batch_size, learning_rate)