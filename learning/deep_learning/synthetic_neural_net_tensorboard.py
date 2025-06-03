import torch  # Core PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network modules (layers, activations, loss functions)
import torch.optim as optim  # Optimization algorithms (e.g., Adam)
import numpy as np  # NumPy for generating synthetic data
import matplotlib.pyplot as plt  # Plotting library for static visualizations
from torch.utils.tensorboard import SummaryWriter  # TensorBoard for real-time progress and graph visualization

# Initialize TensorBoard writer
# Logs saved to 'e:\MyCodes\PytorchTutorial\runs' for real-time visualization and model graph
writer = SummaryWriter(log_dir='runs/synthetic_experiment')

# Set random seed for reproducibility across runs
torch.manual_seed(42)  # Seed for CPU operations to ensure consistent results
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)  # Seed for GPU operations on RTX 4070

# 1. Generate synthetic 2D dataset
def generate_data(n_samples=1000):
    """
    Generate synthetic 2D data for binary classification.
    Args:
        n_samples (int): Number of data points to generate
    Returns:
        tuple: (X, y) where X is tensor of shape (n_samples, 2) and y is tensor of labels
    """
    np.random.seed(42)  # Set NumPy seed for consistent data generation
    X = np.random.randn(n_samples, 2)  # Generate 1000 points with 2 features (x, y)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)  # Label 1 if x + y > 0, else 0
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Create dataset
X, y = generate_data()  # Generate 1000 points and corresponding labels

# 2. Set device for computation
# Check if CUDA is available (RTX 4070 supports CUDA 12.1) and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Display device (should be 'cuda' for RTX 4070)
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # Confirm GPU is RTX 4070

# 3. Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Initialize the neural network for 2D synthetic data classification.
        Inherits from nn.Module, the base class for all PyTorch models.
        Model Architecture:
            - Input Layer: 2 input features (x, y coordinates of 2D points)
            - First Hidden Layer: Linear transformation (2 -> 64 units) + ReLU activation
            - Dropout: 20% dropout to prevent overfitting
            - Second Hidden Layer: Linear transformation (64 -> 32 units) + ReLU activation
            - Dropout: 20% dropout for additional regularization
            - Output Layer: Linear transformation (32 -> 2 units) for binary classification
        Total Parameters:
            - Linear(2, 64): 2*64 + 64 = 192 parameters (weights + biases)
            - Linear(64, 32): 64*32 + 32 = 2080 parameters
            - Linear(32, 2): 32*2 + 2 = 66 parameters
            - Total: 192 + 2080 + 66 = 2338 trainable parameters
        """
        super(NeuralNetwork, self).__init__()  # Initialize parent nn.Module class
        # Sequential container: Defines a stack of layers for processing
        self.linear_relu_stack = nn.Sequential(
            # First linear layer: Maps 2 input features to 64 hidden units
            nn.Linear(2, 64),
            # ReLU activation: Applies f(x) = max(0, x) for non-linearity
            nn.ReLU(),
            # Dropout: Randomly sets 20% of activations to 0 during training
            nn.Dropout(0.2),
            # Second linear layer: Maps 64 units to 32 units for deeper feature extraction
            nn.Linear(64, 32),
            # ReLU activation: Adds non-linearity to second layer
            nn.ReLU(),
            # Dropout: Additional 20% dropout for regularization
            nn.Dropout(0.2),
            # Output layer: Maps 32 units to 2 classes for binary classification
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Define the forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        logits = self.linear_relu_stack(x)  # Pass input through the sequential stack
        return logits  # Return raw scores (logits) for each class

# 4. Initialize model, loss function, and optimizer
model = NeuralNetwork().to(device)  # Create model instance and move to GPU
print(model)  # Output model architecture to console for verification

# Calculate and print total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")  # Should print 2338

# Log model graph to TensorBoard
# Create a sample input tensor with shape (1, 2) to trace the forward pass
sample_input = torch.randn(1, 2).to(device)  # Batch of 1 sample with 2 features
writer.add_graph(model, sample_input)  # Add computational graph to TensorBoard

# Define loss function: CrossEntropyLoss for binary classification
# Combines log-softmax and negative log-likelihood for multi-class problems
criterion = nn.CrossEntropyLoss()

# Define optimizer: Adam with learning rate 0.001
# Adam adapts learning rates per parameter, improving convergence
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Move data to GPU
X, y = X.to(device), y.to(device)  # Move input data and labels to GPU (RTX  4070)

# 6. Train the model
num_epochs = 100  # Number of training iterations over the dataset
train_losses = []  # List to store loss values for each epoch
train_accuracies = []  # List to store accuracy values for each epoch

for epoch in range(num_epochs):
    model.train()  # Set model to training mode (activates dropout)
    
    # Forward pass: Compute model predictions and loss
    outputs = model(X)  # Get logits for all data (shape: [1000, 2])
    loss = criterion(outputs, y)  # Compute loss using CrossEntropyLoss

    # Backward pass: Compute gradients and update weights
    optimizer.zero_grad()  # Clear previous gradients to prevent accumulation
    loss.backward()  # Compute gradients via backpropagation
    optimizer.step()  # Update model parameters using Adam optimizer

    # Track loss and accuracy
    train_losses.append(loss.item())  # Store scalar loss value
    _, predicted = torch.max(outputs, 1)  # Get predicted class indices
    accuracy = (predicted == y).float().mean().item() * 100.0  # Compute accuracy (%)
    train_accuracies.append(accuracy)  # Store accuracy value

    # Log metrics to TensorBoard for real-time visualization
    writer.add_scalar('Loss/Train', loss.item(), epoch)  # Log training loss
    writer.add_scalar('Accuracy/Train', accuracy, epoch)  # Log training accuracy

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

# Close TensorBoard writer to save logs
writer.close()  # Finalize and save TensorBoard logs

# 7. Evaluate the model
model.eval()  # Set model to evaluation mode (deactivates dropout)
with torch.no_grad():  # Disable gradient computation for efficiency
    outputs = model(X)  # Compute predictions on training data
    _, predicted = torch.max(outputs, 1)  # Get predicted class indices
    accuracy = (predicted == y).float().mean().item() * 100.0  # Calculate final accuracy
print(f"Final Training Accuracy: {accuracy:.2f}%")

# 8. Visualize training metrics with Matplotlib
plt.figure(figsize=(12, 5))  # Create figure for static plots

# Plot training loss
plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
plt.plot(train_losses, label='Training Loss')  # Plot loss values
plt.xlabel('Epoch')  # X-axis label
plt.ylabel('Loss')  # Y-axis label
plt.title('Training Loss Curve')  # Plot title
plt.legend()  # Show legend

# Plot training accuracy
plt.subplot(1, 2, 2)  # Second subplot
plt.plot(train_accuracies, label='Training Accuracy')  # Plot accuracy values
plt.xlabel('Epoch')  # X-axis label
plt.ylabel('Accuracy (%)')  # Y-axis label
plt.title('Training Accuracy Curve')  # Plot title
plt.legend()  # Show legend

plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Display static plots

# 9. Visualize decision boundary
def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary and data points for the 2D synthetic dataset.
    Args:
        model: Trained neural network
        X (torch.Tensor): Input data of shape (n_samples, 2)
        y (torch.Tensor): Labels of shape (n_samples,)
    """
    model.eval()  # Set model to evaluation mode to disable dropout
    # Calculate grid boundaries with dynamic margins based on data range
    x_range = X[:, 0].max() - X[:, 0].min()  # Range of first feature
    y_range = X[:, 1].max() - X[:, 1].min()  # Range of second feature
    x_min = X[:, 0].min() - 0.1 * x_range  # Add 10% margin to min x
    x_max = X[:, 0].max() + 0.1 * x_range  # Add 10% margin to max x
    y_min = X[:, 1].min() - 0.1 * y_range  # Add 10% margin to min y
    y_max = X[:, 1].max() + 0.1 * y_range  # Add 10% margin to max y

    # Create a grid of points for plotting decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # Predict classes for grid points
    with torch.no_grad():  # Disable gradients for efficiency
        Z = model(grid)  # Compute predictions for grid points
        _, Z = torch.max(Z, 1)  # Get predicted class indices
    Z = Z.cpu().numpy().reshape(xx.shape)  # Reshape predictions for plotting

    # Plot decision boundary and data points
    plt.figure(figsize=(8, 6))  # Create figure for visualization
    plt.contourf(xx, yy, Z, alpha=0.3)  # Plot decision boundary with transparency
    plt.scatter(X.cpu()[:, 0], X.cpu()[:, 1], c=y.cpu(), alpha=0.8)  # Plot data points
    plt.xlabel('Feature 1')  # Label for x-axis (first feature)
    plt.ylabel('Feature 2')  # Label for y-axis (second feature)
    plt.title('Decision Boundary')  # Title of the plot
    plt.show()  # Display the plot

# Call visualization function
plot_decision_boundary(model, X, y)