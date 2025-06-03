# Simple Language Model Training with PyTorch: Tutorial

This tutorial guides you through setting up and running `train_simple_lm.py`, a Python script that trains a simple LSTM-based language model using PyTorch with GPU support. The model predicts the next character in a sequence, trained on a text file (`input.txt`). It tracks training loss and accuracy, generates a plot of these metrics, and prints a summary of the model architecture.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **PyTorch**: With CUDA support for GPU acceleration (optional, falls back to CPU if GPU is unavailable).
- **Matplotlib**: For plotting training loss and accuracy.
- **Text File**: A file named `input.txt` containing training text (e.g., a book, article, or any plain text).
- **GPU (Optional)**: A CUDA-compatible GPU with appropriate drivers for faster training.

## Setup
1. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install torch matplotlib
   ```

2. **Prepare the Input File**:
   - Create a text file named `input.txt` with the training data (e.g., copy text from a book or article).
   - Place `input.txt` in the same directory as `train_simple_lm.py`.

3. **Project Structure**:
   Organize your project directory as follows:
   ```
   project/
   ├── input.txt
   ├── train_simple_lm.py
   ├── simple_lm_tutorial.md
   ```

4. **Verify GPU Setup (Optional)**:
   - If using a GPU, ensure CUDA is installed and compatible with your PyTorch version. Check compatibility at [PyTorch's official site](https://pytorch.org/get-started/locally/).
   - The script automatically detects and uses a GPU if available, otherwise it uses the CPU.

## Running the Script
1. **Navigate to the Project Directory**:
   ```bash
   cd path/to/project
   ```

2. **Run the Script**:
   Execute the script using Python:
   ```bash
   python train_simple_lm.py
   ```

3. **Expected Output**:
   - The script will:
     - Print the device used (e.g., `Using device: cuda` or `Using device: cpu`).
     - Display a model architecture summary, including layers, parameter shapes, and total parameters.
     - Print training loss and accuracy every 100 steps.
     - Save a plot named `training_metrics.png` in the project directory, showing training loss and accuracy curves.

## Code Explanation
The `train_simple_lm.py` script includes the following components:

### 1. Data Loading and Preparation
- **Function**: `load_data(file_path)`
  - Reads the text from `input.txt`.
- **Function**: `prepare_data(text)`
  - Creates a vocabulary of unique characters and maps them to indices.
  - Converts the text into a sequence of indices.
- **Function**: `get_batches(data, seq_length)`
  - Generates input-target pairs of sequences for training (e.g., input: sequence of 100 characters, target: next 100 characters).

### 2. Model Architecture
- **Class**: `SimpleLM(nn.Module)`
  - Defines a simple LSTM-based language model with:
    - **Embedding Layer**: Maps characters to dense vectors (size: 128).
    - **LSTM Layer**: Processes sequences with 2 layers and 256 hidden units.
    - **Linear Layer**: Outputs probabilities for the next character (vocab size).
  - The model predicts the next character in a sequence.
- **Function**: `print_model_summary(model, vocab_size, seq_length)`
  - Prints a custom summary of the model, including layer details, parameter counts, and input/output shapes.

### 3. Training
- **Function**: `train(model, data, epochs, seq_length, batch_size, lr)`
  - Trains the model using the Adam optimizer and cross-entropy loss.
  - Processes the text in batches (batch size: 32, sequence length: 100).
  - Tracks loss and accuracy (percentage of correctly predicted characters).
  - Prints progress every 100 steps.
- **GPU Support**: Uses CUDA if available, ensuring efficient training on GPU-enabled systems.

### 4. Visualization
- **Function**: `plot_metrics(losses, accuracies)`
  - Generates a plot with two subplots:
    - Left: Training loss over steps.
    - Right: Training accuracy over steps.
  - Saves the plot as `training_metrics.png`.

## Example Output
### Model Summary
For a vocabulary size of 65 characters, the model summary might look like:
```
Using device: cuda

Model Architecture:
------------------
SimpleLM(
  (embedding): Embedding(65, 128)
  (lstm): LSTM(128, 256, num_layers=2, batch_first=True)
  (fc): Linear(in_features=256, out_features=65, bias=True)
)

Model Parameters:
-----------------
embedding.weight: torch.Size([65, 128]) (8320 parameters)
lstm.weight_ih_l0: torch.Size([1024, 128]) (131072 parameters)
...
Total Parameters: 946,125
Input Shape: [batch_size, 100]
Output Shape: [batch_size, 100, 65]
```

### Training Progress
During training, you’ll see output like:
```
Epoch: 1/10, Step: 0/123, Loss: 4.1742, Accuracy: 3.12%
Epoch: 1/10, Step: 100/123, Loss: 3.2456, Accuracy: 25.67%
...
```

### Plot
A file named `training_metrics.png` will be saved, showing loss and accuracy curves.

## Notes
- **Hyperparameters**: Adjust in `train_simple_lm.py` as needed:
  - `embed_size`: 128 (embedding dimension).
  - `hidden_size`: 256 (LSTM hidden units).
  - `num_layers`: 2 (LSTM layers).
  - `seq_length`: 100 (sequence length).
  - `batch_size`: 32.
  - `epochs`: 10.
  - `learning_rate`: 0.001.
- **Model Limitations**: This is a character-based model for simplicity. For better performance, consider token-based models or larger architectures (e.g., transformers).
- **Extending the Script**: You can add text generation functionality by implementing a sampling function using the trained model.
- **Troubleshooting**:
  - **FileNotFoundError**: Ensure `input.txt` exists in the project directory.
  - **GPU Issues**: Verify CUDA compatibility with your PyTorch version.
  - **Plotting Issues**: Ensure `matplotlib` is installed.

## Contributing
If you enhance the script (e.g., add text generation, support for larger datasets), consider contributing via a pull request to the repository.