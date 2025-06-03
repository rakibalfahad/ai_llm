# Simple Language Model Training with Hugging Face and PyTorch: Tutorial

This tutorial guides you through setting up and running `train_simple_lm_v1.py`, a Python script that trains a Transformer-based (GPT-like) language model using PyTorch and Hugging Face's `transformers` and `tokenizers` libraries with GPU support. The model predicts the next character in a sequence, trained on a text file (`input.txt`). It includes advanced features like gradient clipping, learning rate scheduling, early stopping, model/tokenizer saving, and training loss/accuracy tracking.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **PyTorch**: With CUDA support for GPU acceleration (optional, falls back to CPU).
- **Hugging Face Transformers and Tokenizers**: For tokenization and model utilities.
- **Matplotlib**: For plotting training loss and accuracy.
- **Text File**: A file named `input.txt` containing training text (e.g., a book, article, or any plain text).
- **GPU (Optional)**: A CUDA-compatible GPU with appropriate drivers.

## Setup
1. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install torch transformers tokenizers matplotlib
   ```

2. **Prepare the Input File**:
   - Create a text file named `input.txt` with the training data (e.g., copy text from a book or article).
   - Place `input.txt` in the same directory as `train_simple_lm_v1.py`.

3. **Project Structure**:
   Organize your project directory as follows:
   ```
   project/
   ├── input.txt
   ├── train_simple_lm_v1.py
   ├── simple_lm_tutorial.md
   ```

4. **Verify GPU Setup (Optional)**:
   - If using a GPU, ensure CUDA is installed and compatible with your PyTorch version. Check compatibility at [PyTorch's official site](https://pytorch.org/get-started/locally/).
   - The script automatically detects and uses a GPU if available.

## Running the Script
1. **Navigate to the Project Directory**:
   ```bash
   cd path/to/project
   ```

2. **Run the Script**:
   Execute the script using Python:
   ```bash
   python train_simple_lm_v1.py
   ```

3. **Expected Output**:
   - The script will:
     - Print the device used (e.g., `Using device: cuda` or `Using device: cpu`).
     - Display a model architecture summary, including layers, parameter counts, and input/output shapes.
     - Print training loss, accuracy, and validation loss every 100 steps.
     - Save a plot named `training_metrics_v1.png` with training loss, accuracy, and validation loss curves.
     - Save the trained model and tokenizer in a `saved_model` directory.
     - Stop early if validation loss doesn't improve for 3 epochs.

## Code Explanation
The `train_simple_lm_v1.py` script includes the following components:

### 1. Data Loading and Tokenization
- **Function**: `load_data(file_path)`
  - Reads the text from `input.txt`.
- **Function**: `create_tokenizer(text)`
  - Creates a character-level tokenizer using Hugging Face's `tokenizers` library.
  - Saves the tokenizer configuration for reuse.
- **Function**: `prepare_data(text, tokenizer, seq_length)`
  - Tokenizes the text and creates input-target sequences for training and validation (80/20 split).
- **Function**: `get_batches(data, batch_size)`
  - Generates batches of input-target pairs for efficient training.

### 2. Model Architecture
- **Class**: `SimpleGPT(nn.Module)`
  - Defines a GPT-like Transformer model with:
    - **Embedding Layer**: Maps tokens to dense vectors (size: 128).
    - **Positional Encoding**: Adds position information to token embeddings.
    - **Transformer Decoder**: Processes sequences with 4 layers, 4 attention heads, and 256 hidden units.
    - **Linear Layer**: Outputs probabilities for the next token (vocab size).
  - The model predicts the next character in a sequence.
- **Function**: `print_model_summary(model, vocab_size, seq_length)`
  - Prints a custom summary of the model, including layer details, parameter counts, and input/output shapes.

### 3. Training
- **Function**: `train(model, tokenizer, train_data, val_data, epochs, batch_size, lr)`
  - Trains the model using the Adam optimizer with learning rate scheduling (linear warmup and decay).
  - Applies gradient clipping to prevent exploding gradients.
  - Implements early stopping based on validation loss (patience: 3 epochs).
  - Tracks loss and accuracy (percentage of correctly predicted characters).
  - Prints progress every 100 steps.
- **GPU Support**: Uses CUDA if available for efficient training.

### 4. Visualization and Saving
- **Function**: `plot_metrics(losses, accuracies, val_losses)`
  - Generates a plot with three subplots: training loss, training accuracy, and validation loss.
  - Saves the plot as `training_metrics_v1.png`.
- **Model and Tokenizer Saving**: Saves the trained model (`model.pt`) and tokenizer (`tokenizer.json`) to a `saved_model` directory when validation loss improves.

## Example Output
### Model Summary
For a vocabulary size of ~65 characters, the model summary might look like:
```
Using device: cuda

Model Architecture:
------------------
SimpleGPT(
  (embedding): Embedding(65, 128)
  (pos_encoding): PositionalEncoding(...)
  (transformer): TransformerDecoder(...)
  (fc): Linear(in_features=256, out_features=65, bias=True)
)

Model Parameters:
-----------------
embedding.weight: torch.Size([65, 128]) (8320 parameters)
transformer.layers.0.self_attn.in_proj_weight: torch.Size([768, 256]) (196608 parameters)
...
Total Parameters: ~1,200,000
Input Shape: [batch_size, 100]
Output Shape: [batch_size, 100, 65]
```

### Training Progress
During training, you’ll see output like:
```
Epoch: 1/10, Step: 0/123, Train Loss: 4.2345, Train Accuracy: 2.50%, Val Loss: 4.2100
Epoch: 1/10, Step: 100/123, Train Loss: 3.1234, Train Accuracy: 28.45%, Val Loss: 3.1500
...
```

### Saved Files
- `training_metrics_v1.png`: Plot with training loss, accuracy, and validation loss.
- `saved_model/model.pt`: Trained model weights.
- `saved_model/tokenizer.json`: Tokenizer configuration.

## Notes
- **Hyperparameters**: Adjust in `train_simple_lm_v1.py` as needed:
  - `embed_size`: 128 (embedding dimension).
  - `hidden_size`: 256 (Transformer hidden units).
  - `num_layers`: 4 (Transformer layers).
  - `num_heads`: 4 (attention heads).
  - `seq_length`: 100 (sequence length).
  - `batch_size`: 32.
  - `epochs`: 10.
  - `learning_rate`: 0.001.
  - `patience`: 3 (early stopping patience).
- **Model Complexity**: The Transformer-based model captures longer dependencies compared to LSTM-based models.
- **Extending the Script**: Add text generation by loading the saved model and tokenizer (e.g., using `model.generate`).
- **Troubleshooting**:
  - **FileNotFoundError**: Ensure `input.txt` exists.
  - **GPU Issues**: Verify CUDA compatibility with PyTorch.
  - **Tokenizer Issues**: Ensure `transformers` and `tokenizers` are installed.
  - **Memory Issues**: Reduce `batch_size` or `seq_length` for large datasets.

## Contributing
If you enhance the script (e.g., add text generation, word-level tokenization), consider contributing via a pull request to the repository.