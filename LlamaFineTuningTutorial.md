# Tutorial: Fine-Tuning Llama 3 on a Windows Laptop with NVIDIA RTX 4070 (8 GB VRAM)

This tutorial guides you through fine-tuning a Llama 3 model on a Windows laptop with an NVIDIA RTX 4070 (8 GB VRAM) using Python, PyTorch, and open-source libraries. The fine-tuning process uses datasets in Excel, CSV, Word, TXT, and PDF formats, and the resulting model is optimized for local inference. The code is organized in a Git repository for easy reuse.

## Prerequisites
- **Hardware**: Windows laptop with NVIDIA RTX 4070 (8 GB VRAM).
- **Software**:
  - Windows 10/11.
  - Python 3.10 or 3.11 (3.12 may have compatibility issues with some libraries).
  - NVIDIA CUDA Toolkit 12.1 (for GPU support).
  - Git (for repository cloning).
- **Storage**: At least 20 GB free disk space for model weights, datasets, and dependencies.
- **Internet**: Required for downloading models and libraries.

## Step 1: Set Up the Environment
1. **Install NVIDIA Drivers and CUDA Toolkit**:
   - Download and install the latest NVIDIA driver for RTX 4070 from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx).
   - Install CUDA Toolkit 12.1 from [NVIDIA's CUDA downloads](https://developer.nvidia.com/cuda-12-1-0-download-archive).
   - Verify installation:
     ```bash
     nvidia-smi
     ```
     Ensure your RTX 4070 is listed with CUDA version 12.1.

2. **Install Python**:
   - Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3109/).
   - Install and add to PATH during setup.

3. **Set Up a Virtual Environment**:
   ```bash
   python -m venv finetune_env
   .\finetune_env\Scripts\activate
   ```

4. **Install Dependencies**:
   Install PyTorch with CUDA support, Hugging Face libraries, and tools for data processing:
   ```bash
   pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.44.2 peft==0.8.2 datasets==2.14.5 pandas openpyxl python-docx pypdf2
   ```

## Step 2: Prepare the Git Repository
1. **Create a Git Repository**:
   ```bash
   mkdir llama-finetune
   cd llama-finetune
   git init
   ```

2. **Repository Structure**:
   ```
   llama-finetune/
   ├── data/
   │   ├── raw/             # Place your Excel, CSV, Word, TXT, PDF files here
   │   ├── processed/       # Processed datasets
   ├── scripts/
   │   ├── data_processing.py
   │   ├── finetune.py
   │   ├── inference.py
   ├── models/
   │   ├── finetuned_model/ # Output directory for fine-tuned model
   ├── README.md
   ├── requirements.txt
   ```

3. **Create `requirements.txt`**:
   ```text
   torch==2.2.2
   transformers==4.44.2
   peft==0.8.2
   datasets==2.14.5
   pandas
   openpyxl
   python-docx
   pypdf2
   ```

4. **Initialize Git and Push to Remote** (e.g., GitHub):
   ```bash
   git add .
   git commit -m "Initial commit for Llama fine-tuning repo"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

## Step 3: Prepare the Dataset
Place your Excel, CSV, Word, TXT, and PDF files in the `data/raw/` directory. The `data_processing.py` script will extract text and format it for fine-tuning.

<xaiArtifact artifact_id="9962f0e8-fb63-4214-af71-cb233e0ff6dc" artifact_version_id="3295c883-78eb-4789-96de-41a8b7539bcc" title="data_processing.py" contentType="text/python">
import os
import pandas as pd
from docx import Document
import PyPDF2
from datasets import Dataset

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def process_files(input_dir, output_dir):
    texts = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            text = " ".join(df.astype(str).values.flatten())
            texts.append(text)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            text = " ".join(df.astype(str).values.flatten())
            texts.append(text)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
            texts.append(text)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            texts.append(text)
        elif filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            texts.append(text)
    
    # Create a dataset with a simple instruction-response format
    dataset = [{"instruction": "Process the following text:", "response": text} for text in texts]
    
    # Save as Hugging Face Dataset
    os.makedirs(output_dir, exist_ok=True)
    hf_dataset = Dataset.from_list(dataset)
    hf_dataset.save_to_disk(os.path.join(output_dir, "processed_dataset"))

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed"
    process_files(input_dir, output_dir)