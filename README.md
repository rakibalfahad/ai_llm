# ai_llm

A hands-on project for AI model building, fine-tuning, and deep learning experimentation.

---

## deeplearning/

GPU-accelerated Docker environment for deep learning and LLM projects, targeting **Tesla V100-PCIE-16GB**.

### Stack

| Category | Packages |
|----------|----------|
| **Frameworks** | PyTorch 2.6 (cu124), TensorFlow 2.x, ONNX Runtime GPU |
| **Hugging Face** | transformers, datasets, accelerate, peft, trl, evaluate, tokenizers |
| **Quantization** | bitsandbytes, auto-gptq, autoawq, optimum |
| **Attention** | xformers (sm_70 memory-efficient attention) |
| **LLM / RAG** | langchain, openai, tiktoken, faiss-cpu, chromadb, llama-cpp-python |
| **Data Science** | scikit-learn, pandas, matplotlib, seaborn, plotly, scipy, opencv |
| **Tracking** | wandb, tensorboard, mlflow |
| **Serving / UI** | gradio, Jupyter Lab |

### Quick Start

```bash
# Build the Docker image
cd deeplearning/
bash create_docker.sh

# Run interactive shell
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -it deeplearning:v100-llm

# Launch Jupyter Lab
docker run --rm --gpus all \
  -p 8888:8888 \
  -v $(pwd)/data:/workspace/data \
  deeplearning:v100-llm \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Quick GPU test
docker run --rm --gpus all deeplearning:v100-llm python3 /test_gpu.py
```

### Structure

```
deeplearning/
├── Dockerfile              # GPU-accelerated image definition
├── create_docker.sh        # Build script
├── test_docker.sh          # Comprehensive GPU/library test
├── data/
│   ├── mnist/              # MNIST dataset (raw)
│   ├── mnist_model_pt/     # Saved PyTorch model
│   └── mnist_model_tf/     # Saved TensorFlow model
├── notebooks/
│   └── GPU_Test_and_Diagnostics.ipynb
└── scripts/
    ├── pt_mnist_gpu.py     # PyTorch MNIST training (GPU)
    └── tf_mnist_gpu.py     # TensorFlow MNIST training (GPU)
```

See [deeplearning/README.md](deeplearning/README.md) for full details.
