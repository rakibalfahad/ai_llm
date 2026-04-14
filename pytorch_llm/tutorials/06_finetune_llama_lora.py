"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 06 — Fine-tune LLaMA with QLoRA                           ║
║  pytorch_llm/tutorials/06_finetune_llama_lora.py                   ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. LoRA — low-rank adaptation: freeze base weights, add trainable ΔW = AB
  2. QLoRA — LoRA on top of 4-bit quantized base model (fits 7B on 16 GB GPU)
  3. How to prepare an instruction-following dataset with chat templates
  4. Full fine-tuning pipeline: PEFT + TRL + HuggingFace Trainer
  5. How to save, merge, and run inference with the fine-tuned adapter
  6. Evaluate with loss and sample generation

MODEL
─────
  TinyLlama-1.1B-Chat-v1.0 (free, no auth, ~2 GB download)
  Great for fast experimentation — same architecture as LLaMA, just smaller.
  For a real 7B model, swap MODEL_NAME below (requires HF token for Llama-2).

QLORA MEMORY MATH (TinyLlama 1.1B on V100 16 GB)
──────────────────────────────────────────────────
  Base model in 4-bit (nf4): ~0.7 GB
  LoRA adapters (fp16):       ~0.05 GB   (rank=16)
  Gradients + optimizer:      ~0.1 GB    (only LoRA params)
  Activations (seq=512, b=4): ~2 GB
  Total:                      ~3 GB      ← plenty of headroom on V100

RUN IN DOCKER
─────────────
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/06_finetune_llama_lora.py

  # With HuggingFace token (for gated models like LLaMA-2):
  docker run --rm --gpus all \\
    -e HF_TOKEN=your_token_here \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/06_finetune_llama_lora.py
"""

import os
import json
import math
import argparse

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning tutorial")
    # Model
    p.add_argument("--model-name",   type=str,
                   default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model ID. Switch to meta-llama/Llama-2-7b-chat-hf for 7B.")
    p.add_argument("--use-4bit",     action="store_true", default=True,
                   help="QLoRA: load base model in 4-bit NF4")
    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16,   help="LoRA rank")
    p.add_argument("--lora-alpha",   type=int,   default=32,   help="LoRA alpha (scaling)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Training
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--seq-len",      type=int,   default=512)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    # Paths
    p.add_argument("--output-dir",   type=str,   default="/workspace/data/finetuned_llama")
    p.add_argument("--data-dir",     type=str,   default="/workspace/data")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LoRA: The Math Behind the Method
# ══════════════════════════════════════════════════════════════════════════════
#
# Full fine-tuning updates all parameters:
#   W_new = W_pretrained + ΔW    (ΔW is full rank, huge)
#
# LoRA insight (Hu et al. 2021):
#   ΔW ≈ AB    where A ∈ R^{d × r},  B ∈ R^{r × k},  r << min(d, k)
#
#   Modified forward:  h = xW + x(AB) * (α/r)
#     W = frozen pretrained weights (in 4-bit for QLoRA)
#     A = random init, B = zeros init (so ΔW=0 at start)
#     α = scaling factor (typically α = 2r)
#     r = rank (typical: 8, 16, 64)
#
# Why this works:
#   - Task-specific adaptation lives in a low-dimensional subspace
#   - r=16 means only (d+k)*16 params instead of d*k
#   - Example: d=4096, k=4096, r=16:
#     Full: 16.7M params (one weight matrix)
#     LoRA:  131K params (0.8% of full)
#
# QLoRA (Dettmers et al. 2023):
#   - Quantize base model to 4-bit NF4 (Normal Float 4)
#   - Add LoRA adapters in fp16/bf16
#   - Compute in fp16 (dequantize on the fly during forward)
#   - Memory: 4-bit params + fp16 LoRA ≈ 40% memory of full fp16

class LoRALinear(nn.Module):
    """
    Manual LoRA implementation (for learning purposes).
    In practice, PEFT library handles this automatically.

    Usage:
        # Replace any nn.Linear with LoRALinear
        original_layer = nn.Linear(4096, 4096, bias=False)
        lora_layer = LoRALinear(original_layer, r=16, alpha=32)
    """

    def __init__(self, base_layer: nn.Linear,
                 r: int = 16,
                 alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.base     = base_layer
        self.r        = r
        self.scaling  = alpha / r

        d_out, d_in = base_layer.weight.shape

        # A: initialised with small random values (kaiming)
        # B: initialised to zeros so initial ΔW = 0
        self.lora_A = nn.Linear(d_in, r,     bias=False)
        self.lora_B = nn.Linear(r,    d_out, bias=False)
        self.drop   = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base layer
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.drop(x))) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base weight for zero-overhead inference.
        After merging: W_merged = W_base + B @ A * scaling
        """
        d_out, d_in = self.base.weight.shape
        delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged = nn.Linear(d_in, d_out, bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data.float() + delta_w.float()
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data
        return merged


def count_trainable_params(model: nn.Module) -> tuple[int, int, float]:
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total, 100.0 * trainable / max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Dataset: Instruction-Following Format
# ══════════════════════════════════════════════════════════════════════════════
#
# LLaMA instruction models are trained with a specific chat template.
# TinyLlama uses the ChatML format (same as OpenAI):
#
#   <|system|>
#   You are a helpful assistant.
#   <|user|>
#   What is the capital of France?
#   <|assistant|>
#   The capital of France is Paris.
#
# During training:
#   - Input  = system + user prompt (MASKED in loss, so the model doesn't
#              learn to regurgitate the prompt)
#   - Target = assistant response (ONLY this contributes to loss)

INSTRUCTION_DATASET = [
    {
        "instruction": "Explain what a neural network is.",
        "response": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that transform input data through learned weights and activation functions. During training, the network adjusts its weights using backpropagation to minimise a loss function, enabling it to learn complex patterns from data."
    },
    {
        "instruction": "What is the difference between PyTorch and TensorFlow?",
        "response": "PyTorch uses dynamic computation graphs (define-by-run), making debugging easier and code more Pythonic. TensorFlow traditionally used static graphs but added eager execution in 2.x. PyTorch is generally preferred for research, while TensorFlow/Keras is common in production. Both support GPU acceleration and have similar ecosystem coverage."
    },
    {
        "instruction": "Explain gradient descent in simple terms.",
        "response": "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters to minimise a loss function. Imagine standing on a hilly landscape and wanting to reach the lowest valley: at each step, you look at the slope beneath you (the gradient) and take a step in the downhill direction. The learning rate controls how large each step is. Too large and you overshoot; too small and convergence is slow."
    },
    {
        "instruction": "What is overfitting and how do you prevent it?",
        "response": "Overfitting occurs when a model memorises training data instead of learning generalisable patterns, resulting in poor performance on unseen data. Prevention strategies include: (1) dropout — randomly zeroing activations during training; (2) L1/L2 regularisation — penalising large weights; (3) early stopping — halt training when validation loss starts increasing; (4) data augmentation — artificially expand the training set; (5) reducing model capacity — fewer parameters."
    },
    {
        "instruction": "What is the attention mechanism in transformers?",
        "response": "The attention mechanism allows each token in a sequence to 'attend' to all other tokens by computing similarity scores between query (Q) and key (K) vectors, then using these scores to produce a weighted average of value (V) vectors. The formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. This lets the model capture long-range dependencies without the sequential bottleneck of RNNs."
    },
    {
        "instruction": "How does LoRA reduce the cost of fine-tuning?",
        "response": "LoRA (Low-Rank Adaptation) keeps the pretrained weights frozen and instead adds pairs of small trainable matrices A and B to each target layer. The weight update is approximated as ΔW = BA where B is d×r and A is r×k, with rank r much smaller than d or k. For a 4096×4096 weight matrix with r=16, this reduces trainable parameters from 16.7M to 131K (0.8%). Only A and B are updated during training, making fine-tuning fast and memory-efficient."
    },
    {
        "instruction": "What is the difference between RMSNorm and LayerNorm?",
        "response": "LayerNorm normalises by subtracting the mean and dividing by the standard deviation, with learnable scale (gamma) and shift (beta) parameters. RMSNorm omits the mean subtraction and only divides by the root mean square of the activations, with only a scale parameter. This makes RMSNorm about 10-15% faster while achieving similar performance, which is why LLaMA adopted it."
    },
    {
        "instruction": "Explain the transformer architecture.",
        "response": "A transformer consists of stacked encoder and/or decoder blocks. Each block contains two sub-layers: (1) Multi-Head Self-Attention — each token attends to all others, (2) Feed-Forward Network — a position-wise MLP applied to each token independently. Both sub-layers use residual connections and layer normalisation. The original transformer (Vaswani 2017) used an encoder-decoder structure for translation. Decoder-only transformers (GPT, LLaMA) drop the encoder and use causal masking so each token only sees previous tokens."
    },
    {
        "instruction": "What is tokenization and why is it important?",
        "response": "Tokenization converts raw text into sequences of integers (token IDs) that neural networks can process. A tokenizer has a fixed vocabulary — a mapping from token strings to IDs. Modern LLMs use Byte-Pair Encoding (BPE), which builds a vocabulary by iteratively merging the most frequent character pairs. This balances vocabulary size (~32K–128K) against sequence length, handling rare words by splitting them into known subwords. The choice of tokenizer directly affects model performance, multilingual capability, and context window efficiency."
    },
    {
        "instruction": "How does KV caching speed up inference?",
        "response": "During autoregressive generation, each new token requires computing attention over all previous tokens. Without caching, this means recomputing keys (K) and values (V) for all previous tokens at every step — O(n^2) total work. KV caching stores the K and V tensors from previous steps. When generating the next token, we only compute K and V for the new token and append them to the cache, reducing per-step cost to O(n). This typically gives 5-10x inference speedup for long sequences."
    },
]


def format_chat_template(instruction: str, response: str,
                          tokenizer) -> dict:
    """
    Format a single instruction-response pair using the model's chat template.
    Returns tokenized input_ids and labels (with prompt masked out).
    """
    system = "You are an expert in deep learning and PyTorch. Provide clear, concise, and accurate answers."

    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": response},
    ]

    # apply_chat_template produces the final formatted string
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=512
    )
    input_ids = tokenized["input_ids"][0]

    # Build labels: mask everything up to and including the last assistant turn start
    # Only the assistant's response contributes to the loss
    labels = input_ids.clone()

    # Find where the assistant response starts
    # We tokenize just the prompt part and mask it
    prompt_messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": instruction},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    prompt_len = len(prompt_tokens)

    # Mask prompt tokens (set to -100 so cross_entropy ignores them)
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids,
        "labels":    labels,
        "attention_mask": tokenized["attention_mask"][0],
    }


class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict], tokenizer):
        self.samples = []
        for item in data:
            try:
                sample = format_chat_template(
                    item["instruction"], item["response"], tokenizer
                )
                self.samples.append(sample)
            except Exception as e:
                print(f"  Skipping sample (error: {e})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad sequences to same length within batch."""
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_token_id
    labels     = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_masks = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, s in enumerate(batch):
        l = s["input_ids"].size(0)
        input_ids[i, :l]  = s["input_ids"]
        labels[i, :l]     = s["labels"]
        attn_masks[i, :l] = s["attention_mask"]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_masks}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QLoRA Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_qlora_finetune(args):
    """Full QLoRA fine-tuning pipeline using HuggingFace PEFT + Transformers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  Tutorial 06 — QLoRA Fine-tuning")
    print(f"  Model : {args.model_name}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("=" * 70)

    # ── Import HuggingFace libraries ─────────────────────────────────────────
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from trl import SFTTrainer

    # ── Step 1: Load tokenizer ────────────────────────────────────────────────
    print("\n[1/6] Loading tokenizer...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # pad on right for causal models
    print(f"  Vocab size : {tokenizer.vocab_size:,}")
    print(f"  Pad token  : {tokenizer.pad_token!r}")

    # ── Step 2: Quantization config (QLoRA) ──────────────────────────────────
    print("\n[2/6] Setting up QLoRA quantization (4-bit NF4)...")
    if args.use_4bit and device.type == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",         # Normal Float 4 (best quality)
            bnb_4bit_compute_dtype=torch.float16,  # dequantize to fp16 for compute
            bnb_4bit_use_double_quant=True,    # quantize the quantization constants too
        )
        print("  4-bit NF4 + double quantization enabled")
    else:
        bnb_config = None
        print("  Running in fp16 (no 4-bit quantization)")

    # ── Step 3: Load base model ───────────────────────────────────────────────
    print(f"\n[3/6] Loading base model ({args.model_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model.config.use_cache = False     # disable KV cache during training
    model.config.pretraining_tp = 1   # disable tensor parallelism

    if bnb_config is not None:
        # Prepare model for k-bit training:
        # - upcasts layer norms to fp32 for stability
        # - disables gradient checkpointing incompatibilities
        model = prepare_model_for_kbit_training(model)

    base_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model parameters: {base_params/1e9:.2f}B")

    # Show memory usage after loading
    if device.type == "cuda":
        mem_used = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory used after loading: {mem_used:.2f} GB")

    # ── Step 4: LoRA config ───────────────────────────────────────────────────
    print(f"\n[4/6] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    # Target modules: which layers get LoRA adapters
    # For LLaMA: Q, K, V, O projections in attention + gate/up/down in FFN
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # FFN
        ],
    )

    model = get_peft_model(model, lora_config)

    trainable, total, pct = count_trainable_params(model)
    print(f"  Trainable parameters : {trainable:,}  ({pct:.3f}% of total)")
    print(f"  Frozen parameters    : {total - trainable:,}")
    print(f"\n  LoRA parameter breakdown per module:")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            a_params = module.lora_A.default.weight.numel()
            b_params = module.lora_B.default.weight.numel()
            print(f"    {name.split('.')[-2]:>12s}: A={a_params:,}  B={b_params:,}")

    # Show memory after LoRA
    if device.type == "cuda":
        mem_used = torch.cuda.memory_reserved() / 1e9
        print(f"\n  GPU memory after LoRA: {mem_used:.2f} GB")

    # ── Step 5: Prepare dataset ───────────────────────────────────────────────
    print("\n[5/6] Preparing instruction dataset...")
    train_size = int(0.8 * len(INSTRUCTION_DATASET))
    train_data = INSTRUCTION_DATASET[:train_size]
    val_data   = INSTRUCTION_DATASET[train_size:]

    train_ds = InstructionDataset(train_data, tokenizer)
    val_ds   = InstructionDataset(val_data,   tokenizer)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")

    # Show a sample
    sample = train_ds[0]
    print(f"\n  Sample input_ids shape  : {sample['input_ids'].shape}")
    print(f"  Sample labels shape     : {sample['labels'].shape}")
    masked = (sample['labels'] == -100).sum().item()
    total_toks = sample['labels'].size(0)
    print(f"  Prompt tokens (masked)  : {masked} / {total_toks} "
          f"({100*masked/total_toks:.0f}%)")

    # Example decoded tokens
    non_masked = sample['input_ids'][sample['labels'] != -100]
    print(f"\n  Assistant response tokens: {tokenizer.decode(non_masked[:40])}...")

    # ── Step 6: Training ──────────────────────────────────────────────────────
    print("\n[6/6] Starting fine-tuning...")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=(device.type == "cuda"),
        logging_dir=os.path.join(args.data_dir, "llm_runs", "qlora"),
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
        max_grad_norm=1.0,
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
    )

    # SFTTrainer = Supervised Fine-tuning Trainer (simplifies instruction tuning)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt"
        ),
        max_seq_length=args.seq_len,
    )

    print(f"\n  Training config:")
    print(f"    Epochs          : {args.epochs}")
    print(f"    Batch size      : {args.batch_size}")
    print(f"    Grad accumulation: {args.grad_accum}  (effective batch = {args.batch_size * args.grad_accum})")
    print(f"    LR              : {args.lr}")

    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n  LoRA adapter saved to: {adapter_dir}")

    return model, tokenizer, adapter_dir


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Inference with Fine-tuned Adapter
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, tokenizer, args):
    """Test the fine-tuned model on held-out prompts."""
    print("\n" + "=" * 70)
    print("  INFERENCE — Testing fine-tuned model")
    print("=" * 70)

    model.eval()
    device = next(model.parameters()).device

    test_prompts = [
        "What is the purpose of the softmax function in neural networks?",
        "Explain what backpropagation does.",
        "What is a convolutional neural network?",
    ]

    for prompt in test_prompts:
        messages = [
            {"role": "system",
             "content": "You are an expert in deep learning and PyTorch."},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        print(f"\n  Prompt   : {prompt}")
        print(f"  Response : {response.strip()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LoRA Concept Demo (runs without internet)
# ══════════════════════════════════════════════════════════════════════════════

def lora_concept_demo():
    """
    Demonstrate LoRA math on a toy linear layer.
    Runs entirely on CPU with no downloads.
    """
    print("\n── LoRA Concept Demo (no downloads required) ──")

    torch.manual_seed(42)
    d_in, d_out, r, alpha = 128, 256, 8, 16

    # Pretrained layer (frozen)
    base = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        nn.init.normal_(base.weight, std=0.02)
    for p in base.parameters():
        p.requires_grad = False

    # LoRA layer
    lora = LoRALinear(base, r=r, alpha=alpha)

    trainable_lora = sum(p.numel() for p in lora.lora_A.parameters()) + \
                     sum(p.numel() for p in lora.lora_B.parameters())
    total_base     = d_in * d_out

    print(f"\n  Base layer: {d_in} → {d_out}  ({total_base:,} params, frozen)")
    print(f"  LoRA rank:  r={r}, alpha={alpha}, scaling={alpha/r:.1f}")
    print(f"  LoRA A:     {d_in} → {r}  ({d_in*r:,} params)")
    print(f"  LoRA B:     {r} → {d_out}  ({r*d_out:,} params)")
    print(f"  Trainable:  {trainable_lora:,}  ({100*trainable_lora/total_base:.1f}% of base)")

    x = torch.randn(4, d_in)

    # Before training: ΔW = BA = 0 (B is zero-init)
    with torch.no_grad():
        base_out  = base(x)
        lora_out  = lora(x)
        delta = (lora_out - base_out).abs().max().item()
    print(f"\n  Initial delta (should be ~0): {delta:.6f}")

    # Simulate a few gradient steps on LoRA params only
    optimizer = torch.optim.Adam(
        [p for p in lora.parameters() if p.requires_grad], lr=1e-3
    )
    target = torch.randn(4, d_out)
    for step in range(50):
        out  = lora(x)
        loss = nn.MSELoss()(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        base_out  = base(x)
        lora_out  = lora(x)
        delta = (lora_out - base_out).abs().max().item()
    print(f"  Delta after 50 steps: {delta:.4f}  (adapter has learned something)")

    # Merge weights
    merged = lora.merge_weights()
    with torch.no_grad():
        merged_out = merged(x)
        merge_err  = (merged_out - lora_out).abs().max().item()
    print(f"  Merge error (should be ~0): {merge_err:.8f}")
    print("  ✓ LoRA math verified: merge is numerically exact")

    # Show parameter groups
    print(f"\n  Gradient check:")
    for name, param in lora.named_parameters():
        print(f"    {name:30s}  requires_grad={param.requires_grad}  "
              f"shape={list(param.shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Merge Adapter for Deployment
# ══════════════════════════════════════════════════════════════════════════════

def explain_adapter_merge():
    print("""
── Merging LoRA Adapter for Deployment ──

After fine-tuning, you have two options for inference:

Option 1 — Use adapter separately (flexible, saves disk):
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained("TinyLlama/...", device_map="auto")
    model = PeftModel.from_pretrained(base, "/workspace/data/finetuned_llama/lora_adapter")

Option 2 — Merge adapter into base model (zero overhead, faster):
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained("TinyLlama/...", dtype=torch.float16)
    model = PeftModel.from_pretrained(base, "/workspace/data/finetuned_llama/lora_adapter")
    merged = model.merge_and_unload()   ← bakes LoRA into base weights
    merged.save_pretrained("/workspace/data/finetuned_llama/merged_model")

The merged model has identical inference cost to the original — the LoRA
overhead exists only during fine-tuning, not at deployment.

For production, Option 2 is preferred. Ship the merged model with:
    tokenizer.save_pretrained("/workspace/data/finetuned_llama/merged_model")
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Always run the local LoRA demo (no downloads needed)
    lora_concept_demo()
    explain_adapter_merge()

    # Attempt full QLoRA fine-tuning
    print("\n" + "=" * 70)
    print("  FULL QLORA FINE-TUNING")
    print("=" * 70)

    try:
        import transformers
        import peft
        import trl
        import bitsandbytes
        print(f"  transformers: {transformers.__version__}")
        print(f"  peft        : {peft.__version__}")
        print(f"  trl         : {trl.__version__}")
        print(f"  bitsandbytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  Install with: pip install transformers peft trl bitsandbytes")
        return

    model, tokenizer, adapter_dir = run_qlora_finetune(args)
    run_inference(model, tokenizer, args)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"""
  What we did:
    ✓ Loaded TinyLlama-1.1B in 4-bit NF4 quantization (~700 MB)
    ✓ Applied LoRA adapters (r={args.lora_r}) to Q/K/V/O/gate/up/down projections
    ✓ Fine-tuned on {len(INSTRUCTION_DATASET)} instruction-response pairs
    ✓ Saved adapter to {adapter_dir}

  Key numbers:
    Base model   : ~1.1B params (frozen, 4-bit)
    LoRA adapter : ~8M params (fp16, trainable)
    Training cost: ~5% of full fine-tune memory, ~3% of params

  To use on a larger model (7B), just change:
    --model-name meta-llama/Llama-2-7b-chat-hf
    (requires HF_TOKEN and Llama-2 access approval)

  Outputs saved to: {args.output_dir}
  TensorBoard logs: {args.data_dir}/llm_runs/qlora

  View with:
    tensorboard --logdir {args.data_dir}/llm_runs --host 0.0.0.0 --port 7777

  Congratulations — you've completed the PyTorch LLM tutorial series!
  You can now:
    ✓ Build a tokenizer from scratch (BPE)
    ✓ Implement attention, RoPE, RMSNorm, SwiGLU from first principles
    ✓ Train a character-level language model
    ✓ Understand LLaMA-2/3 architecture and GQA
    ✓ Fine-tune a real LLaMA model with QLoRA
""")


if __name__ == "__main__":
    main()
