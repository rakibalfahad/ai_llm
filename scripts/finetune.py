import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

def main():
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_from_disk("data/processed/processed_dataset")

    # Tokenize dataset
    def tokenize_function(examples):
        inputs = [f"{ex['instruction']} {ex['response']}" for ex in examples]
        tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/finetuned_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        optim="adamw_torch",
        max_steps=-1
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("models/finetuned_model")
    tokenizer.save_pretrained("models/finetuned_model")

if __name__ == "__main__":
    main()