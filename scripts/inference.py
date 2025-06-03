import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # Load fine-tuned model and tokenizer
    model_path = "models/finetuned_model"
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)

    # Inference loop
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=200)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()