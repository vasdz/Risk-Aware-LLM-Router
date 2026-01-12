from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_DIR = "models/qwen2.5-3b-bnb-4bit"


def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",          # сам раскинет по CUDA/CPU
        torch_dtype="auto",         # подберёт подходящий dtype
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Привет! Кратко ответь, что ты умеешь."},
    ]

    # Qwen 2.5: обязательно использовать chat_template [web:201]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,      # greedy для стабильности
            temperature=1.0,
        )

    # берём только сгенерированную часть
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n=== MODEL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
