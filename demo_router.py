import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from risk_router import RiskRouter

MODEL_SMALL = "models/qwen2.5-3b-bnb-4bit"
RUN_DIR = "runs/run_004_c600_s42"

def build_prompt(tokenizer, question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "Answer using ONLY the context. If unknown, say 'I don't know'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_hs_tbg(model, tokenizer, prompt: str):
    enc = tokenizer([prompt], return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[-1][0]
    hs_tbg = hs[prompt_len - 1].detach().cpu().tolist()
    return hs_tbg, enc, prompt_len

def greedy_generate(model, tokenizer, enc, prompt_len: int, max_new_tokens=64):
    with torch.no_grad():
        full = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    gen_ids = full[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--alpha_ship", type=float, default=0.02)
    ap.add_argument("--alpha_escalate", type=float, default=0.05)
    ap.add_argument("--mode", type=str, default="raw")  # raw|platt|isotonic
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_SMALL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_SMALL, device_map="auto", torch_dtype="auto")
    model.eval()

    router = RiskRouter(run_dir=RUN_DIR, alpha_ship=args.alpha_ship, alpha_escalate=args.alpha_escalate, mode=args.mode)

    prompt = build_prompt(tok, args.question, args.context)
    hs_tbg, enc, prompt_len = extract_hs_tbg(model, tok, prompt)

    decision = router.decide(hs_tbg)
    print(json.dumps(decision, ensure_ascii=False, indent=2))

    if decision["action"] == "ship_small":
        ans = greedy_generate(model, tok, enc, prompt_len, max_new_tokens=args.max_new_tokens)
        print("\n=== SMALL MODEL ANSWER ===")
        print(ans)
    elif decision["action"] == "escalate_big":
        print("\n=== ESCALATE ===")
        print("Route to bigger model (not implemented here).")
    else:
        print("\n=== REFUSE ===")
        print("Refuse / human-in-the-loop.")

if __name__ == "__main__":
    main()
