import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "models/qwen2.5-3b-bnb-4bit"
MAX_NEW_TOKENS = 64

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
    hs = out.hidden_states[-1][0]          # [seq, dim] [web:867]
    hs_tbg = hs[prompt_len - 1].detach().cpu().tolist()
    return hs_tbg, enc, prompt_len

def greedy_generate(model, tokenizer, enc, prompt_len: int):
    with torch.no_grad():
        full = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    gen_ids = full[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype="auto")
    model.eval()

    prompt = build_prompt(tok, args.question, args.context)
    hs_tbg, enc, prompt_len = extract_hs_tbg(model, tok, prompt)
    greedy = greedy_generate(model, tok, enc, prompt_len)

    rec = {"question": args.question, "context": args.context, "hs_tbg": hs_tbg, "greedy": greedy}
    if args.out:
        Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote:", args.out)
    else:
        print(json.dumps(rec, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
