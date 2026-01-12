import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== RUN CONFIG ==========
RUN_ID = "run_004"
OUT_DIR = Path("runs") / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = "models/qwen2.5-3b-bnb-4bit"

N_TRAIN = 200
N_CALIB = 400
N_TEST = 400

# risk-only (без samples)
K_SAMPLES = 0

MAX_NEW_TOKENS = 24
MAX_CONTEXT_CHARS = 3000
PRINT_EVERY = 25


def build_prompt(tokenizer, question: str, context: str) -> str:
    context = (context or "")[:MAX_CONTEXT_CHARS]
    messages = [
        {"role": "system", "content": "Answer using ONLY the context. If unknown, say 'I don't know'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_gen(tokenizer, full_ids, prompt_len: int):
    gen_ids = full_ids[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def is_correct_squad(gold: str, pred: str) -> int:
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    return int(len(g) > 0 and g in p)


def resume_offset(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    n = 0
    with out_path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def process_split(model, tokenizer, ds, out_path: Path):
    already = resume_offset(out_path)
    total = len(ds)

    if already >= total:
        print(f"[collect] skip {out_path} (already {already}/{total})")
        return

    print(f"[collect] writing {out_path} (resume {already}/{total})")

    with out_path.open("a", encoding="utf-8") as f_out:
        for i in range(already, total):
            ex = ds[i]
            q = ex["question"]
            context = ex["context"]
            gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

            prompt = build_prompt(tokenizer, q, context)
            enc = tokenizer([prompt], return_tensors="pt").to(model.device)
            prompt_len = enc["input_ids"].shape[1]

            # TBG hidden state: один forward по prompt
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[-1][0]  # [seq, dim]
            hs_tbg = hs[prompt_len - 1].detach().cpu().tolist()

            # greedy answer
            with torch.no_grad():
                greedy_full = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            greedy_text = decode_gen(tokenizer, greedy_full, prompt_len)

            y_correct = is_correct_squad(gold, greedy_text)
            y_error = 1 - y_correct

            rec = {
                "id": ex.get("id", str(i)),
                "question": q,
                "context": (context or "")[:MAX_CONTEXT_CHARS],
                "gold": gold,
                "greedy": greedy_text,
                "samples": [],          # risk-only
                "y_correct": y_correct,
                "y_error": y_error,
                "hs_tbg": hs_tbg,
            }

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()

            if (i + 1) % PRINT_EVERY == 0:
                print(f"[collect] {out_path.name}: {i+1}/{total}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    ds = load_dataset("squad", split="train")

    train_ds = ds.select(range(0, N_TRAIN))
    calib_ds = ds.select(range(N_TRAIN, N_TRAIN + N_CALIB))
    test_ds = ds.select(range(N_TRAIN + N_CALIB, N_TRAIN + N_CALIB + N_TEST))

    process_split(model, tokenizer, train_ds, OUT_DIR / "train.jsonl")
    process_split(model, tokenizer, calib_ds, OUT_DIR / "calib.jsonl")
    process_split(model, tokenizer, test_ds, OUT_DIR / "test.jsonl")

    print("Done:", OUT_DIR)


if __name__ == "__main__":
    main()
