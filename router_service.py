import os
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from risk_router import RiskRouter

RUN_DIR = os.environ.get("RUN_DIR", "runs/run_004_c600_s42")
MODEL_SMALL = os.environ.get("MODEL_SMALL", "models/qwen2.5-3b-bnb-4bit")
MODE = os.environ.get("MODE", "raw")
ALPHA_SHIP = float(os.environ.get("ALPHA_SHIP", "0.02"))
ALPHA_ESC = float(os.environ.get("ALPHA_ESC", "0.05"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))

app = FastAPI()

tok = AutoTokenizer.from_pretrained(MODEL_SMALL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_SMALL, device_map="auto", dtype="auto")
model.eval()

router = RiskRouter(run_dir=RUN_DIR, alpha_ship=ALPHA_SHIP, alpha_escalate=ALPHA_ESC, mode=MODE)

class Req(BaseModel):
    question: str
    context: str
    generate_if_ship: bool = True

def build_prompt(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "Answer using ONLY the context. If unknown, say 'I don't know'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_hs_tbg(prompt: str):
    enc = tok([prompt], return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    hs_last = out.hidden_states[-1][0]
    hs_tbg = hs_last[prompt_len - 1].detach().cpu().tolist()
    return hs_tbg, enc, prompt_len

def greedy_generate(enc, prompt_len: int):
    with torch.no_grad():
        full = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    gen_ids = full[0][prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True)

@app.post("/route")
def route(req: Req):
    prompt = build_prompt(req.question, req.context)
    hs_tbg, enc, prompt_len = extract_hs_tbg(prompt)

    decision = router.decide(hs_tbg)
    out = {"decision": decision}

    if req.generate_if_ship and decision["action"] == "ship_small":
        out["answer_small"] = greedy_generate(enc, prompt_len)

    return out
