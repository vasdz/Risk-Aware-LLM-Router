import json
import math
from collections import Counter
from pathlib import Path

RUN_ID = "run_001"
IN_PATH = Path("runs") / RUN_ID / "examples.jsonl"
OUT_PATH = Path("runs") / RUN_ID / "se.csv"


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().replace("\n", " ").split()).strip(" .,!?")


def semantic_entropy(samples):
    if not samples:
        return 0.0
    norm = [normalize_text(s) for s in samples if s.strip()]
    if not norm:
        return 0.0
    counts = Counter(norm)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with IN_PATH.open("r", encoding="utf-8") as f_in, \
         OUT_PATH.open("w", encoding="utf-8") as f_out:
        f_out.write("id,se\n")
        for line in f_in:
            ex = json.loads(line)
            se = semantic_entropy(ex.get("samples", []))
            f_out.write(f"{ex['id']},{se:.6f}\n")
    print(f"Saved SE to {OUT_PATH}")


if __name__ == "__main__":
    main()
