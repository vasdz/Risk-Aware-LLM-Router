import json
import numpy as np
from pathlib import Path
from collections import Counter

RUN_ID = "run_003"
RUN_DIR = Path("runs") / RUN_ID

TARGET_RISKS = [0.05, 0.10, 0.15]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\n", " ").split()).strip(" .,!?")


def self_consistency_uncertainty(samples):
    # uncertainty = 1 - max_cluster_freq/K  (чем больше разнобой, тем выше)
    ss = [norm(x) for x in (samples or []) if norm(x)]
    if not ss:
        return 1.0
    c = Counter(ss)
    max_freq = max(c.values())
    return 1.0 - (max_freq / len(ss))


def pick_threshold(calib_score, calib_y, target_risk):
    # accept if score <= thr
    thresholds = np.sort(np.unique(np.quantile(calib_score, np.linspace(0, 1, 201))))
    best = None
    for thr in thresholds:
        m = calib_score <= thr
        if m.sum() == 0:
            continue
        risk = float(calib_y[m].mean())
        cov = float(m.mean())
        if risk <= target_risk and (best is None or cov > best["cov"]):
            best = {"thr": float(thr), "risk": risk, "cov": cov}
    return best


def eval_thr(score, y, thr):
    m = score <= thr
    if m.sum() == 0:
        return {"cov": 0.0, "risk": float("nan")}
    return {"cov": float(m.mean()), "risk": float(y[m].mean())}


def main():
    calib = load_jsonl(RUN_DIR / "calib.jsonl")
    test = load_jsonl(RUN_DIR / "test.jsonl")

    calib_score = np.array([self_consistency_uncertainty(r["samples"]) for r in calib], dtype=np.float32)
    calib_y = np.array([r["y_error"] for r in calib], dtype=np.int32)

    test_score = np.array([self_consistency_uncertainty(r["samples"]) for r in test], dtype=np.float32)
    test_y = np.array([r["y_error"] for r in test], dtype=np.int32)

    print("=== SELF-CONSISTENCY BASELINE ===")
    print("accept if uncertainty <= thr\n")

    for target in TARGET_RISKS:
        chosen = pick_threshold(calib_score, calib_y, target)
        if chosen is None:
            print(f"target={target:.2f}: NO FEASIBLE")
            continue
        res = eval_thr(test_score, test_y, chosen["thr"])
        print(f"target={target:.2f} | calib_cov={chosen['cov']:.3f} calib_risk={chosen['risk']:.3f} "
              f"| test_cov={res['cov']:.3f} test_risk={res['risk']:.3f}")


if __name__ == "__main__":
    main()
