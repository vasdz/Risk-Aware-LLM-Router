import os, json
from pathlib import Path
import numpy as np
from joblib import load

RUN_ID = os.environ.get("RUN_ID", "run_004")
RUN_DIR = Path("runs") / RUN_ID

ALPHAS = [0.02, 0.05, 0.10]
MODE = os.environ.get("MODE", "platt")  # raw|platt|isotonic
B = 1.0

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def scores(rows, lr, cal, mode):
    X = np.array([r["hs_tbg"] for r in rows], dtype=np.float32)
    y = np.array([r["y_error"] for r in rows], dtype=np.int32)
    if mode == "raw":
        return lr.predict_proba(X)[:, 1], y
    return cal.predict_proba(X)[:, 1], y

def pick_threshold_crc_joint(calib_p, calib_y, alpha, B=1.0):
    n = len(calib_y)
    thresholds = np.sort(np.unique(np.quantile(calib_p, np.linspace(0, 1, 2001))))
    best = None
    for t in thresholds:
        accept = (calib_p <= t)
        rhat = float((calib_y * accept.astype(np.int32)).mean())
        adj = (n / (n + 1)) * rhat + (B / (n + 1))
        if adj <= alpha:
            cov = float(accept.mean())
            if best is None or cov > best["cov"]:
                best = {"t": float(t), "cov": cov, "calib_joint": rhat, "calib_adj": float(adj)}
    return best

def main():
    lr = load(RUN_DIR / "risk_probe_lr.joblib")
    cal = None
    if MODE == "platt":
        cal = load(RUN_DIR / "risk_calibrator_platt.joblib")
    elif MODE == "isotonic":
        cal = load(RUN_DIR / "risk_calibrator_isotonic.joblib")

    calib = load_jsonl(RUN_DIR / "calib.jsonl")
    p, y = scores(calib, lr, cal, MODE)

    out = {"run_id": RUN_ID, "mode": MODE, "B": B, "alphas": {}}
    for a in ALPHAS:
        chosen = pick_threshold_crc_joint(p, y, a, B=B)
        if chosen is None:
            out["alphas"][str(a)] = None
        else:
            out["alphas"][str(a)] = chosen

    out_path = RUN_DIR / f"gate_thresholds_{MODE}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
