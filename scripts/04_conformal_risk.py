import json
from pathlib import Path
import math

import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score

RUN_ID = "run_004"
RUN_DIR = Path("runs") / RUN_ID

TARGET_RISKS = [0.05, 0.10, 0.15]
Z = 1.96  # ~95% two-sided; можно сделать настраиваемым


def load_data(split_name: str):
    path = RUN_DIR / f"{split_name}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def wilson_upper_bound(k, n, z=Z):
    # Upper bound for binomial proportion (Wilson score interval).
    if n == 0:
        return 1.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)
    return (center + margin) / denom


def compute_scores(rows, lr, calibrator=None, mode="raw"):
    X = np.array([r["hs_tbg"] for r in rows], dtype=np.float32)
    y = np.array([r["y_error"] for r in rows], dtype=np.int32)

    if mode == "raw":
        p = lr.predict_proba(X)[:, 1]
        return p, y

    # calibrator теперь sklearn CalibratedClassifierCV, он сам выдаёт P(error)
    if mode in ("isotonic", "platt"):
        p = calibrator.predict_proba(X)[:, 1]
        return p, y

    raise ValueError("mode must be raw|isotonic|platt")

def pick_threshold_ucb(calib_p, calib_y, target_risk):
    # Ищем максимальный coverage при условии UCB(risk) <= target.
    thresholds = np.sort(np.unique(np.quantile(calib_p, np.linspace(0, 1, 2001))))
    best = None

    n_total = len(calib_y)
    for thr in thresholds:
        mask = calib_p <= thr
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(calib_y[mask].sum())

        risk_ucb = float(wilson_upper_bound(k, n))
        cov = float(n / n_total)

        if risk_ucb <= target_risk and (best is None or cov > best["cov"]):
            best = {
                "thr": float(thr),
                "cov": cov,
                "risk_ucb": risk_ucb,
                "n": n,
                "k": k,
            }
    return best


def eval_at(p, y, thr):
    mask = p <= thr
    n = int(mask.sum())
    if n == 0:
        return {"cov": 0.0, "risk": float("nan"), "n": 0, "k": 0}
    k = int(y[mask].sum())
    return {
        "cov": float(n / len(y)),
        "risk": float(k / n),
        "n": n,
        "k": k,
    }


def run_table(mode: str):
    lr = load(RUN_DIR / "risk_probe_lr.joblib")
    iso = load(RUN_DIR / "risk_calibrator_isotonic.joblib")
    platt = load(RUN_DIR / "risk_calibrator_platt.joblib")

    calib = load_data("calib")
    test = load_data("test")

    if mode == "raw":
        calib_p, calib_y = compute_scores(calib, lr, None, "raw")
        test_p, test_y = compute_scores(test, lr, None, "raw")
    elif mode == "isotonic":
        calib_p, calib_y = compute_scores(calib, lr, iso, "isotonic")
        test_p, test_y = compute_scores(test, lr, iso, "isotonic")
    elif mode == "platt":
        calib_p, calib_y = compute_scores(calib, lr, platt, "platt")
        test_p, test_y = compute_scores(test, lr, platt, "platt")
    else:
        raise ValueError

    print(f"\n=== MODE: {mode} ===")
    print(f"Calib AUROC (scores): {roc_auc_score(calib_y, calib_p):.4f}")
    print("┌─────────────┬──────────────────────────┬──────────────────────────┐")
    print("│ Target risk │ Calib cov / risk_UCB     │ Test cov / risk          │")
    print("├─────────────┼──────────────────────────┼──────────────────────────┤")

    for target in TARGET_RISKS:
        chosen = pick_threshold_ucb(calib_p, calib_y, target)
        if chosen is None:
            print(f"│ {target:>10.2f} │ NO FEASIBLE              │ -                        │")
            continue

        tr = eval_at(test_p, test_y, chosen["thr"])
        print(
            f"│ {target:>10.2f} │ {chosen['cov']:>6.3f} / {chosen['risk_ucb']:<8.3f}"
            f"│ {tr['cov']:>6.3f} / {tr['risk']:<8.3f} │"
        )

    print("└─────────────┴──────────────────────────┴──────────────────────────┘")


def main():
    print("=== CONFORMAL / SELECTIVE RISK CONTROL ===")
    print(f"RUN_ID={RUN_ID} TARGET_RISKS={TARGET_RISKS} Z={Z}")
    run_table("raw")
    run_table("isotonic")
    run_table("platt")


if __name__ == "__main__":
    main()
