import json
from pathlib import Path
import os

import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score

RUN_ID = os.environ.get("RUN_ID", "run_004")
RUN_DIR = Path("runs") / RUN_ID

# Здесь alpha — это допустимая доля "вредных" случаев: ошибка И мы ответили.
TARGET_ALPHAS = [0.02, 0.05, 0.10]

# CRC требует bounded loss, верхняя граница B.
B = 1.0


def load_data(split_name: str):
    path = RUN_DIR / f"{split_name}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_scores(rows, lr, calibrator=None, mode="raw"):
    X = np.array([r["hs_tbg"] for r in rows], dtype=np.float32)
    y = np.array([r["y_error"] for r in rows], dtype=np.int32)

    if mode == "raw":
        return lr.predict_proba(X)[:, 1], y

    # calibrator = CalibratedClassifierCV -> сразу P(error)
    if mode in ("isotonic", "platt"):
        return calibrator.predict_proba(X)[:, 1], y

    raise ValueError("mode must be raw|isotonic|platt")


def pick_threshold_crc_joint(calib_p, calib_y, alpha):
    """
    CRC (в духе формулы (4) из paper): выбираем максимально "либеральный" порог t
    (максимизируем coverage), такой что
        n/(n+1) * Rhat(t) + B/(n+1) <= alpha,
    где Rhat(t) = mean( y_error * 1[p <= t] ).
    """
    n = len(calib_y)
    # монотонный параметр: t растёт => принимаем больше => joint-risk не убывает
    thresholds = np.sort(np.unique(np.quantile(calib_p, np.linspace(0, 1, 2001))))

    best = None
    for t in thresholds:
        accept = (calib_p <= t)
        # joint loss: ошибка и приняли
        losses = calib_y * accept.astype(np.int32)
        rhat = float(losses.mean())
        # CRC-adjusted empirical risk
        adj = (n / (n + 1)) * rhat + (B / (n + 1))

        if adj <= alpha:
            cov = float(accept.mean())
            # хотим максимум coverage => выбираем самый большой t, проходящий constraint
            if best is None or cov > best["cov"]:
                best = {"t": float(t), "cov": cov, "rhat": rhat, "adj": float(adj)}

    return best


def eval_joint(p, y, t):
    accept = (p <= t)
    cov = float(accept.mean())
    joint = float((y * accept.astype(np.int32)).mean())
    # conditional (для справки; не гарантируется CRC напрямую)
    cond = float((y[accept].mean())) if accept.sum() > 0 else float("nan")
    return {"cov": cov, "joint": joint, "cond": cond}


def run_mode(mode: str):
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

    print(f"\n=== CRC MODE: {mode} ===")
    print(f"Calib AUROC (scores): {roc_auc_score(calib_y, calib_p):.4f}")
    print("┌──────────┬───────────────────────────────┬───────────────────────────────┐")
    print("│ Alpha    │ Calib cov / joint / adj       │ Test  cov / joint / cond       │")
    print("├──────────┼───────────────────────────────┼───────────────────────────────┤")

    for a in TARGET_ALPHAS:
        chosen = pick_threshold_crc_joint(calib_p, calib_y, a)
        if chosen is None:
            print(f"│ {a:>7.3f} │ NO FEASIBLE                   │ -                             │")
            continue

        te = eval_joint(test_p, test_y, chosen["t"])
        print(
            f"│ {a:>7.3f} │ {chosen['cov']:>6.3f} / {chosen['rhat']:<6.3f} / {chosen['adj']:<6.3f} │ "
            f"{te['cov']:>6.3f} / {te['joint']:<6.3f} / {te['cond']:<6.3f} │"
        )

    print("└──────────┴───────────────────────────────┴───────────────────────────────┘")


def main():
    print("=== CONFORMAL RISK CONTROL (JOINT RISK) ===")
    print(f"RUN_ID={RUN_ID} TARGET_ALPHAS={TARGET_ALPHAS} B={B}")
    run_mode("raw")
    run_mode("isotonic")
    run_mode("platt")


if __name__ == "__main__":
    main()
