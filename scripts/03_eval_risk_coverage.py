import json
from pathlib import Path

import numpy as np
from joblib import load

RUN_ID = "run_002"
RUN_DIR = Path("runs") / RUN_ID

TARGET_RISK = 0.05  # хотим <= 5% ошибок среди отвеченных


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_scores(rows, lr, iso):
    X = np.array([r["hs_tbg"] for r in rows], dtype=np.float32)
    y = np.array([r["y_error"] for r in rows], dtype=np.int32)
    p_raw = lr.predict_proba(X)[:, 1]
    p = iso.transform(p_raw)  # calibrated P(error)
    return p, y


def pick_threshold(calib_p, calib_y, target_risk):
    # перебираем пороги по квантилям скоринга
    # принимаем ответ если p <= thr
    cands = np.quantile(calib_p, np.linspace(0.0, 1.0, 101))
    best = None
    for thr in cands:
        mask = calib_p <= thr
        if mask.sum() == 0:
            continue
        risk = calib_y[mask].mean()
        coverage = mask.mean()
        if risk <= target_risk:
            # хотим максимум coverage при выполнении risk constraint
            if best is None or coverage > best["coverage"]:
                best = {"thr": float(thr), "risk": float(risk), "coverage": float(coverage)}
    return best


def eval_at_threshold(p, y, thr):
    mask = p <= thr
    if mask.sum() == 0:
        return {"coverage": 0.0, "risk": float("nan"), "answered": 0}
    return {
        "coverage": float(mask.mean()),
        "risk": float(y[mask].mean()),
        "answered": int(mask.sum()),
    }


def main():
    lr = load(RUN_DIR / "risk_probe_lr.joblib")
    iso = load(RUN_DIR / "risk_calibrator_isotonic.joblib")

    calib = load_jsonl(RUN_DIR / "calib.jsonl")
    test = load_jsonl(RUN_DIR / "test.jsonl")

    calib_p, calib_y = compute_scores(calib, lr, iso)
    test_p, test_y = compute_scores(test, lr, iso)

    chosen = pick_threshold(calib_p, calib_y, TARGET_RISK)
    if chosen is None:
        print("Could not find threshold that satisfies target risk on calibration set.")
        return

    print(f"[calib] chosen thr={chosen['thr']:.4f} risk={chosen['risk']:.4f} coverage={chosen['coverage']:.4f}")

    test_res = eval_at_threshold(test_p, test_y, chosen["thr"])
    print(f"[test]  coverage={test_res['coverage']:.4f} risk={test_res['risk']:.4f} answered={test_res['answered']}")

    # Дополнительно: кривая risk-coverage на test
    print("\nRisk-Coverage curve (test):")
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        thr = float(np.quantile(test_p, q))
        res = eval_at_threshold(test_p, test_y, thr)
        print(f"  q={q:.1f} thr={thr:.4f} coverage={res['coverage']:.4f} risk={res['risk']:.4f}")


if __name__ == "__main__":
    main()
