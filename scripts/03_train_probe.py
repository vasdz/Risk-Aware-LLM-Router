import os
import json
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

RUN_ID = os.environ.get("RUN_ID", "run_004")
RUN_DIR = Path("runs") / RUN_ID

OUT_PROBE = RUN_DIR / "risk_probe_lr.joblib"
OUT_ISO = RUN_DIR / "risk_calibrator_isotonic.joblib"  # CalibratedClassifierCV
OUT_PLATT = RUN_DIR / "risk_calibrator_platt.joblib"   # CalibratedClassifierCV

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    train = load_jsonl(RUN_DIR / "train.jsonl")
    calib = load_jsonl(RUN_DIR / "calib.jsonl")

    X_train = np.array([r["hs_tbg"] for r in train], dtype=np.float32)
    y_train = np.array([r["y_error"] for r in train], dtype=np.int32)

    X_calib = np.array([r["hs_tbg"] for r in calib], dtype=np.float32)
    y_calib = np.array([r["y_error"] for r in calib], dtype=np.int32)

    # base probe
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    p_raw = lr.predict_proba(X_calib)[:, 1]

    # calibrators trained ONLY on train via CV (leakage-free)
    iso = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000),
        method="isotonic",
        cv=5,
    )
    iso.fit(X_train, y_train)
    p_iso = iso.predict_proba(X_calib)[:, 1]

    platt = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=2000),
        method="sigmoid",
        cv=5,
    )
    platt.fit(X_train, y_train)
    p_platt = platt.predict_proba(X_calib)[:, 1]

    print(f"[calib] AUROC raw P(error): {roc_auc_score(y_calib, p_raw):.4f}")
    print(f"[calib] Brier raw:         {brier_score_loss(y_calib, p_raw):.4f}")
    print(f"[calib] Brier isotonic:    {brier_score_loss(y_calib, p_iso):.4f}")
    print(f"[calib] Brier platt:       {brier_score_loss(y_calib, p_platt):.4f}")

    dump(lr, OUT_PROBE)
    dump(iso, OUT_ISO)
    dump(platt, OUT_PLATT)

    print("Saved:", OUT_PROBE)
    print("Saved:", OUT_ISO)
    print("Saved:", OUT_PLATT)

if __name__ == "__main__":
    main()
