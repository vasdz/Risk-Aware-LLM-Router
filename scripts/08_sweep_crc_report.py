import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from joblib import load


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_scores(rows, lr, calibrator, mode):
    X = np.array([r["hs_tbg"] for r in rows], dtype=np.float32)
    y = np.array([r["y_error"] for r in rows], dtype=np.int32)

    if mode == "raw":
        return lr.predict_proba(X)[:, 1], y

    # совместимость: calibrator может быть (a) CalibratedClassifierCV => predict_proba(X)
    # (b) IsotonicRegression => transform(p_raw)
    # (c) LogisticRegression(platt on p_raw) => predict_proba(p_raw.reshape(-1,1))
    p_raw = lr.predict_proba(X)[:, 1]

    if hasattr(calibrator, "transform"):  # isotonic old-style
        return calibrator.transform(p_raw), y

    if hasattr(calibrator, "predict_proba"):
        nfi = getattr(calibrator, "n_features_in_", None)
        if nfi is None:
            # безопасный fallback
            try:
                return calibrator.predict_proba(X)[:, 1], y
            except Exception:
                return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1], y
        if nfi == 1:
            return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1], y
        return calibrator.predict_proba(X)[:, 1], y

    raise ValueError("Unknown calibrator type")


def pick_threshold_crc_joint(calib_p, calib_y, alpha, B=1.0):
    n = len(calib_y)
    thresholds = np.sort(np.unique(np.quantile(calib_p, np.linspace(0, 1, 2001))))

    best = None
    for t in thresholds:
        accept = (calib_p <= t)
        losses = calib_y * accept.astype(np.int32)  # error & accepted
        rhat = float(losses.mean())
        adj = (n / (n + 1)) * rhat + (B / (n + 1))

        if adj <= alpha:
            cov = float(accept.mean())
            if best is None or cov > best["cov"]:
                best = {"t": float(t), "cov": cov, "rhat": rhat, "adj": float(adj)}
    return best


def eval_joint(p, y, t):
    accept = (p <= t)
    cov = float(accept.mean())
    joint = float((y * accept.astype(np.int32)).mean())
    cond = float(y[accept].mean()) if accept.sum() > 0 else float("nan")
    return cov, joint, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", type=str, default="run_004_s*")
    ap.add_argument("--alphas", type=str, default="0.02,0.05,0.10")
    ap.add_argument("--modes", type=str, default="raw,platt")
    ap.add_argument("--out_csv", type=str, default="runs/sweep_crc_report.csv")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    run_dirs = sorted(Path("runs").glob(args.runs_glob))
    assert run_dirs, f"No runs matched: runs/{args.runs_glob}"

    out_rows = []
    for rd in run_dirs:
        run_id = rd.name

        env = os.environ.copy()
        env["RUN_ID"] = run_id

        # train probe (+ calibrators if configured)
        subprocess.run(["python", "scripts/03_train_probe.py"], check=True, env=env)

        lr = load(rd / "risk_probe_lr.joblib")
        iso = load(rd / "risk_calibrator_isotonic.joblib") if (rd / "risk_calibrator_isotonic.joblib").exists() else None
        platt = load(rd / "risk_calibrator_platt.joblib") if (rd / "risk_calibrator_platt.joblib").exists() else None

        calib = load_jsonl(rd / "calib.jsonl")
        test = load_jsonl(rd / "test.jsonl")

        for mode in modes:
            cal = None
            if mode == "isotonic":
                cal = iso
            elif mode == "platt":
                cal = platt

            calib_p, calib_y = get_scores(calib, lr, cal, mode)
            test_p, test_y = get_scores(test, lr, cal, mode)

            for a in alphas:
                chosen = pick_threshold_crc_joint(calib_p, calib_y, a, B=1.0)
                if chosen is None:
                    out_rows.append({
                        "run_id": run_id, "mode": mode, "alpha": a,
                        "calib_cov": np.nan, "calib_joint": np.nan, "calib_adj": np.nan,
                        "test_cov": np.nan, "test_joint": np.nan, "test_cond": np.nan,
                    })
                    continue

                tcov, tjoint, tcond = eval_joint(test_p, test_y, chosen["t"])
                out_rows.append({
                    "run_id": run_id, "mode": mode, "alpha": a,
                    "calib_cov": chosen["cov"], "calib_joint": chosen["rhat"], "calib_adj": chosen["adj"],
                    "test_cov": tcov, "test_joint": tjoint, "test_cond": tcond,
                })

        print(f"[sweep] done {run_id}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # write CSV manually (без pandas)
    cols = ["run_id","mode","alpha","calib_cov","calib_joint","calib_adj","test_cov","test_joint","test_cond"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in out_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print("[sweep] wrote:", out_path)


if __name__ == "__main__":
    main()
