import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="run_004")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_calib", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    train_p = run_dir / "train.jsonl"
    calib_p = run_dir / "calib.jsonl"
    test_p = run_dir / "test.jsonl"

    for p in (train_p, calib_p, test_p):
        assert p.exists(), f"Missing file: {p}"

    rows = read_jsonl(train_p) + read_jsonl(calib_p) + read_jsonl(test_p)
    n = len(rows)
    need = args.n_train + args.n_calib + args.n_test
    assert n == need, f"Expected total {need}, got {n}"

    if args.backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = run_dir.parent / f"{args.run_id}__bak_{ts}"
        bak.mkdir(parents=True, exist_ok=False)
        shutil.copy2(train_p, bak / "train.jsonl")
        shutil.copy2(calib_p, bak / "calib.jsonl")
        shutil.copy2(test_p, bak / "test.jsonl")
        # на всякий случай сохраняем артефакты модели/калибраторов, если есть
        for fn in ["risk_probe_lr.joblib", "risk_calibrator_isotonic.joblib", "risk_calibrator_platt.joblib"]:
            src = run_dir / fn
            if src.exists():
                shutil.copy2(src, bak / fn)
        print(f"[resplit] Backup saved to: {bak}")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    rows = [rows[i] for i in idx]

    train = rows[: args.n_train]
    calib = rows[args.n_train : args.n_train + args.n_calib]
    test = rows[args.n_train + args.n_calib :]

    write_jsonl(train_p, train)
    write_jsonl(calib_p, calib)
    write_jsonl(test_p, test)

    # чтобы не было “несостыковки” (модели обучены на старом сплите)
    for fn in ["risk_probe_lr.joblib", "risk_calibrator_isotonic.joblib", "risk_calibrator_platt.joblib"]:
        p = run_dir / fn
        if p.exists():
            p.unlink()
    print(f"[resplit] Done. Rewrote train/calib/test in {run_dir} with seed={args.seed}")
    print("[resplit] Deleted old *.joblib; rerun 03_train_probe.py next.")


if __name__ == "__main__":
    main()
