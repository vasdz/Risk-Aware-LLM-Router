import argparse
import json
from pathlib import Path
import numpy as np


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_seeds(s: str):
    # "42-51" or "42,43,44"
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_run_dir", type=str, required=True, help="e.g. runs/run_004__bak_20251213_184037")
    ap.add_argument("--dst_prefix", type=str, default="run_004_s", help="e.g. run_004_s -> run_004_s42")
    ap.add_argument("--seeds", type=str, default="42-51")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_calib", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=400)
    args = ap.parse_args()

    src = Path(args.src_run_dir)
    assert src.exists(), f"Missing: {src}"

    rows = read_jsonl(src / "train.jsonl") + read_jsonl(src / "calib.jsonl") + read_jsonl(src / "test.jsonl")
    total = len(rows)
    need = args.n_train + args.n_calib + args.n_test
    assert total == need, f"Expected {need} rows, got {total}"

    seeds = parse_seeds(args.seeds)
    runs_root = Path("runs")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = np.arange(total)
        rng.shuffle(idx)
        shuf = [rows[i] for i in idx]

        train = shuf[: args.n_train]
        calib = shuf[args.n_train : args.n_train + args.n_calib]
        test = shuf[args.n_train + args.n_calib :]

        run_id = f"{args.dst_prefix}{seed}"
        out = runs_root / run_id
        out.mkdir(parents=True, exist_ok=True)

        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "calib.jsonl", calib)
        write_jsonl(out / "test.jsonl", test)

        # чистим артефакты, чтобы не было несостыковок
        for fn in ["risk_probe_lr.joblib", "risk_calibrator_isotonic.joblib", "risk_calibrator_platt.joblib"]:
            p = out / fn
            if p.exists():
                p.unlink()

        # (опционально) сохраняем метаданные
        meta = {"seed": seed, "n_train": args.n_train, "n_calib": args.n_calib, "n_test": args.n_test, "src": str(src)}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[make-seed-runs] wrote: {out}")

    print("[make-seed-runs] done.")


if __name__ == "__main__":
    main()
