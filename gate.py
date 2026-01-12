import json
import numpy as np
from joblib import load
from pathlib import Path

class RiskGate:
    def __init__(self, run_dir: str, alpha: float, route_on_reject: str = "bigger_model", thresholds_file: str | None = None):
        run_dir = Path(run_dir)
        self.lr = load(run_dir / "risk_probe_lr.joblib")

        if thresholds_file is None:
            thresholds_file = "gate_thresholds.json"  # backward compatible
        cfg = json.loads((run_dir / thresholds_file).read_text(encoding="utf-8"))

        self.mode = cfg["mode"]
        self.alpha = str(alpha)
        self.t = cfg["alphas"][self.alpha]["t"]
        self.route_on_reject = route_on_reject

        self.cal = None
        if self.mode == "platt":
            self.cal = load(run_dir / "risk_calibrator_platt.joblib")
        elif self.mode == "isotonic":
            self.cal = load(run_dir / "risk_calibrator_isotonic.joblib")

    def score_error_prob(self, hs_tbg):
        X = np.array(hs_tbg, dtype=np.float32).reshape(1, -1)
        if self.mode == "raw":
            return float(self.lr.predict_proba(X)[:, 1][0])
        return float(self.cal.predict_proba(X)[:, 1][0])

    def decide(self, hs_tbg):
        p_err = self.score_error_prob(hs_tbg)
        accept = (p_err <= self.t)
        if accept:
            return {"accept": True, "route": "small_model", "p_error": p_err, "threshold": self.t}
        return {"accept": False, "route": self.route_on_reject, "p_error": p_err, "threshold": self.t}
