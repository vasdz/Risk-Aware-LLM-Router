from gate import RiskGate

class RiskRouter:
    def __init__(self, run_dir: str, alpha_ship=0.02, alpha_escalate=0.05, mode="raw"):
        thresholds_file = f"gate_thresholds_{mode}.json"
        self.g_ship = RiskGate(run_dir=run_dir, alpha=alpha_ship, route_on_reject="escalate", thresholds_file=thresholds_file)
        self.g_esc  = RiskGate(run_dir=run_dir, alpha=alpha_escalate, route_on_reject="refuse", thresholds_file=thresholds_file)

    def decide(self, hs_tbg):
        d1 = self.g_ship.decide(hs_tbg)
        if d1["accept"]:
            return {"action": "ship_small", **d1}

        d2 = self.g_esc.decide(hs_tbg)
        if d2["accept"]:
            return {"action": "escalate_big", **d2}

        return {"action": "refuse", **d2}
