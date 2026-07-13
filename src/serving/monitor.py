"""
Phase 8 — Monitoring: is the live model still working?

A signal you can't audit live is just a backtest with extra steps. This scores
logged signals against what actually happened (FCPO's realised 3-day move) and
checks whether recent performance still resembles the backtest — flagging decay.

For each logged signal old enough to have an outcome:
  realised 3-day forward return on FCPO's calendar →
  was the directional call right?  what did acting on it earn (gross)?

It compares a RECENT window to the full history; a persistently negative recent
window past `min_signals` raises a decay flag.

Bootstrapping: `--backfill` seeds the log from the model's out-of-sample history
(results/phase5/oos_predictions_gbm.parquet) so you have a track record on day
one instead of waiting months for live signals to accumulate.

Run: python -m src.serving.monitor
     python -m src.serving.monitor --backfill   # seed history first
"""

import sys

import numpy as np
import pandas as pd

from src.backtest.backtest import _fcpo_close, _gated_position
from src.serving.predict import _FLAT
from src.utils.config import load_config, project_root
from src.utils.io import save_parquet, load_parquet

_HORIZON = 3
_RECENT = 30          # trades in the "recent" decay window


def backfill_from_oos() -> None:
    """Seed the signal log from frozen-model OOS predictions + its threshold."""
    import json
    reg = json.loads((project_root() / load_config("serve")["model"]["registry"]).read_text())
    oos = load_parquet(project_root() / "results" / "phase5" / "oos_predictions_gbm.parquet").sort_index()
    thr = reg["threshold"]

    raw = oos["y_pred"]
    conf = np.where(raw == "BUY", oos["p_BUY"], np.where(raw == "SELL", oos["p_SELL"], oos["p_HOLD"]))
    final = [r if (r == "HOLD" or c >= thr) else _FLAT[r] for r, c in zip(raw, conf)]
    close = _fcpo_close().reindex(oos.index)

    log = pd.DataFrame({
        "fcpo_close": close.values,
        "raw_signal": raw.values,
        "confidence": np.round(conf, 3),
        "threshold": thr,
        "final_signal": final,
        "p_BUY": oos["p_BUY"].values, "p_HOLD": oos["p_HOLD"].values, "p_SELL": oos["p_SELL"].values,
        "model_file": reg["model_file"],
        "trained_through": reg["trained_through"],
        "predicted_at": "backfill",
    }, index=oos.index)
    log.index.name = "asof"
    save_parquet(log, project_root() / load_config("serve")["signals"]["log"])
    print(f"[backfill] seeded {len(log)} historical signals from OOS predictions")


def _score(log: pd.DataFrame) -> pd.DataFrame:
    close = _fcpo_close().sort_index()
    fwd = close.shift(-_HORIZON) / close - 1.0          # realised 3-day move
    hold_band = load_config("features")["label"]["buy_threshold"]   # same band as labels
    s = log.copy()
    s["fwd_ret_3d"] = fwd.reindex(s.index)
    s = s.dropna(subset=["fwd_ret_3d"])                 # keep only signals with an outcome
    pos = {"BUY": 1, "SELL": -1, "HOLD": 0}
    s["position"] = s["final_signal"].map(pos)
    s["realised_pnl"] = s["position"] * s["fwd_ret_3d"]
    s["correct"] = np.where(
        s["final_signal"] == "HOLD", s["fwd_ret_3d"].abs() <= hold_band,
        s["realised_pnl"] > 0)
    return s


def monitor() -> None:
    cfg = load_config("serve")
    log_path = project_root() / cfg["signals"]["log"]
    if not log_path.exists():
        print("[monitor] no signal log yet — run predict, or `--backfill` to seed history.")
        return

    scored = _score(load_parquet(log_path).sort_index())
    trades = scored[scored["position"] != 0]
    min_sig = cfg["monitor"]["min_signals"]

    lines = ["# FCPO model monitoring", ""]
    lines.append(f"- Signals scored (outcome known): **{len(scored)}**")
    lines.append(f"- Directional trades (non-HOLD): **{len(trades)}**")

    if len(trades) == 0:
        lines.append("\nNo directional trades to score yet.")
        verdict = "INSUFFICIENT DATA"
    else:
        hit = trades["correct"].mean()
        avg = trades["realised_pnl"].mean()
        cum = trades["realised_pnl"].sum()
        recent = trades.tail(_RECENT)
        r_hit, r_avg = recent["correct"].mean(), recent["realised_pnl"].mean()
        lines += [
            f"- Directional hit rate: **{hit:.1%}**  (recent {len(recent)}: {r_hit:.1%})",
            f"- Avg realised return / trade: **{avg:+.2%}**  (recent: {r_avg:+.2%})",
            f"- Cumulative realised (gross): **{cum:+.1%}**",
        ]
        if len(scored) < min_sig:
            verdict = f"WARMING UP ({len(scored)}/{min_sig} signals)"
        elif r_avg < 0 and r_hit < 0.5:
            verdict = "⚠️ POSSIBLE DECAY — recent trades losing; review / retrain"
        else:
            verdict = "✓ HEALTHY — recent performance in line with expectations"

    lines.append(f"\n## Verdict: {verdict}")
    report = "\n".join(lines)

    out = project_root() / cfg["monitor"]["report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[monitor] report → {out}")


if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill_from_oos()
    monitor()
