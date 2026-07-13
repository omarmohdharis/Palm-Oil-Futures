"""
Phase 8 — Daily prediction (the deliverable's core).

Loads the frozen model from the registry, scores the most recent feature row,
applies the frozen confidence threshold, and emits today's recommendation. The
call is appended to a dated signal log and written as a readable report.

It NEVER retrains and never looks at the future — it only reads the latest
available feature row and the frozen artifact.

Expected daily order (see serve.py): refresh FCPO data → build features → predict.

Run: python -m src.serving.predict
"""

import json
import datetime as dt

import joblib
import pandas as pd

from src.backtest.backtest import _fcpo_close
from src.utils.config import load_config, project_root
from src.utils.io import save_parquet, load_parquet

_FLAT = {"BUY": "HOLD", "SELL": "HOLD"}   # what a low-confidence call collapses to


def _load_registry() -> dict:
    path = project_root() / load_config("serve")["model"]["registry"]
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen model at {path}. Run `python -m src.serving.freeze` first.")
    return json.loads(path.read_text())


def _latest_features(reg: dict) -> tuple[pd.Series, pd.Timestamp]:
    feat_path = project_root() / load_config("data")["paths"]["features"] / "feature_matrix.parquet"
    feats = load_parquet(feat_path).sort_index()
    row = feats.iloc[-1]
    missing = [c for c in reg["features"] if c not in feats.columns]
    if missing:
        raise ValueError(f"feature matrix is missing columns the model needs: {missing}")
    return row[reg["features"]], feats.index[-1]


def predict() -> dict:
    cfg = load_config("serve")
    reg = _load_registry()
    model = joblib.load(project_root() / cfg["model"]["dir"] / reg["model_file"])

    row, asof = _latest_features(reg)
    proba = model.predict_proba(row.to_frame().T)[0]
    probs = dict(zip(reg["classes"], proba.round(3)))
    raw = reg["classes"][int(proba.argmax())]
    confidence = float(proba.max())
    threshold = reg["threshold"]

    # confidence gate: a directional call below threshold becomes HOLD (stay flat)
    final = raw if (raw == "HOLD" or confidence >= threshold) else _FLAT[raw]

    close = _fcpo_close()
    record = {
        "asof": pd.Timestamp(asof),
        "fcpo_close": float(close.reindex([asof]).iloc[0]) if asof in close.index else float("nan"),
        "raw_signal": raw,
        "confidence": round(confidence, 3),
        "threshold": threshold,
        "final_signal": final,
        "p_BUY": probs.get("BUY"), "p_HOLD": probs.get("HOLD"), "p_SELL": probs.get("SELL"),
        "model_file": reg["model_file"],
        "trained_through": reg["trained_through"],
        "predicted_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _append_log(record, cfg["signals"]["log"])
    _write_report(record, cfg["signals"]["report"])

    gate = "confident" if final == raw else f"LOW-confidence → flat (was {raw})"
    print(f"[predict] {record['asof'].date()}  →  {final}  "
          f"(conf {confidence:.0%} vs thr {threshold:.0%}, {gate})")
    print(f"[predict] probs {probs}")
    return record


def _append_log(record: dict, log_rel: str) -> None:
    path = project_root() / log_rel
    new = pd.DataFrame([record]).set_index("asof")
    if path.exists():
        log = load_parquet(path)
        log = pd.concat([log[log.index != new.index[0]], new]).sort_index()
    else:
        log = new
    save_parquet(log, path)
    print(f"[predict] logged → {path}  ({len(log)} signals on file)")


def _write_report(r: dict, report_rel: str) -> None:
    path = project_root() / report_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}[r["final_signal"]]
    md = f"""# FCPO signal — {r['asof'].date()}

## Recommendation:  {emoji}

| | |
|---|---|
| As of (data date) | {r['asof'].date()} |
| FCPO close | {r['fcpo_close']:.2f} |
| Raw model call | {r['raw_signal']} |
| Confidence | {r['confidence']:.0%} (threshold {r['threshold']:.0%}) |
| Class probabilities | BUY {r['p_BUY']:.0%} · HOLD {r['p_HOLD']:.0%} · SELL {r['p_SELL']:.0%} |
| Model | {r['model_file']} (trained through {r['trained_through']}) |

> **Advisory only.** This is a decision-support signal over a ~3-day horizon, not
> financial advice or an order. Run on paper before risking capital. A low-confidence
> directional call is intentionally downgraded to HOLD.
"""
    path.write_text(md, encoding="utf-8")
    print(f"[predict] report → {path}")


if __name__ == "__main__":
    predict()
