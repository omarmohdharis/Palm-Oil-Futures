"""
Phase 8 — Short-horizon direction forecast (1–2 days) + accuracy tracking.

Predicts whether FCPO will be UP or DOWN over the next 1 and 2 trading days, with
a confidence, and logs every forecast with the date it is FOR. As new prices
arrive, score() compares each past forecast to what actually happened and reports
a running directional accuracy — which sharpens as more data accumulates.

Honesty:
  - Every historical forecast comes from the walk-forward (the model never saw the
    outcome), so the accuracy is genuinely out-of-sample, not hindsight.
  - The most recent 1–2 days have no outcome yet — they are logged as PENDING and
    scored automatically once their target date's price exists.
  - Short horizons are noisier than the 3-day signal; expect accuracy only modestly
    above the 50% coin-flip baseline. That gap IS the edge.

Run: python -m src.serving.forecast          # make + log forecasts, then score
     python -m src.serving.forecast --score  # just re-score the existing log
"""

import sys

import numpy as np
import pandas as pd

from src.backtest.backtest import _fcpo_close
from src.models.baseline import walk_forward
from src.models.gbm import _new_gbm
from src.utils.config import load_config, project_root
from src.utils.io import save_parquet, load_parquet


def _direction(fwd_ret: pd.Series) -> pd.Series:
    return fwd_ret.apply(lambda v: "UP" if v > 0 else ("DOWN" if v < 0 else None))


def _target_dates(dates: pd.DatetimeIndex, horizon: int, cal: pd.DatetimeIndex):
    """The trading day `horizon` sessions after each date (NaT if still in future)."""
    pos = cal.get_indexer(dates)
    return [cal[p + horizon] if (p >= 0 and p + horizon < len(cal)) else pd.NaT
            for p in pos]


def _records_for_horizon(horizon: int) -> pd.DataFrame:
    feats = load_parquet(project_root() / load_config("data")["paths"]["features"]
                         / "feature_matrix.parquet").sort_index()
    close = _fcpo_close().sort_index()
    fwd = close.shift(-horizon) / close - 1.0
    y = _direction(fwd).reindex(feats.index).dropna()
    X = feats.loc[y.index]

    # historical: out-of-sample walk-forward forecasts
    oos = walk_forward(X, y, _new_gbm)
    # pending: most recent feature rows that have no outcome yet → predict live
    pending_idx = feats.index[feats.index > y.index[-1]]
    final = _new_gbm().fit(X, y)

    frames = []
    for idx, src, yhat, proba in [
        (oos.index, "oos", oos["y_pred"].values,
         oos[[f"p_{c}" for c in final.classes_]].values),
        (pending_idx, "pending",
         final.predict(feats.loc[pending_idx]) if len(pending_idx) else np.array([]),
         final.predict_proba(feats.loc[pending_idx]) if len(pending_idx) else np.empty((0, len(final.classes_)))),
    ]:
        if len(idx) == 0:
            continue
        p_up = proba[:, list(final.classes_).index("UP")]
        frames.append(pd.DataFrame({
            "horizon": horizon,
            "predicted_dir": yhat,
            "p_up": np.round(p_up, 3),
            "confidence": np.round(np.maximum(p_up, 1 - p_up), 3),
            "close_at_forecast": close.reindex(idx).values,
            "target_date": _target_dates(idx, horizon, close.index),
            "source": src,
        }, index=idx))
    out = pd.concat(frames)
    out.index.name = "forecast_date"
    return out


def make_forecasts() -> pd.DataFrame:
    cfg = load_config("serve")["forecast"]
    log = pd.concat([_records_for_horizon(h) for h in cfg["horizons"]])
    log = log.sort_values(["horizon", "forecast_date"])
    save_parquet(log, project_root() / cfg["log"])

    print("\n[forecast] latest call per horizon:")
    for h in cfg["horizons"]:
        last = log[log["horizon"] == h].iloc[-1]
        print(f"   next {h}d: {last['predicted_dir']:>4}  "
              f"(confidence {last['confidence']:.0%}, as of {last.name.date()})")
    return log


def score() -> None:
    cfg = load_config("serve")["forecast"]
    log_path = project_root() / cfg["log"]
    if not log_path.exists():
        print("[forecast] no forecast log — run `python -m src.serving.forecast` first.")
        return
    log = load_parquet(log_path)
    close = _fcpo_close().sort_index()

    # realised outcome for forecasts whose target date now has a price
    tgt_close = close.reindex(pd.to_datetime(log["target_date"]))
    actual_ret = tgt_close.values / log["close_at_forecast"].values - 1.0
    log = log.assign(actual_ret=actual_ret)
    log["actual_dir"] = _direction(pd.Series(actual_ret, index=log.index))
    scored = log.dropna(subset=["actual_dir"]).copy()
    scored["correct"] = scored["predicted_dir"] == scored["actual_dir"]

    lines = ["# Short-horizon forecast accuracy", "",
             "How often the 1–2 day UP/DOWN call was right (out-of-sample). "
             "Coin-flip baseline = 50%.", ""]
    recent_n = cfg["recent"]
    for h in cfg["horizons"]:
        s = scored[scored["horizon"] == h]
        pend = log[(log["horizon"] == h) & (log["actual_dir"].isna())]
        if len(s) == 0:
            lines.append(f"- **{h}-day:** no scored forecasts yet ({len(pend)} pending)")
            continue
        acc = s["correct"].mean()
        racc = s["correct"].tail(recent_n).mean()
        # accuracy weighted by how big the actual move was (did it call the big ones?)
        wacc = np.average(s["correct"], weights=s["actual_ret"].abs() + 1e-9)
        lines.append(
            f"- **{h}-day:** {acc:.1%} overall ({len(s)} scored) · "
            f"{racc:.1%} last {min(recent_n, len(s))} · "
            f"{wacc:.1%} on big moves · {len(pend)} pending")

    # most recent forecasts and how they turned out
    lines += ["", "## Latest forecasts", "",
              "| made on | horizon | call | confidence | for date | actual | result |",
              "|---|---|---|---|---|---|---|"]
    for idx, r in log.sort_values("forecast_date").groupby("horizon").tail(4).sort_values("forecast_date").iterrows():
        if pd.isna(r["actual_dir"]):
            res, act = "⏳ pending", "—"
        else:
            res = "✅ right" if r["predicted_dir"] == r["actual_dir"] else "❌ wrong"
            act = f"{r['actual_dir']} ({r['actual_ret']:+.1%})"
        tgt = pd.to_datetime(r["target_date"]).date() if pd.notna(r["target_date"]) else "—"
        lines.append(f"| {idx.date()} | {r['horizon']}d | {r['predicted_dir']} | "
                     f"{r['confidence']:.0%} | {tgt} | {act} | {res} |")

    report = "\n".join(lines)
    out = project_root() / cfg["report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[forecast] accuracy report → {out}")


if __name__ == "__main__":
    if "--score" not in sys.argv:
        make_forecasts()
    score()
