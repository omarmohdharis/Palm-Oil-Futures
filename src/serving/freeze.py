"""
Phase 8 — Freeze a production model.

The research code retrains a model inside every walk-forward fold and throws it
away. To SERVE predictions we need one fixed, versioned model plus the decision
threshold to apply to it.

This step:
  1. runs the walk-forward once to get honest out-of-sample predictions,
  2. picks the confidence threshold that maximised net Sharpe over that history
     (past data only — the same rule the live model will face going forward),
  3. trains the final GBM on ALL available data, and
  4. saves the model (.pkl) + a registry JSON recording the threshold, the exact
     feature columns, the class order, and how far the training data runs.

predict.py loads the registry — it never retrains.

Run: python -m src.serving.freeze
"""

import json
import datetime as dt

import joblib

from src.models.baseline import _load_dataset, walk_forward
from src.models.gbm import _new_gbm
from src.backtest.backtest import _fcpo_close, _gated_position, _net_sharpe
from src.utils.config import load_config, project_root


def _pick_threshold(oos) -> tuple[float, dict]:
    """Choose the confidence threshold with the best net Sharpe over all OOS
    history (uses only data the live model will also have seen)."""
    bt = load_config("backtest")
    costs = bt["costs"]
    capital = bt["position"]["initial_capital"]
    lag = bt["execution_lag_days"]
    candidates = load_config("serve")["model"]["threshold_candidates"]

    close = _fcpo_close().reindex(oos.index)
    asset_ret = close.pct_change().fillna(0.0)
    scored = {t: _net_sharpe(_gated_position(oos, t), close, asset_ret, costs, capital, lag)
              for t in candidates}
    best = max(scored, key=scored.get)
    return best, {str(k): round(v, 3) for k, v in scored.items()}


def freeze() -> None:
    cfg = load_config("serve")["model"]
    X, y = _load_dataset()

    print("[freeze] running walk-forward to choose the live threshold …")
    oos = walk_forward(X, y, _new_gbm)
    threshold, scored = _pick_threshold(oos)
    print(f"[freeze] threshold candidates (net Sharpe): {scored}")
    print(f"[freeze] chosen threshold: {threshold}")

    print(f"[freeze] training final GBM on all {len(X)} rows …")
    model = _new_gbm().fit(X, y)

    version = dt.date.today().isoformat()
    model_dir = project_root() / cfg["dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"gbm_{version}.pkl"
    joblib.dump(model, model_path)

    registry = {
        "model_file": model_path.name,
        "model_type": "HistGradientBoostingClassifier",
        "threshold": threshold,
        "features": list(X.columns),
        "classes": list(model.classes_),
        "trained_through": str(X.index[-1].date()),
        "n_train_rows": int(len(X)),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    registry_path = project_root() / cfg["registry"]
    registry_path.write_text(json.dumps(registry, indent=2))

    print(f"[OK] model  → {model_path}")
    print(f"[OK] registry → {registry_path}")
    print(f"     serves classes {registry['classes']} with threshold {threshold}, "
          f"trained through {registry['trained_through']}")


if __name__ == "__main__":
    freeze()
