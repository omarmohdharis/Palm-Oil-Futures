"""
Phase 7 — Backtest: turn out-of-sample predictions into a P&L curve.

This is the project's REAL scoreboard (the plan's target is Sharpe + drawdown,
not accuracy). It reads the walk-forward OOS predictions from Phase 5 and the
FCPO price, applies realistic frictions from config/backtest.yaml, and reports
risk-adjusted performance vs a buy-&-hold benchmark.

Signal → position:
  BUY → +1 (long), SELL → -1 (short), HOLD → 0 (flat), one unit of exposure.

Timing / leakage:
  A prediction uses information through the close of day t. With
  execution_lag_days=1 we cannot act until the next session, so the position is
  shifted forward by the lag before it earns any return — the model never trades
  on a move it has already seen.

Costs (charged on TURNOVER, i.e. whenever the position changes):
  slippage  : slippage_pct, split per side
  spread    : spread_ticks × RM1/tonne, half-spread per side, as a fraction of price
  commission: commission_per_lot / (contract_tonnes × price) per side
  A flat→long is 1 side; a long→short flip is 2 sides.

Simplification (documented honestly): returns are close-to-close between
consecutive OOS dates rather than open-fill modelled tick-by-tick. With the
1-day lag this is the standard vectorised approximation; it will not capture
intraday entry slippage beyond the spread/slippage terms above.

Output:
  results/phase7/backtest.parquet   per-day equity, returns, position
  results/phase7/equity_curve.png
Run: python -m src.backtest.backtest
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, project_root
from src.utils.io import save_parquet, load_parquet

_CONTRACT_TONNES = 25          # FCPO contract size (metric tonnes)
_TICK_RM = 1.0                 # 1 tick = RM 1 / tonne
_TRADING_DAYS = 252
_SIGNAL_MAP = {"BUY": 1, "HOLD": 0, "SELL": -1}


def _fcpo_close() -> pd.Series:
    path = project_root() / "data" / "raw" / "fcpo" / "fcpo_raw.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["close"].sort_index()


def _per_side_cost(price: pd.Series, cfg: dict) -> pd.Series:
    """One-way trading cost as a fraction of notional, per unit of position."""
    slippage = cfg["slippage_pct"] / 2.0
    spread = (cfg["spread_ticks"] / 2.0) * _TICK_RM / price
    commission = cfg["commission_per_lot"] / (_CONTRACT_TONNES * price)
    return slippage + spread + commission


def _gated_position(oos: pd.DataFrame, threshold: float) -> pd.Series:
    """Map signals to exposure, but stay FLAT when the predicted directional
    class isn't confident enough (predicted-class probability < threshold).
    threshold=0 reproduces the always-trade strategy."""
    desired = oos["y_pred"].map(_SIGNAL_MAP).astype(float)
    conf = np.where(oos["y_pred"] == "BUY", oos["p_BUY"],
            np.where(oos["y_pred"] == "SELL", oos["p_SELL"], 1.0))
    desired[conf < threshold] = 0.0
    return desired


def _simulate(desired: pd.Series, close: pd.Series, asset_ret: pd.Series,
              costs: dict, capital: float, lag: int) -> tuple[pd.DataFrame, pd.Series]:
    """Apply execution lag, costs on turnover, and compound to an equity curve."""
    position = desired.shift(lag).fillna(0.0)
    sides = position.diff().abs().fillna(position.abs())
    cost = sides * _per_side_cost(close, costs)
    gross = position * asset_ret
    net = gross - cost
    equity = (1 + net).cumprod() * capital
    out = pd.DataFrame({
        "position": position, "asset_ret": asset_ret,
        "gross_ret": gross, "net_ret": net, "cost": cost, "equity": equity,
    })
    return out, sides


def _net_sharpe(desired, close, asset_ret, costs, capital, lag) -> float:
    out, _ = _simulate(desired, close, asset_ret, costs, capital, lag)
    return _metrics(out["net_ret"], out["equity"], "x")["sharpe"]


def adaptive_backtest(oos, close, asset_ret, costs, capital, lag,
                      candidates=(0.45, 0.50, 0.55, 0.60, 0.65),
                      block=63, warmup=378):
    """Honest threshold selection: at each block, pick the confidence threshold
    that worked best on PAST OOS data only, then apply it to the next block.
    No future information ever touches the choice — this is the number to trust,
    unlike the full-sample sweep (which peeks at the test set)."""
    desired = pd.Series(0.0, index=oos.index)
    chosen = []
    n = len(oos)
    for b in range(warmup, n, block):
        hist = oos.iloc[:b]
        best = max(candidates, key=lambda t: _net_sharpe(
            _gated_position(hist, t), close.iloc[:b], asset_ret.iloc[:b], costs, capital, lag))
        seg = oos.iloc[b:b + block]
        desired.loc[seg.index] = _gated_position(seg, best).values
        chosen.append(best)
    eval_idx = oos.index[warmup:]
    out, sides = _simulate(desired, close, asset_ret, costs, capital, lag)
    ev = out.loc[eval_idx].copy()
    ev["equity"] = (1 + ev["net_ret"]).cumprod() * capital
    m = _metrics(ev["net_ret"], ev["equity"], "adaptive (honest)")
    n_tr = int((sides.loc[eval_idx] > 0).sum())
    return m, n_tr, pd.Series(chosen), ev


def _metrics(returns: pd.Series, equity: pd.Series, label: str) -> dict:
    ann_ret = (1 + returns).prod() ** (_TRADING_DAYS / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(_TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "strategy": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
    }


def run_backtest(pred_name: str = "oos_predictions.parquet") -> pd.DataFrame:
    cfg = load_config("backtest")
    costs = cfg["costs"]
    capital = cfg["position"]["initial_capital"]
    lag = cfg["execution_lag_days"]

    print(f"[backtest] scoring predictions: {pred_name}")
    oos = load_parquet(project_root() / "results" / "phase5" / pred_name).sort_index()
    close = _fcpo_close().reindex(oos.index)
    asset_ret = close.pct_change().fillna(0.0)
    bh_equity = (1 + asset_ret).cumprod() * capital      # buy & hold benchmark

    print(f"[backtest] {len(oos)} OOS days ({oos.index[0].date()} → {oos.index[-1].date()}), "
          f"lag {lag}d\n")

    # ── confidence-threshold sweep (the turnover fix) ──
    print("Confidence no-trade band sweep:")
    print(f"{'threshold':>10}{'trades':>8}{'in mkt':>8}{'net Sharpe':>12}"
          f"{'net total':>11}{'max DD':>9}")
    sweep_rows = []
    for thr in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60]:
        out, sides = _simulate(_gated_position(oos, thr), close, asset_ret, costs, capital, lag)
        m = _metrics(out["net_ret"], out["equity"], f"thr={thr}")
        n_tr = int((sides > 0).sum())
        in_mkt = (out["position"] != 0).mean()
        sweep_rows.append((thr, m, out, n_tr))
        print(f"{thr:>10.2f}{n_tr:>8}{in_mkt:>8.1%}{m['sharpe']:>12.2f}"
              f"{m['total_return']:>11.1%}{m['max_drawdown']:>9.1%}")

    # pick threshold with best net Sharpe for the saved curve
    best_thr, best_m, best_out, best_trades = max(sweep_rows, key=lambda r: r[1]["sharpe"])
    trades_full = sweep_rows[0][3]   # trade count at thr=0.00 (always-trade)
    print(f"\n[sweep best — OPTIMISTIC, peeks at test set] threshold={best_thr:.2f}  "
          f"net Sharpe={best_m['sharpe']:.2f}  trades={best_trades} (was {trades_full} @0.00)")

    # ── honest, no-peeking threshold selection ──
    am, an_tr, chosen, adaptive_out = adaptive_backtest(oos, close, asset_ret, costs, capital, lag)
    print(f"[adaptive — HONEST, past data only] net Sharpe={am['sharpe']:.2f}  "
          f"total={am['total_return']:.1%}  maxDD={am['max_drawdown']:.1%}  trades={an_tr}  "
          f"(thresholds used: {chosen.value_counts().to_dict()})")

    # ── full report at the best threshold vs benchmarks ──
    out = best_out
    out["equity_buyhold"] = bh_equity

    # The SAVED artifact (what the dashboard displays) is the HONEST adaptive
    # curve — the sweep-best above peeks at the test set and stays console-only.
    adaptive_out["equity_buyhold"] = ((1 + asset_ret.loc[adaptive_out.index]).cumprod()
                                      * capital)
    m_net = _metrics(out["net_ret"], out["equity"], f"strategy net (thr={best_thr})")
    m_gross = _metrics(out["gross_ret"], (1 + out["gross_ret"]).cumprod() * capital, "strategy gross")
    m_bh = _metrics(asset_ret, bh_equity, "buy & hold")
    report = pd.DataFrame([m_net, m_gross, m_bh]).set_index("strategy")
    print("\n" + report.to_string(formatters={
        "ann_return": "{:.1%}".format, "ann_vol": "{:.1%}".format,
        "sharpe": "{:.2f}".format, "max_drawdown": "{:.1%}".format,
        "total_return": "{:.1%}".format,
    }))

    # ── plot (honest adaptive curve) ──
    out_dir = project_root() / "results" / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    ax = adaptive_out[["equity", "equity_buyhold"]].plot(
        figsize=(11, 5), title="Phase 7 — strategy (net, honest adaptive) vs buy & hold")
    ax.set_ylabel("equity (MYR)")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "equity_curve.png", dpi=120)
    plt.close(ax.figure)

    save_parquet(adaptive_out, out_dir / "backtest.parquet")
    print(f"\n[OK] Phase 7 backtest → {out_dir / 'backtest.parquet'}")
    print(f"     equity curve  → {out_dir / 'equity_curve.png'}")
    return out


if __name__ == "__main__":
    import sys
    run_backtest(sys.argv[1] if len(sys.argv) > 1 else "oos_predictions.parquet")
