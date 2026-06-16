"""
Phase 2 — Exploratory Data Analysis for the FCPO trading pipeline.

Runs entirely on Phase-1 data already collected (MPOB not required):
  - FCPO close          data/raw/fcpo/fcpo_raw.csv
  - related instruments data/raw/related/{soybean_oil,brent,wti,usdmyr}.csv
  - CPO spot price      data/processed/mpob_monthly_clean.parquet  (monthly)

Four analyses, each answering a concrete modelling question:
  1. ADF stationarity   → do we model price LEVELS or RETURNS?
  2. Granger causality  → which related series LEAD FCPO, and at what lag?
  3. Cointegration      → is FCPO tied long-run to soybean oil? (spread feature)
  4. Regime plot        → visualise volatility clustering before labelling

Output: console report + PNG plots in results/eda/.

Run:  python -m src.processing.eda
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint

from src.utils.config import project_root

# Daily series we can build from Phase-1 CSVs.  FCPO is the target; the rest
# are candidate predictors.  All are daily close prices.
_RELATED = ["soybean_oil", "brent", "wti", "usdmyr"]
_ALPHA = 0.05                  # significance level used for every verdict


# ── data loading ──────────────────────────────────────────────────────────────

def _read_close(path, name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["close"].rename(name).sort_index()


def load_price_panel() -> pd.DataFrame:
    """
    Daily close-price panel aligned on common trading days.

    FCPO trades on the Malaysian calendar and the related instruments on the
    US calendar, so we inner-join: only days where every series traded survive.
    That keeps returns honest (no fabricated overnight gaps) at the cost of a
    few holidays — fine for EDA.
    """
    root = project_root()
    series = [_read_close(root / "data" / "raw" / "fcpo" / "fcpo_raw.csv", "fcpo")]
    for name in _RELATED:
        series.append(_read_close(root / "data" / "raw" / "related" / f"{name}.csv", name))

    panel = pd.concat(series, axis=1, join="inner").dropna()
    print(f"[data] aligned panel: {panel.shape[0]} rows "
          f"({panel.index[0].date()} → {panel.index[-1].date()}), "
          f"columns: {list(panel.columns)}\n")
    return panel


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns — the stationary form used for Granger tests.

    WTI closed NEGATIVE on 2020-04-20 (-$37.63, the COVID storage crash), and
    log() is undefined for non-positive prices. We mask non-positive values to
    NaN so that single day drops out of the return series instead of poisoning
    it with NaNs/warnings — it's a genuine market print, not a data error.
    """
    positive = panel.where(panel > 0)
    return np.log(positive).diff().dropna()


# ── 1. stationarity ─────────────────────────────────────────────────────────────

def adf_report(panel: pd.DataFrame) -> None:
    """
    Augmented Dickey-Fuller test on each series in LEVELS and in LOG-RETURNS.

    Reading it: ADF's null hypothesis is "has a unit root" (non-stationary).
    p < 0.05  → reject the null → stationary.
    The expected commodity-price result is: levels NON-stationary, returns
    stationary — which is exactly why we model returns, not raw prices.
    """
    print("=" * 70)
    print("1. STATIONARITY  (ADF — null = non-stationary; p<0.05 ⇒ stationary)")
    print("=" * 70)
    rets = log_returns(panel)
    print(f"{'series':<14}{'level p':>12}{'verdict':>16}"
          f"{'return p':>12}{'verdict':>16}")
    print("-" * 70)
    for col in panel.columns:
        p_lvl = adfuller(panel[col].dropna(), autolag="AIC")[1]
        p_ret = adfuller(rets[col].dropna(), autolag="AIC")[1]
        v_lvl = "stationary" if p_lvl < _ALPHA else "NON-stationary"
        v_ret = "stationary" if p_ret < _ALPHA else "NON-stationary"
        print(f"{col:<14}{p_lvl:>12.4f}{v_lvl:>16}{p_ret:>12.4f}{v_ret:>16}")
    print("\n→ Model the STATIONARY form (returns) for any series flagged "
          "NON-stationary in levels.\n")


# ── 2. Granger causality ────────────────────────────────────────────────────────

def granger_report(panel: pd.DataFrame, maxlag: int = 5) -> None:
    """
    Does each related series' return help predict FCPO's return?

    Granger's null is "predictor does NOT help". We scan lags 1..maxlag and
    report the smallest p-value (and the lag where it occurs). A small p-value
    means that lag of the predictor carries information about next FCPO moves —
    a direct hint for which lagged features to engineer in Phase 3.
    """
    print("=" * 70)
    print(f"2. GRANGER CAUSALITY → FCPO return  (lags 1..{maxlag}; "
          "null = 'does NOT help')")
    print("=" * 70)
    rets = log_returns(panel)
    print(f"{'predictor':<14}{'best lag':>10}{'min p':>12}{'verdict':>20}")
    print("-" * 70)
    for pred in _RELATED:
        pair = rets[["fcpo", pred]].dropna()        # col0=target, col1=predictor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # silence verbose deprecation
            res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
        pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
        best_lag = min(pvals, key=pvals.get)
        min_p = pvals[best_lag]
        verdict = f"leads FCPO @lag{best_lag}" if min_p < _ALPHA else "no evidence"
        print(f"{pred:<14}{best_lag:>10}{min_p:>12.4f}{verdict:>20}")
    print("\n→ Series that 'lead FCPO' are worth lagged features in Phase 3.\n")


# ── 3. cointegration ────────────────────────────────────────────────────────────

def cointegration_report(panel: pd.DataFrame) -> None:
    """
    Engle-Granger cointegration test between FCPO and soybean oil (log levels).

    Palm and soybean oil are close substitutes, so their prices may share a
    long-run equilibrium even though each wanders on its own day to day.
    Null = "no cointegration". p < 0.05 ⇒ a stable spread exists, which
    justifies the FCPO–soybean-oil spread as a mean-reversion feature.
    """
    print("=" * 70)
    print("3. COINTEGRATION  FCPO ↔ soybean oil  (null = 'no cointegration')")
    print("=" * 70)
    fcpo = np.log(panel["fcpo"])
    soy = np.log(panel["soybean_oil"])
    t_stat, p_value, _ = coint(fcpo, soy)
    verdict = ("COINTEGRATED — spread is mean-reverting; build the spread feature"
               if p_value < _ALPHA else
               "not cointegrated — spread may drift; treat with caution")
    print(f"  t-statistic : {t_stat:.4f}")
    print(f"  p-value     : {p_value:.4f}")
    print(f"  → {verdict}\n")


# ── 4. regime visualisation ──────────────────────────────────────────────────────

def regime_plot(panel: pd.DataFrame, window: int = 21) -> None:
    """
    Save two plots: normalised price paths, and FCPO rolling annualised
    volatility. The vol plot makes regime/clustering visible before labelling —
    high-vol clusters are where buy/sell/hold thresholds behave differently.
    """
    out_dir = project_root() / "results" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    rets = log_returns(panel)

    # normalised prices (all start at 100) — relative co-movement at a glance
    norm = panel / panel.iloc[0] * 100
    ax = norm.plot(figsize=(11, 5), title="Normalised prices (start = 100)")
    ax.set_ylabel("index")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "normalised_prices.png", dpi=120)
    plt.close(ax.figure)

    # FCPO rolling annualised volatility
    vol = rets["fcpo"].rolling(window).std() * np.sqrt(252)
    ax = vol.plot(figsize=(11, 5),
                  title=f"FCPO rolling {window}-day annualised volatility")
    ax.set_ylabel("annualised σ")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "fcpo_volatility_regime.png", dpi=120)
    plt.close(ax.figure)

    print("=" * 70)
    print("4. REGIME PLOTS")
    print("=" * 70)
    print(f"  saved → {out_dir / 'normalised_prices.png'}")
    print(f"  saved → {out_dir / 'fcpo_volatility_regime.png'}")
    print(f"  FCPO annualised vol: median {vol.median():.1%}, "
          f"max {vol.max():.1%}\n")


# ── entry point ──────────────────────────────────────────────────────────────────

def run_eda() -> None:
    panel = load_price_panel()
    adf_report(panel)
    granger_report(panel)
    cointegration_report(panel)
    regime_plot(panel)
    print("[OK] Phase 2 EDA complete. Note which series are stationary, which "
          "lead FCPO, and whether the soy spread is cointegrated —\n"
          "     those answers drive the Phase 3 feature list.")


if __name__ == "__main__":
    run_eda()
