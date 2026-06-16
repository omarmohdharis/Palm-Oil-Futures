"""
Phase 3 — Feature engineering for the FCPO buy/sell/hold classifier.

Builds a leakage-safe daily feature matrix from Phase-1 prices, using the
windows in config/features.yaml and the relationships confirmed in Phase 2 EDA.

Every feature at row t uses ONLY information available at the close of day t
(lags and trailing rolling windows look backwards). The forward-looking label
is built separately in Phase 4 — this module never touches the future.

Feature families:
  1. Lagged log returns
       - FCPO's own returns (autocorrelation / momentum)
       - soybean oil, Brent, WTI returns — the series Phase 2 Granger-flagged
         as LEADING FCPO. USD-MYR is deliberately EXCLUDED (no Granger evidence).
  2. FCPO rolling annualised volatility (regime context for labelling)
  3. FCPO–soybean-oil log-spread and its rolling z-score
       CAVEAT: Phase 2 found FCPO and soy are NOT cointegrated (p=0.12), so the
       spread is not reliably mean-reverting. The z-score is included because it
       was in the plan, but it is a weak mean-reversion signal — treat model
       importance on it with suspicion.

Output: data/features/feature_matrix.parquet
Run:    python -m src.features.build_features
"""

import numpy as np
import pandas as pd

from src.utils.config import load_config, project_root
from src.utils.io import save_parquet
from src.processing.eda import load_price_panel, log_returns

# Series whose LAGGED returns earn a feature. FCPO (own momentum) plus the
# Granger-significant leaders from Phase 2. usdmyr omitted on purpose.
_LEADERS = ["fcpo", "soybean_oil", "brent", "wti"]
_TRADING_DAYS = 252


def _lagged_returns(rets: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = {}
    for col in _LEADERS:
        for lag in lags:
            out[f"{col}_ret_lag{lag}"] = rets[col].shift(lag)
    return pd.DataFrame(out, index=rets.index)


def _rolling_vol(rets: pd.Series, windows: list[int]) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"fcpo_vol_{w}"] = rets.rolling(w).std() * np.sqrt(_TRADING_DAYS)
    return pd.DataFrame(out, index=rets.index)


def _spread_features(panel: pd.DataFrame, z_window: int) -> pd.DataFrame:
    """
    Log-spread between FCPO and soybean oil, plus a rolling z-score.

    Using logs makes the spread scale-free (the two prices are quoted in
    different units). The z-score is computed over a trailing window only, so
    no future data leaks in.  See module caveat on cointegration.
    """
    spread = np.log(panel["fcpo"]) - np.log(panel["soybean_oil"])
    roll = spread.rolling(z_window)
    z = (spread - roll.mean()) / roll.std()
    return pd.DataFrame(
        {"fcpo_soy_spread": spread, "fcpo_soy_spread_z": z},
        index=panel.index,
    )


def build_features() -> pd.DataFrame:
    cfg = load_config("features")
    lags = cfg["returns"]["lags"]
    vol_windows = cfg["rolling_vol"]["windows"]
    z_window = cfg["spread"]["z_score_window"]

    panel = load_price_panel()
    rets = log_returns(panel)

    features = pd.concat(
        [
            _lagged_returns(rets, lags),
            _rolling_vol(rets["fcpo"], vol_windows),
            _spread_features(panel, z_window).reindex(rets.index),
        ],
        axis=1,
    )

    # Drop the warm-up rows where the longest lag / rolling window isn't filled.
    before = len(features)
    features = features.dropna()
    print(f"[features] {features.shape[1]} columns, "
          f"{features.shape[0]} rows after dropping {before - len(features)} "
          f"warm-up rows ({features.index[0].date()} → {features.index[-1].date()})")

    out_path = project_root() / load_config("data")["paths"]["features"] / "feature_matrix.parquet"
    save_parquet(features, out_path)
    print(f"[OK] Phase 3 features → {out_path}")
    print("     columns:", list(features.columns))
    return features


if __name__ == "__main__":
    build_features()
