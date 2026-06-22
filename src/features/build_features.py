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


def _sentiment_features(index: pd.DatetimeIndex, news_dir=None, lag: int = 1) -> pd.DataFrame:
    """News-sentiment features per topic, aligned to trading days and lagged.

    For each topic (data/raw/news/<topic>.csv from news_sentiment.py) we build:
      <topic>_tone      : sentiment level
      <topic>_tone_chg  : tone vs its recent average (anticipation / momentum)
      <topic>_tone_z    : how stretched tone is (precedes the 'square-off' reversal)
      <topic>_buzz      : article volume (attention / fear)

    News is forward-filled onto trading days, then lagged by `lag` trading days so
    only news available before each decision is used. Returns empty if no news has
    been fetched yet — so the rest of the pipeline is unaffected until you run
    `python -m src.ingestion.news_sentiment`.
    """
    news_dir = news_dir or (project_root() / "data" / "raw" / "news")
    if not news_dir.exists():
        return pd.DataFrame(index=index)

    cols = {}
    for path in sorted(news_dir.glob("*.csv")):
        topic = path.stem
        df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
        # forward-fill news onto the trading calendar (news persists between prints)
        aligned = df.reindex(index.union(df.index)).sort_index().ffill().reindex(index)
        tone, vol = aligned.get("tone"), aligned.get("volume")
        if tone is not None:
            cols[f"{topic}_tone"] = tone.shift(lag)
            cols[f"{topic}_tone_chg"] = (tone - tone.rolling(5).mean()).shift(lag)
            roll = tone.rolling(21)
            cols[f"{topic}_tone_z"] = ((tone - roll.mean()) / roll.std()).shift(lag)
        if vol is not None:
            cols[f"{topic}_buzz"] = vol.shift(lag)
    return pd.DataFrame(cols, index=index)


def _mpob_features(index: pd.DatetimeIndex, release_lag_days: int = 45) -> pd.DataFrame:
    """MPOB monthly supply/demand features (stocks, production, exports), aligned
    to trading days with a release lag so the model never sees a figure before it
    was public.

    MPOB publishes month M's data ~the 10th of month M+1. We shift each monthly
    value forward by `release_lag_days` (~the release date), then forward-fill onto
    trading days. For each series we add the level and its month-over-month change.
    Returns empty if data/processed/mpob_monthly.parquet doesn't exist yet, so the
    pipeline is unaffected until MPOB files are parsed. NaNs (pre-2023) are fine —
    the gradient-boosted model handles them.
    """
    path = project_root() / load_config("data")["paths"]["processed"] / "mpob_monthly.parquet"
    if not path.exists():
        return pd.DataFrame(index=index)

    monthly = pd.read_parquet(path).sort_index()
    cols = {}
    for col in monthly.columns:
        level = monthly[col]
        change = level.pct_change()                       # MoM change (computed monthly)
        for name, ser in [(f"mpob_{col}", level), (f"mpob_{col}_chg", change)]:
            eff = ser.copy()
            eff.index = eff.index + pd.Timedelta(days=release_lag_days)   # ~public date
            cols[name] = (eff.reindex(index.union(eff.index)).sort_index()
                          .ffill().reindex(index))
    return pd.DataFrame(cols, index=index)


def build_features() -> pd.DataFrame:
    cfg = load_config("features")
    lags = cfg["returns"]["lags"]
    vol_windows = cfg["rolling_vol"]["windows"]
    z_window = cfg["spread"]["z_score_window"]

    panel = load_price_panel()
    rets = log_returns(panel)

    # core price features — drop the warm-up rows where lags/rolling aren't filled
    features = pd.concat(
        [
            _lagged_returns(rets, lags),
            _rolling_vol(rets["fcpo"], vol_windows),
            _spread_features(panel, z_window).reindex(rets.index),
        ],
        axis=1,
    ).dropna()

    # news sentiment — joined WITHOUT dropping rows (early/missing tone stays blank;
    # the gradient-boosted model handles NaNs natively)
    sentiment = _sentiment_features(features.index)
    if not sentiment.empty:
        features = features.join(sentiment)
        print(f"[features] + {sentiment.shape[1]} news-sentiment columns "
              f"({sentiment.notna().any(axis=1).sum()} rows have sentiment)")

    mpob = _mpob_features(features.index)
    if not mpob.empty:
        features = features.join(mpob)
        print(f"[features] + {mpob.shape[1]} MPOB supply/demand columns "
              f"({mpob.notna().any(axis=1).sum()} rows have MPOB data)")

    print(f"[features] {features.shape[1]} columns, {features.shape[0]} rows "
          f"({features.index[0].date()} → {features.index[-1].date()})")

    out_path = project_root() / load_config("data")["paths"]["features"] / "feature_matrix.parquet"
    save_parquet(features, out_path)
    print(f"[OK] Phase 3 features → {out_path}")
    print("     columns:", list(features.columns))
    return features


if __name__ == "__main__":
    build_features()
