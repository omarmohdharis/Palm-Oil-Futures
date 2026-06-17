"""
Fetch related instruments via yfinance:
  soybean_oil : ZL=F  (CBOT)
  brent       : BZ=F  (ICE)
  wti         : CL=F  (NYMEX)
  usdmyr      : MYR=X (spot FX)

Output: one CSV per instrument in data/raw/related/
"""

import datetime as dt

import pandas as pd
import yfinance as yf

from src.utils.config import load_config, project_root
from src.utils.io import save_raw_csv


def _default_end() -> str:
    # yfinance's `end` is EXCLUSIVE, so use tomorrow to include today's bar.
    return (dt.date.today() + dt.timedelta(days=1)).isoformat()


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    # yfinance >=0.2.40 wraps columns in a MultiIndex with the ticker as level 1
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].rename(columns=str.lower)


def fetch_related(start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    cfg = load_config("data")
    start = start or cfg["dates"]["start"]
    end = end or _default_end()          # roll forward to today, not the static config date
    raw_dir = project_root() / cfg["paths"]["raw"] / "related"

    results: dict[str, pd.DataFrame] = {}
    for name, ticker in cfg["tickers"].items():
        print(f"[→] {name} ({ticker}) …")
        df = _download(ticker, start, end)
        if df.empty:
            print(f"  [WARN] no data returned for {ticker} — verify ticker is still active")
            continue
        save_raw_csv(df, raw_dir / f"{name}.csv")
        results[name] = df

    return results


if __name__ == "__main__":
    fetch_related()
