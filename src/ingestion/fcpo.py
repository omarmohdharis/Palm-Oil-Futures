"""
FCPO (Crude Palm Oil Futures) data ingestion.

Tries sources in this order:
  1. yfinance       — free but spotty; may return partial history or nothing
  2. Manual CSV     — place downloaded files in data/raw/fcpo/ and re-run

Accepted CSV formats (auto-detected):
  a) Investing.com export  : Date, Price, Open, High, Low, Vol., Change %
  b) Standard OHLCV        : date, open, high, low, close[, volume]

Run with no FCPO files present to see download instructions.
"""

import re
import pandas as pd
import yfinance as yf
from pathlib import Path

from src.utils.config import load_config, project_root
from src.utils.io import save_raw_csv

_INVESTING_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ── CSV parsers ────────────────────────────────────────────────────────────────

def _parse_investing_date(s: str) -> pd.Timestamp:
    s = s.strip().replace(",", "")
    parts = s.split()
    if len(parts) == 3:
        m = _INVESTING_MONTHS.get(parts[0].lower())
        if m:
            return pd.Timestamp(year=int(parts[2]), month=m, day=int(parts[1]))
    return pd.Timestamp(s)


def _parse_investing_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["date"] = df["Date"].apply(_parse_investing_date)
    df = df.set_index("date").sort_index()
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace('"', "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.rename(columns={"Price": "close", "Open": "open",
                               "High": "high",  "Low":  "low"})[
        [c for c in ["open", "high", "low", "close"] if c in df.rename(
            columns={"Price": "close", "Open": "open", "High": "high", "Low": "low"}
        ).columns]
    ].copy()


def _parse_standard_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index.name = "date"
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].sort_index()


def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        header = path.read_text(encoding="utf-8", errors="ignore").split("\n")[0]
        if "Change %" in header or "Vol." in header:
            return _parse_investing_csv(path)
        return _parse_standard_csv(path)
    except Exception as e:
        print(f"  [WARN] could not parse {path.name}: {e}")
        return None


# ── yfinance attempt ───────────────────────────────────────────────────────────

def _try_yfinance(candidates: list[str], start: str, end: str) -> pd.DataFrame | None:
    for ticker in candidates:
        print(f"  trying yfinance ticker {ticker!r} …")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print("    no data returned")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        print(f"    {len(df):,} rows returned")
        return df
    return None


# ── public entry point ─────────────────────────────────────────────────────────

def fetch_fcpo(start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    cfg = load_config("data")
    start = start or cfg["dates"]["start"]
    end = end or cfg["dates"]["end"]
    raw_dir = project_root() / cfg["paths"]["raw"] / "fcpo"

    print("[→] FCPO — attempting yfinance …")
    df = _try_yfinance(cfg["fcpo"]["yfinance_candidates"], start, end)

    # yfinance is considered insufficient if it returns fewer than 100 rows
    if df is None or len(df) < 100:
        print("[→] FCPO — yfinance insufficient; scanning data/raw/fcpo/ for CSVs …")
        csvs = sorted(raw_dir.glob("*.csv"))
        if not csvs:
            _print_manual_instructions()
            return None
        frames = [_load_csv(p) for p in csvs]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            print("[ERROR] found CSVs but could not parse any — check file format")
            return None
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        print(f"  merged {len(csvs)} CSV file(s) → {len(df):,} rows total")

    save_raw_csv(df, raw_dir / "fcpo_raw.csv")
    return df


def _print_manual_instructions() -> None:
    print("""
[ACTION REQUIRED] No usable FCPO data found.

Option A — Investing.com (quickest, no account needed):
  1. Go to:  investing.com → Commodities → Crude Palm Oil → Historical Data
  2. Set date range: Jan 2015 → today
  3. Click "Download Data" (CSV)
  4. Move the downloaded file into:  data/raw/fcpo/
  5. Re-run this script

Option B — Barchart.com (standard OHLCV format):
  1. Search "FCPO Continuous" at barchart.com
  2. Download historical data (free tier ≈ 1 year; paid = full history)
  3. Move CSV into data/raw/fcpo/  — standard OHLCV is auto-detected

Option C — IBKR paper account (cleanest programmatic path):
  1. Open a free paper trading account at interactivebrokers.com
  2. Install:  pip install ib_insync
  3. Connect to Trader Workstation (TWS) and request FCPO on exchange MDEX
  4. Export as CSV and move to data/raw/fcpo/

Cross-check tip: if you get data from two sources, load both CSVs into
data/raw/fcpo/ — the parser merges them and deduplicates by date.
""")


if __name__ == "__main__":
    fetch_fcpo()
