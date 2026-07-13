"""
FCPO ingestion — Bursa Malaysia crude palm oil futures (FCPO), MYR/tonne.

Primary source : TradingView MYX:FCPO1! daily bars via tvdatafeed (anonymous).
                 One request returns the full ~20y history, including sessions
                 investing.com drops. Values verified against official Bursa
                 settlements (MPOC) on 2026-07-13.
Validation     : MPOC daily settlement table (official, courtesy of Bursa).
Fallbacks      : data/raw/fcpo/fcpo_ibkr.csv        (IBKR feed, if connected)
                 data/raw/fcpo/fcpo_manual_myr.csv  (typed via fcpo_manual.py)

── Date convention (empirically pinned 2026-07-13 — do not "simplify") ───────
FCPO's trade date T starts with the After-Hours night session at 21:00 MYT on
T-1 (Bursa: "trades performed during T+1 are included in the following day's
session"). TradingView stamps the daily bar at that session OPEN, rendered in
US-Eastern time — i.e. the bar is dated T-1. Monday sessions have no night
portion, so their bar opens at the 10:30 MYT day open = Sunday ~22:00 EDT.
Therefore: TRADE DATE = TV bar date + 1 calendar day, and the mapping restores
a normal Mon–Fri session calendar that matches MPOC settlement dates exactly.

The last TV bar is the CURRENTLY OPEN session (settles 18:00 MYT on its trade
date); it is dropped until settled so the model never sees a live/partial bar.

investing.com (kept as fcpo_investing_myr.csv for cross-checks only) moved to
the same night-open stamping in Jan-2025 AND drops the Sunday-stamped rows, so
its 2025+ data is one day off and missing all Mondays — never merge it.

── Instrument history note (2026-07-13) ──────────────────────────────────────
Before this date the project was accidentally built on investing.com "CPOc1",
CME's USD-denominated palm contract (~1/4 the MYR price, settlement-only
prints). Old files stay untouched in data/raw/fcpo/ but are EXCLUDED from the
merge. Mixing the two denominations corrupts every return downstream.

Run: python -m src.ingestion.fcpo
"""

import datetime as dt
import io

import pandas as pd
import requests

from src.utils.config import load_config, project_root
from src.utils.io import save_raw_csv

_MPOC = "https://www.mpoc.org.my/market-insight/daily-palm-oil-prices/"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_MYT = dt.timezone(dt.timedelta(hours=8))
# canonical MYR inputs, in merge order — LATER WINS on overlapping dates
_INPUTS = ["fcpo_tradingview_myr.csv", "fcpo_ibkr.csv", "fcpo_manual_myr.csv"]


# ── TradingView primary feed ──────────────────────────────────────────────────

def refresh_tradingview() -> pd.DataFrame | None:
    """Pull MYX:FCPO1! daily bars, map to trade dates, upsert the cache.

    tvdatafeed returns the whole history in one call, so every run self-heals
    past gaps. On any failure we keep serving from the existing cache."""
    cache = project_root() / "data" / "raw" / "fcpo" / "fcpo_tradingview_myr.csv"
    start = pd.Timestamp(load_config("data")["dates"]["start"])

    try:
        from tvDatafeed import TvDatafeed, Interval
        bars = TvDatafeed().get_hist(symbol="FCPO1!", exchange="MYX",
                                     interval=Interval.in_daily, n_bars=5000)
        if bars is None or bars.empty:
            raise RuntimeError("no bars returned")
    except Exception as e:
        print(f"[fcpo] TradingView fetch failed: {e}")
        print("       Continuing on the cached series (if any).")
        return None

    df = bars[["open", "high", "low", "close", "volume"]].copy()
    # bar stamp = session open on T-1 (see module docstring) -> trade date = +1 day
    df.index = pd.to_datetime(df.index).normalize() + pd.Timedelta(days=1)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= start]

    # drop the still-open session: its settlement prints 18:00 MYT on trade date
    now_myt = dt.datetime.now(_MYT)
    settled_through = pd.Timestamp(now_myt.date())
    if now_myt.hour < 18:
        settled_through -= pd.Timedelta(days=1)
    live = df.index > settled_through
    if live.any():
        print(f"[fcpo] dropping {live.sum()} unsettled session(s): "
              f"{', '.join(str(d.date()) for d in df.index[live])}")
        df = df[~live]

    if cache.exists():                       # never lose rows TV stops serving
        old = pd.read_csv(cache, parse_dates=["date"], index_col="date")
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_csv(cache)
    print(f"[fcpo] TradingView MYR series: {len(df):,} rows "
          f"({df.index[0].date()} -> {df.index[-1].date()}) -> {cache.name}")
    return df


# ── MPOC official settlement cross-check ─────────────────────────────────────

def _mpoc_settlements() -> pd.Series | None:
    """Recent official Bursa settlements. MPOC's 'Pricing Date' IS the trade
    date (verified by value alignment 2026-07-13) — no shift."""
    try:
        r = requests.get(_MPOC, headers={"User-Agent": _UA}, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        t = next(t for t in tables if "Settlement Price RM" in t.columns)
        return pd.Series(
            pd.to_numeric(t["Settlement Price RM"], errors="coerce").values,
            index=pd.to_datetime(t["Pricing Date"], format="%d %b %y"),
            name="mpoc_settlement",
        ).dropna().sort_index()
    except Exception as e:
        print(f"[fcpo] MPOC cross-check unavailable ({e}) — skipping validation")
        return None


def validate_vs_mpoc(close: pd.Series) -> None:
    """Warn loudly if our series disagrees with official settlements."""
    mpoc = _mpoc_settlements()
    if mpoc is None or mpoc.empty:
        return
    both = pd.concat([close.rename("ours"), mpoc], axis=1, join="inner").dropna()
    if both.empty:
        print("[fcpo] MPOC cross-check: no overlapping dates yet")
        return
    diff = (both["ours"] / both["mpoc_settlement"] - 1).abs()
    bad = both[diff > 0.005]
    if len(bad):
        print(f"[fcpo] WARNING — {len(bad)}/{len(both)} closes differ >0.5% from "
              f"official Bursa settlements (check date mapping / roll):")
        print(bad.assign(diff_pct=(diff[bad.index] * 100).round(2)).to_string())
    else:
        print(f"[fcpo] MPOC cross-check OK — {len(both)}/{len(both)} closes match "
              f"official settlements (incl. dates investing.com lacks)")


# ── canonical merge ───────────────────────────────────────────────────────────

def fetch_fcpo() -> pd.DataFrame | None:
    """Refresh the TradingView feed, then rebuild fcpo_raw.csv from the explicit
    MYR allowlist (later files win on overlap; output is never an input)."""
    raw_dir = project_root() / "data" / "raw" / "fcpo"

    refresh_tradingview()

    frames = []
    for name in _INPUTS:
        path = raw_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        df.columns = [c.strip().lower() for c in df.columns]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        frames.append(df[keep])
        print(f"[fcpo] merged {name}: {len(df):,} rows")

    if not frames:
        print("[fcpo] ERROR — no MYR input files at all. Run this module once with "
              "internet access, or add prices via fcpo_manual.py.")
        return None

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    # sanity guard: MYR FCPO has traded ~1,800-5,500 since 2015; anything near
    # ~1,100 is the old USD series sneaking back in through a stray file
    suspicious = merged["close"] < 1500
    if suspicious.any():
        print(f"[fcpo] WARNING — dropped {suspicious.sum()} rows with close < RM1,500 "
              f"(USD-denominated contamination?):")
        print(merged.loc[suspicious, "close"].tail().to_string())
        merged = merged[~suspicious]

    save_raw_csv(merged, raw_dir / "fcpo_raw.csv")
    validate_vs_mpoc(merged["close"])
    return merged


if __name__ == "__main__":
    fetch_fcpo()
