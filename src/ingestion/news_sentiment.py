"""
News sentiment via GDELT (free, no API key) — the short-term/anticipatory signal.

Commodity markets are forward-looking: they price expected events, then "square
off" once the event lands. Sentiment captures that anticipation. For each topic
(palm oil, crude/biodiesel, soybean, palm-region weather) we pull two free daily
series from GDELT's DOC 2.0 API:
  - tone   : average sentiment of matching news (negative = bearish mood)
  - volume : how much coverage the topic is getting (attention / fear)

GDELT rate limit is 1 request / 5 seconds, so requests are throttled (>=6s) and
retried with backoff. Series are cached per topic and updated incrementally, so
a daily run makes only a handful of small requests.

Coverage starts ~2017; earlier dates are simply absent (the model tolerates the
gaps). Tone is a crude, broad proxy — a finance-specific scorer (FinBERT/LLM on
headlines) would be sharper but needs heavier infra; GDELT is the free, historical,
automatable option.

Run: python -m src.ingestion.news_sentiment
"""

import time
import datetime as dt

import requests
import pandas as pd

from src.utils.config import load_config, project_root

_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_MIN_INTERVAL = 6.0          # seconds between requests (limit is 1 per 5s)
_last_call = [0.0]


def _throttle() -> None:
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _gdelt(query: str, mode: str, start: str, end: str, retries: int = 4) -> dict | None:
    params = {"query": query, "mode": mode, "format": "json",
              "startdatetime": start, "enddatetime": end}
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(_GDELT, params=params, timeout=60)
        except Exception as e:
            print(f"  [news] request failed: {e}")
            return None
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            return r.json()
        if r.status_code == 429:
            back = 10 * (attempt + 1)
            print(f"  [news] rate-limited; backing off {back}s …")
            time.sleep(back)
            continue
        print(f"  [news] {mode} HTTP {r.status_code}: {r.text[:80].strip()}")
        return None
    print(f"  [news] gave up on {mode} after {retries} tries (still rate-limited)")
    return None


def _to_series(js: dict | None, name: str) -> pd.Series:
    """Pull the {date, value} points out of a GDELT timeline response."""
    if not js or not js.get("timeline"):
        return pd.Series(dtype=float, name=name)
    data = js["timeline"][0]["data"]
    idx = pd.to_datetime([d["date"] for d in data], utc=True, errors="coerce")
    s = pd.Series([float(d["value"]) for d in data], index=idx, name=name)
    s = s[~s.index.isna()]
    s.index = s.index.tz_localize(None).normalize()
    return s


def fetch_topic(query: str, start: str, end: str) -> pd.DataFrame:
    tone = _to_series(_gdelt(query, "timelinetone", start, end), "tone")
    vol = _to_series(_gdelt(query, "timelinevol", start, end), "volume")
    df = pd.concat([tone, vol], axis=1).sort_index()
    df.index.name = "date"
    return df


def fetch_all() -> None:
    cfg = load_config("data")["news"]
    start = cfg["start"].replace("-", "") + "000000"
    end = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d") + "000000"
    out_dir = project_root() / "data" / "raw" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, query in cfg["topics"].items():
        print(f"[news] {name}: {query}")
        df = fetch_topic(query, start, end)
        if df.empty:
            print("  no data returned")
            continue
        path = out_dir / f"{name}.csv"
        if path.exists():                       # merge with cache (incremental)
            old = pd.read_csv(path, parse_dates=[0], index_col=0)
            df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        df.to_csv(path)
        print(f"  saved {len(df)} days ({df.index.min().date()} → {df.index.max().date()}) → {path}")


if __name__ == "__main__":
    fetch_all()
