"""
News sentiment via GDELT (free, no API key) — the short-term/anticipatory signal.

For each topic (palm oil, crude/biodiesel, soybean, palm-region weather) we pull
two daily series from GDELT's DOC 2.0 API:
  - tone   : average sentiment of matching news (negative = bearish mood)
  - volume : how much coverage the topic is getting (attention / fear)

── Why this failed in June 2026, and what changed ────────────────────────────
1. The old code requested the FULL 2017→today timeline for every topic on every
   run (8 heavy requests each time). GDELT quota-banned the network's IP.
   Now: the first successful run backfills history ONCE; every later run asks
   only for the last ~2 weeks per topic. A daily run is 8 tiny requests.
2. GDELT often throttles with HTTP 200 + a plain-text message (not 429). The
   old code treated any non-JSON 200 as fatal and gave up instantly.
   Now: 429 AND non-JSON 200 both count as throttling → exponential backoff.
3. A preflight probe (one request) checks whether this network is currently
   blocked before burning the per-topic requests into a wall.

If the preflight fails, the run exits gracefully — the pipeline continues
without sentiment (features are NaN-tolerant). The IP block is per-network:
running once from a phone hotspot / home connection is enough to backfill
history; afterwards the tiny daily increments matter much less.

Coverage starts ~2017. GDELT tone is a crude, broad proxy — a finance-specific
scorer (FinBERT on headlines) would be sharper but needs heavier infra.

Run: python -m src.ingestion.news_sentiment
"""

import time
import datetime as dt

import requests
import pandas as pd

from src.utils.config import load_config, project_root

_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_MIN_INTERVAL = 6.5          # seconds between requests (limit is 1 per 5s)
_BACKOFFS = [15, 45, 120]    # seconds after a throttle response
_INCREMENT_DAYS = 14         # re-fetch window once a topic has history
_last_call = [0.0]


def _throttle() -> None:
    wait = _MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _gdelt(query: str, mode: str, start: str, end: str) -> dict | None:
    """One GDELT call with backoff. Returns parsed JSON or None.

    GDELT signals throttling BOTH as HTTP 429 and as HTTP 200 with a plain-text
    body ("Please limit requests to one every 5 seconds...") — treat them the
    same. Any other failure is returned as None immediately."""
    params = {"query": query, "mode": mode, "format": "json",
              "startdatetime": start, "enddatetime": end}
    for attempt, back in enumerate([0] + _BACKOFFS):
        if back:
            print(f"  [news] throttled; backing off {back}s ...")
            time.sleep(back)
        _throttle()
        try:
            r = requests.get(_GDELT, params=params, timeout=60)
        except Exception as e:
            print(f"  [news] request failed: {e}")
            return None
        body = r.text.lstrip()
        if r.status_code == 200 and body.startswith("{"):
            return r.json()
        if r.status_code == 429 or (r.status_code == 200 and not body.startswith("{")):
            continue                                    # throttled → back off & retry
        print(f"  [news] {mode} HTTP {r.status_code}: {body[:80]}")
        return None
    print(f"  [news] still throttled after {len(_BACKOFFS)} backoffs — giving up on {mode}")
    return None


def preflight() -> bool:
    """One tiny request to see if this network is currently allowed in."""
    week_ago = (dt.date.today() - dt.timedelta(days=7)).strftime("%Y%m%d") + "000000"
    now = dt.date.today().strftime("%Y%m%d") + "235959"
    ok = _gdelt('"palm oil"', "timelinetone", week_ago, now) is not None
    if not ok:
        print("""
[news] GDELT is rate-limiting this network (shared/university IPs stay
       saturated by other users). Nothing is wrong with the code or query.

       To backfill history ONCE, run this from a different network
       (phone hotspot / home Wi-Fi):
           python -m src.ingestion.news_sentiment
       After that, daily increments are tiny and failures are harmless —
       the pipeline just runs without fresh sentiment that day.
""")
    return ok


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
    full_start = cfg["start"].replace("-", "") + "000000"
    end = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y%m%d") + "000000"
    out_dir = project_root() / "data" / "raw" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not preflight():
        return

    for name, query in cfg["topics"].items():
        path = out_dir / f"{name}.csv"
        old = None
        if path.exists():
            old = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
        if old is not None and len(old):
            # incremental: only re-fetch a recent window (GDELT revises little)
            inc_start = (old.index.max() - pd.Timedelta(days=_INCREMENT_DAYS))
            start = inc_start.strftime("%Y%m%d") + "000000"
            print(f"[news] {name}: incremental from {inc_start.date()}")
        else:
            start = full_start
            print(f"[news] {name}: FULL backfill from {cfg['start']} (first run)")

        df = fetch_topic(query, start, end)
        if df.empty:
            print("  no data returned")
            continue
        if old is not None:
            df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        df.to_csv(path)
        print(f"  saved {len(df)} days ({df.index.min().date()} -> {df.index.max().date()}) -> {path.name}")


if __name__ == "__main__":
    fetch_all()
