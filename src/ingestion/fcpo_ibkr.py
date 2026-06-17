"""
FCPO ingestion via Interactive Brokers (paper account) — the hands-off feed.

Replaces the manual investing.com download. Pulls recent daily FCPO bars through
the IB API and drops them as a CSV into data/raw/fcpo/, where the existing
fetch_fcpo() merge logic consolidates and de-duplicates them into fcpo_raw.csv.
So this module only has to fetch + write; downstream is unchanged.

PREREQUISITES (cannot be verified from here — must be done on your machine):
  1. Open a FREE Interactive Brokers PAPER account.
  2. Install & run IB Gateway or TWS, logged into the paper account, with the
     API enabled (Configure → API → Settings → "Enable ActiveX and Socket
     Clients"). Paper ports: TWS 7497, IB Gateway 4002 (set in config/serve.yaml).
  3. pip install ib_insync   (already in requirements.txt)

CONTRACT CAVEAT: IB symbology varies. This uses a continuous future
(ContFuture symbol/exchange/currency from config) and asks IB to qualify it. If
qualification returns nothing, the contract details in config/serve.yaml are
wrong — open TWS, look up the real FCPO contract, and correct them. Do NOT
assume the defaults are right; verify against a contract IB actually returns.

Run: python -m src.ingestion.fcpo_ibkr
"""

import datetime as dt

import pandas as pd

from src.utils.config import load_config, project_root


def fetch_fcpo_ibkr() -> pd.DataFrame | None:
    cfg = load_config("serve")["ibkr"]
    try:
        from ib_insync import IB, ContFuture, util
    except ImportError:
        print("[ibkr] ib_insync not installed — run: pip install ib_insync")
        return None

    ib = IB()
    try:
        print(f"[ibkr] connecting to {cfg['host']}:{cfg['port']} (clientId {cfg['client_id']}) …")
        ib.connect(cfg["host"], cfg["port"], clientId=cfg["client_id"], timeout=10)
    except Exception as e:
        print(f"[ibkr] could not connect: {e}")
        print("       Is IB Gateway/TWS running, logged into paper, with the API enabled?")
        print("       Falling back to the manual drop-folder workflow (data/raw/fcpo/*.csv).")
        return None

    try:
        c = cfg["contract"]
        contract = ContFuture(c["symbol"], c["exchange"], currency=c["currency"])
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"[ibkr] contract {c} did not resolve — verify symbol/exchange in TWS "
                  f"and fix config/serve.yaml. No data written.")
            return None

        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=cfg["lookback"],
            barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1)
        if not bars:
            print("[ibkr] no bars returned (market data permissions? contract?).")
            return None

        df = util.df(bars)[["date", "open", "high", "low", "close"]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date").sort_index()
    finally:
        ib.disconnect()

    out = project_root() / "data" / "raw" / "fcpo" / "fcpo_ibkr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"[ibkr] wrote {len(df)} daily bars ({df.index[0]} → {df.index[-1]}) → {out}")
    print("[ibkr] now run fetch_fcpo() (or serve.py) to merge into fcpo_raw.csv")
    return df


if __name__ == "__main__":
    fetch_fcpo_ibkr()
