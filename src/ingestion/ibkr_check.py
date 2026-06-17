"""
IBKR setup diagnostic — run this ONCE after IB Gateway/TWS is up.

It does three things and prints what it finds:
  1. confirms the API connection works (host/port/clientId from config/serve.yaml),
  2. searches IB for palm-oil / FCPO instruments so you can see the REAL contract
     symbology (it varies — don't guess),
  3. lists the available FCPO futures expiries and tests the continuous future.

Use the output to correct `ibkr.contract` in config/serve.yaml if needed, then
run `python -m src.ingestion.fcpo_ibkr` to pull data for real.

Run: python -m src.ingestion.ibkr_check
"""

from src.utils.config import load_config


def main() -> None:
    cfg = load_config("serve")["ibkr"]
    try:
        from ib_async import IB, Future, ContFuture
    except ImportError:
        from ib_insync import IB, Future, ContFuture

    ib = IB()
    try:
        print(f"[check] connecting to {cfg['host']}:{cfg['port']} (clientId {cfg['client_id']}) …")
        ib.connect(cfg["host"], cfg["port"], clientId=cfg["client_id"], timeout=10)
    except Exception as e:
        print(f"[check] CONNECTION FAILED: {e}")
        print("        → Is IB Gateway/TWS running, logged into PAPER, API enabled,")
        print(f"          and listening on port {cfg['port']}? (TWS 7497 / Gateway 4002)")
        return
    print(f"[check] connected ✓  (server version {ib.client.serverVersion()})\n")

    # 1 — symbol search: what does IB call palm oil?
    print("[check] searching IB symbols for 'palm' and 'FCPO' …")
    for term in ("palm", "FCPO"):
        try:
            matches = ib.reqMatchingSymbols(term)
            for m in (matches or [])[:8]:
                c = m.contract
                print(f"   {term:>5}: symbol={c.symbol!r:10} secType={c.secType:6} "
                      f"exch={c.primaryExchange or c.exchange!r}  desc={getattr(m, 'description', '')}")
        except Exception as e:
            print(f"   {term}: search failed ({e})")

    # 2 — list FCPO future expiries on the configured exchange
    c = cfg["contract"]
    print(f"\n[check] futures matching symbol={c['symbol']!r} exchange={c['exchange']!r}:")
    try:
        details = ib.reqContractDetails(Future(c["symbol"], exchange=c["exchange"], currency=c["currency"]))
        if not details:
            print("   none — symbol/exchange likely wrong; use the symbol search above.")
        for d in details[:10]:
            k = d.contract
            print(f"   localSymbol={k.localSymbol!r:10} expiry={k.lastTradeDateOrContractMonth} "
                  f"conId={k.conId} multiplier={k.multiplier}")
    except Exception as e:
        print(f"   contract details failed ({e})")

    # 3 — test the continuous future the data feed will use
    print(f"\n[check] qualifying ContFuture({c['symbol']}, {c['exchange']}, {c['currency']}) …")
    try:
        cf = ContFuture(c["symbol"], c["exchange"], currency=c["currency"])
        q = ib.qualifyContracts(cf)
        print(f"   {'RESOLVED ✓ — fcpo_ibkr is good to go' if q else 'did NOT resolve — fix config/serve.yaml'}")
        if q:
            print(f"   {q[0]}")
    except Exception as e:
        print(f"   qualify failed ({e})")

    ib.disconnect()
    print("\n[check] done.")


if __name__ == "__main__":
    main()
