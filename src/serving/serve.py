"""
Phase 8 — Daily serving orchestrator.

Runs the production loop end to end, in order:
  1. refresh related instruments (yfinance)
  2. refresh FCPO via IBKR, then merge all data/raw/fcpo/ CSVs (drop-folder
     fallback works automatically if IBKR is unavailable)
  3. rebuild the feature matrix
  4. predict today's signal (loads the frozen model; never retrains)
  5. (weekly / on demand) monitor live performance vs realised outcomes

Each data refresh is best-effort: a failure is logged and the loop continues on
whatever data is on disk, so a flaky feed never blocks the signal.

Schedule this after Bursa close (~18:00 MYT) with Windows Task Scheduler / cron.
The model is refrozen separately and far less often — see freeze.py.

Run: python -m src.serving.serve            # daily signal
     python -m src.serving.serve --monitor  # also run monitoring
"""

import sys


def _step(name, fn):
    try:
        fn()
    except Exception as e:                       # never let one source kill the run
        print(f"[serve] step '{name}' failed (continuing): {e}")


def serve(run_monitor: bool = False) -> None:
    from src.ingestion.related import fetch_related
    from src.ingestion.fcpo import fetch_fcpo
    from src.ingestion.fcpo_ibkr import fetch_fcpo_ibkr
    from src.features.build_features import build_features
    from src.serving.predict import predict
    from src.serving.dashboard import build_dashboard

    print("=" * 60)
    print("FCPO daily serving loop")
    print("=" * 60)

    _step("refresh related (yfinance)", fetch_related)
    _step("refresh FCPO (IBKR)", fetch_fcpo_ibkr)
    _step("refresh FCPO (TradingView + merge)", fetch_fcpo)
    from src.ingestion.news_sentiment import fetch_all as fetch_news
    _step("refresh news sentiment (GDELT)", fetch_news)   # fails soft when IP-blocked
    _step("build features", build_features)

    # prediction is the deliverable — if it fails we want to see it loudly
    predict()

    from src.serving.forecast import make_forecasts, score
    _step("short-horizon forecast", make_forecasts)
    _step("score forecasts", score)

    if run_monitor:
        from src.serving.monitor import monitor
        print("\n" + "=" * 60)
        monitor()

    _step("refresh dashboard", build_dashboard)    # regenerate results/dashboard.html


if __name__ == "__main__":
    serve(run_monitor="--monitor" in sys.argv)
