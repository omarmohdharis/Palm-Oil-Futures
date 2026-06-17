# FCPO Decision-Support — Runbook

A daily buy/sell/hold **signal service** for FCPO (Bursa Malaysia crude palm oil
futures). It is **decision support, not an auto-trader**: it emits an advisory
recommendation a human reviews. Run it on **paper first** and prove it live
(via monitoring) before risking capital.

> Honest status: on honest, no-peeking out-of-sample tests (2021–2026) the model
> shows net Sharpe ≈ 0.6 with ~half the drawdown of buy-and-hold — promising, but
> measured on one instrument over a calm-ish window with estimated costs. Treat
> it as a research-grade signal under live evaluation, not a proven system.

---

## 1. One-time setup

```bash
pip install -r requirements.txt
```

For the hands-off FCPO feed (optional but recommended):
1. Open a **free Interactive Brokers paper account**.
2. Install **IB Gateway** (or TWS); log into the **paper** account.
3. Enable the API: Configure → API → Settings → *Enable ActiveX and Socket Clients*.
4. Confirm the port in `config/serve.yaml` (`ibkr.port`): TWS paper = 7497, IB Gateway paper = 4002.
5. **Verify the connection & FCPO contract**: run `python -m src.ingestion.ibkr_check`.
   It confirms the API connection and searches IB for the real palm-oil contract
   symbology (it varies — don't guess). Correct `ibkr.contract` / `ibkr.port` in
   `config/serve.yaml` from its output, then `python -m src.ingestion.fcpo_ibkr`.

If you skip IBKR, the **drop-folder fallback** works: drop a daily FCPO CSV
(investing.com export or standard OHLCV) into `data/raw/fcpo/` and the pipeline
merges it automatically.

---

## 2. Freeze a model (once, then on a cadence)

```bash
python -m src.serving.freeze
```
Trains the GBM on all data, picks the live confidence threshold from past
out-of-sample history, and writes `models/gbm_<date>.pkl` + `models/current.json`.
**Re-run monthly** (or when monitoring flags decay).

Seed the monitor with a track record so you don't wait months:
```bash
python -m src.serving.monitor --backfill
```

---

## 3. Daily run (schedule this)

After Bursa close (~18:00 MYT):
```bash
python -m src.serving.serve            # refresh data → features → signal
python -m src.serving.serve --monitor  # also score live performance
```
Outputs:
- `results/dashboard.html` — **the decision tool**: open in any browser (self-contained)
- `results/signals/latest_signal.md` — today's recommendation in plain text
- `results/signals/signal_log.parquet` — every dated call (audit trail)
- `results/signals/monitor_report.md` — live performance + decay verdict

**Sharing it:** `dashboard.html` is a single self-contained file — email it, or to put
it online copy it into a `docs/` folder and enable GitHub Pages (note: `results/`
is gitignored, so move/commit the file deliberately if you want it hosted).

**Scheduling (Windows Task Scheduler):** create a daily task running
`python -m src.serving.serve --monitor` in this directory. On Linux/macOS use cron.

---

## 4. What each component does

| Module | Role |
|---|---|
| `src/ingestion/related.py` | yfinance feeds (Brent, soy, WTI, USD/MYR) — auto |
| `src/ingestion/fcpo_ibkr.py` | FCPO via IBKR paper account |
| `src/ingestion/fcpo.py` | merges/de-dupes FCPO CSVs → `fcpo_raw.csv` |
| `src/features/build_features.py` | builds the 25-feature matrix |
| `src/serving/freeze.py` | trains & versions the live model + threshold |
| `src/serving/predict.py` | today's signal (loads frozen model) |
| `src/serving/monitor.py` | live signals vs realised outcomes; decay alert |
| `src/serving/serve.py` | orchestrates the daily loop |

---

## 5. How to read a signal

- **Recommendation** is the gated call: a directional view below the confidence
  threshold is intentionally downgraded to **HOLD** (don't trade weak signals).
- **Confidence vs threshold** tells you how close it was.
- Horizon is ~3 trading days. Re-evaluate daily.

## 6. When to retrain / stop trusting it
- Monitor verdict shows **POSSIBLE DECAY** → re-freeze; if it persists, stop and review.
- A market regime change (new export policy, biodiesel mandate, supply shock) →
  expect degraded performance until retrained.
- MPOB monthly supply/demand is currently deferred; adding it (as event/surprise
  features) is a known future improvement.
