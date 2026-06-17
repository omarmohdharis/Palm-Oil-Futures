"""
Phase 8 — Standalone HTML dashboard (the deployable decision tool).

Turns the data we already have (frozen model registry, signal log, backtest
equity curve, live scoring) into ONE self-contained file: results/dashboard.html.
No server, no JavaScript, no external assets — open it in any browser, email it,
or drop it on GitHub Pages. Re-run after each daily loop to refresh it.

Run: python -m src.serving.dashboard
"""

import json
import datetime as dt

import numpy as np
import pandas as pd

from src.backtest.backtest import _fcpo_close
from src.serving.monitor import _score
from src.utils.config import load_config, project_root
from src.utils.io import load_parquet

_COL = {"BUY": "#1d9e75", "SELL": "#d8503a", "HOLD": "#6b7280"}


def _svg_lines(series: dict[str, tuple[str, pd.Series]], w=620, h=200, pad=8,
               legend=True) -> str:
    """Tiny dependency-free line chart from one or more equity series."""
    allvals = pd.concat([s for _, s in series.values()])
    lo, hi = float(allvals.min()), float(allvals.max())
    rng = (hi - lo) or 1.0
    paths = []
    for key, (color, s) in series.items():
        s = s.dropna()
        step = max(1, len(s) // 240)                 # downsample for a light file
        s = s.iloc[::step]
        n = len(s)
        pts = []
        for i, v in enumerate(s):
            x = pad + (w - 2 * pad) * (i / max(1, n - 1))
            y = pad + (h - 2 * pad) * (1 - (v - lo) / rng)
            pts.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                     f'points="{" ".join(pts)}"/>')
    leg = ""
    if legend:
        items = " &nbsp; ".join(
            f'<span style="color:{c}">&#9632;</span> {k}' for k, (c, _) in series.items())
        leg = f'<div class="muted" style="font-size:12px;margin-top:4px">{items}</div>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="line chart">{"".join(paths)}</svg>{leg}')


def _bar(label: str, pct: float, color: str) -> str:
    return (f'<div class="barrow"><span class="barlab">{label}</span>'
            f'<span class="bartrack"><span class="barfill" '
            f'style="width:{pct*100:.0f}%;background:{color}"></span></span>'
            f'<span class="barval">{pct:.0%}</span></div>')


def _price_panel() -> str:
    """Current FCPO price, recent changes, and a price chart (last ~1 year)."""
    close = _fcpo_close().sort_index()
    latest = float(close.iloc[-1])

    def chg(n):
        return close.iloc[-1] / close.iloc[-1 - n] - 1 if len(close) > n else float("nan")

    def tag(c, label):
        col = _COL["BUY"] if c >= 0 else _COL["SELL"]
        return (f'<div style="margin-right:22px"><span style="color:{col};font-weight:700">'
                f'{c:+.1%}</span><div class="k">{label}</div></div>')

    window = close.iloc[-252:]
    chart = _svg_lines({"price": ("#534ab7", window)}, h=170, legend=False)
    return (f'<div class="card" style="grid-column:1/-1"><h2>Palm oil price (RM per tonne)</h2>'
            f'<div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap">'
            f'<div style="font-size:30px;font-weight:800">{latest:,.0f}</div>'
            f'<div style="display:flex">{tag(chg(1), "1 day")}{tag(chg(5), "1 week")}'
            f'{tag(chg(21), "1 month")}</div></div>{chart}'
            f'<div class="muted" style="font-size:11.5px;margin-top:4px">'
            f'Last ~1 year of the front-month futures price, through {close.index[-1].date()}.</div></div>')


def _forecast_panel() -> str:
    """Short-horizon (1-2 day) direction calls + their running accuracy, if present."""
    cfg = load_config("serve").get("forecast", {})
    path = project_root() / cfg.get("log", "results/signals/forecast_log.parquet")
    if not path.exists():
        return ""
    log = load_parquet(path)
    close = _fcpo_close().sort_index()
    tgt = close.reindex(pd.to_datetime(log["target_date"]))
    aret = tgt.values / log["close_at_forecast"].values - 1.0
    log = log.assign(actual_dir=np.where(aret > 0, "UP", np.where(aret < 0, "DOWN", None)))

    cells = []
    for h in cfg.get("horizons", [1, 2]):
        sub = log[log["horizon"] == h]
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        scored = sub.dropna(subset=["actual_dir"])
        acc = (scored["predicted_dir"] == scored["actual_dir"]).mean() if len(scored) else float("nan")
        col = _COL["BUY"] if latest["predicted_dir"] == "UP" else _COL["SELL"]
        arrow = "&#9650;" if latest["predicted_dir"] == "UP" else "&#9660;"
        cells.append(
            f'<div style="flex:1;min-width:150px"><div class="k">next {h} '
            f'day{"s" if h > 1 else ""}</div>'
            f'<div style="font-size:20px;font-weight:700;color:{col}">{arrow} '
            f'{latest["predicted_dir"]}</div>'
            f'<div class="k">{latest["confidence"]:.0%} sure · '
            f'right {acc:.0%} of the time so far</div></div>')

    return (f'<div class="card" style="grid-column:1/-1"><h2>Quick look: next 1–2 days</h2>'
            f'<div style="display:flex;gap:24px;flex-wrap:wrap">{"".join(cells)}</div>'
            f'<div class="muted" style="font-size:11.5px;margin-top:8px">A short-term guess of '
            f'direction only — <b>less reliable</b> than the main suggestion above (1–2 day moves '
            f'are mostly noise). Accuracy updates automatically as new prices come in; 50% = a coin flip.</div></div>')


def build_dashboard(publish: bool = False) -> None:
    reg = json.loads((project_root() / load_config("serve")["model"]["registry"]).read_text())
    log = load_parquet(project_root() / load_config("serve")["signals"]["log"]).sort_index()
    r = log.iloc[-1]
    sig = r["final_signal"]

    # live track record (scored signals)
    scored = _score(log)
    trades = scored[scored["position"] != 0]
    hit = trades["correct"].mean() if len(trades) else float("nan")
    avg = trades["realised_pnl"].mean() if len(trades) else float("nan")
    cum = trades["realised_pnl"].sum() if len(trades) else float("nan")

    # backtest equity vs buy & hold
    bt = load_parquet(project_root() / "results" / "phase7" / "backtest.parquet")
    eq, bh = bt["equity"], bt["equity_buyhold"]
    dd = (eq / eq.cummax() - 1).min()
    bh_dd = (bh / bh.cummax() - 1).min()
    chart = _svg_lines({"strategy (net)": (_COL["BUY"], eq),
                        "buy & hold": ("#9aa0a6", bh)})

    recent = log[["raw_signal", "confidence", "final_signal"]].tail(12).iloc[::-1]
    rows = "".join(
        f'<tr><td>{i.date()}</td><td>{x.raw_signal}</td>'
        f'<td>{x.confidence:.0%}</td>'
        f'<td><b style="color:{_COL[x.final_signal]}">{x.final_signal}</b></td></tr>'
        for i, x in recent.iterrows())

    gated = "" if sig == r["raw_signal"] else (
        f'<div class="note">The tool leaned toward <b>{r["raw_signal"]}</b>, but at only '
        f'{r["confidence"]:.0%} it isn\'t sure enough (it waits for {r["threshold"]:.0%}), '
        f'so the safer suggestion is to <b>hold</b> for now.</div>')

    fc = _forecast_panel()
    price = _price_panel()

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Palm Oil Futures — Buy, Sell or Hold?</title><style>
:root{{--bg:#f4f4f1;--card:#fff;--ink:#1a1a1a;--muted:#6b6b6b;--line:#e3e3df}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif;padding:24px}}
.wrap{{max-width:980px;margin:0 auto}}h1{{font-size:22px;margin:0 0 2px}}
.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:1fr 1fr;
gap:16px;margin-top:16px}}.card{{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:18px}}.reco{{grid-column:1/-1;display:flex;
align-items:center;gap:20px}}.badge{{font-size:30px;font-weight:800;padding:14px 26px;
border-radius:12px;color:#fff}}.k{{font-size:12px;color:var(--muted)}}
.v{{font-size:15px;font-weight:600}}table{{width:100%;border-collapse:collapse;
font-size:13px}}td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:600}}.barrow{{display:flex;align-items:center;
gap:10px;margin:7px 0;font-size:13px}}.barlab{{width:44px}}.bartrack{{flex:1;height:9px;
background:var(--line);border-radius:5px;overflow:hidden}}.barfill{{display:block;height:100%}}
.barval{{width:38px;text-align:right}}.metrics{{display:flex;gap:26px;flex-wrap:wrap}}
.metric .big{{font-size:22px;font-weight:700}}.note{{background:#fff8e6;
border:1px solid #f0e0a8;border-radius:8px;padding:10px 12px;font-size:13px;margin-top:12px}}
.disc{{font-size:12px;color:var(--muted);margin-top:18px;line-height:1.5}}
h2{{font-size:14px;margin:0 0 10px}}.explain{{font-size:13px;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:12px;line-height:1.55}}</style></head>
<body><div class="wrap">
<h1>Palm Oil Futures — Buy, Sell or Hold?</h1>
<div class="muted">A plain-English suggestion for the next few days. Updated {dt.datetime.now():%Y-%m-%d %H:%M}.</div>
<div class="explain">This tool studies how palm oil prices and related markets (like soybean oil and crude oil)
have moved together, and suggests whether to <b>buy</b> (it expects the price to rise),
<b>sell</b> (expects it to fall), or <b>hold</b> (no clear move — better to wait). It looks about 3 days ahead.</div>

<div class="grid">
  <div class="card reco">
    <span class="badge" style="background:{_COL[sig]}">{sig}</span>
    <div>
      <div class="k">Suggestion for {r.name.date()} &nbsp;|&nbsp; last price {r['fcpo_close']:.0f}</div>
      <div class="v">How sure the tool is: {r['confidence']:.0%} &nbsp;(it only acts above {r['threshold']:.0%})</div>
      {gated}
    </div>
  </div>

  {price}

  {fc}

  <div class="card"><h2>What the tool expects (next few days)</h2>
    {_bar("Up", float(r['p_BUY']), _COL['BUY'])}
    {_bar("Flat", float(r['p_HOLD']), _COL['HOLD'])}
    {_bar("Down", float(r['p_SELL']), _COL['SELL'])}
    <div class="muted" style="font-size:12px;margin-top:8px">
      The estimated chance of each outcome. Higher bar = more likely.</div>
  </div>

  <div class="card"><h2>How it has done in the past</h2>
    <div class="metrics">
      <div class="metric"><div class="big">{hit:.0%}</div><div class="k">of {len(trades)} past calls were right</div></div>
      <div class="metric"><div class="big">{avg:+.2%}</div><div class="k">average move per call</div></div>
      <div class="metric"><div class="big">{cum:+.0%}</div><div class="k">total if you'd followed it*</div></div>
    </div>
    <div class="muted" style="font-size:11.5px;margin-top:8px">*Before trading fees. Past results don't guarantee the future.</div>
  </div>

  <div class="card" style="grid-column:1/-1"><h2>Following the tool vs. just holding (tested on past data)</h2>
    {chart}
    <div class="muted" style="font-size:12px;margin-top:6px">
      Worst drop along the way: following the tool {dd:.0%} vs. just holding {bh_dd:.0%}.
      The tool's main strength is losing much less when the market falls.</div>
  </div>

  <div class="card" style="grid-column:1/-1"><h2>Recent suggestions</h2>
    <table><tr><th>Date</th><th>Leaning</th><th>How sure</th><th>Suggestion</th></tr>
    {rows}</table>
  </div>
</div>

<div class="disc"><b>This is an educational tool, not financial advice.</b> It is still being
tested and can be wrong. Please don't trade real money based on it without doing your own
research. The numbers use data up to {r.name.date()}. When the tool isn't confident, it suggests holding.</div>
</div></body></html>"""

    out = project_root() / "results" / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[dashboard] {sig} as of {r.name.date()} → {out}")
    print(f"[dashboard] open it: file:///{str(out).replace(chr(92), '/')}")

    if publish:                                    # git-tracked copy for GitHub Pages
        pub = project_root() / "docs" / "index.html"
        pub.parent.mkdir(parents=True, exist_ok=True)
        pub.write_text(html, encoding="utf-8")
        print(f"[dashboard] published copy → {pub}  (commit & push to update Pages)")


if __name__ == "__main__":
    import sys
    build_dashboard(publish="--publish" in sys.argv)
