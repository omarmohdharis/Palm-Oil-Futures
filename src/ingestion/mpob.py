"""
Malaysian palm oil supply & demand — two free public sources (no login required).

Source 1 — MPOB monthly Excel files (manual download):
  Monthly production, exports and closing stocks for Malaysian palm oil.
  Download from bepi.mpob.gov.my and save as data/raw/mpob/mpob_YYYY.xlsx
  (one file per year, or one combined file — the loader scans them all).
  This is genuine MONTHLY data — the market-moving release for FCPO.

Source 2 — World Bank Pink Sheet:
  CPO spot price in USD/tonne, converted to MYR via the USD/MYR FX rate
  saved by fetch_related().  Run fetch_related() before this script.

Output: data/processed/mpob_monthly_clean.parquet
Fields:
  production_kt      : 000 metric tonnes  (MPOB monthly)
  exports_kt         : 000 metric tonnes
  closing_stocks_kt  : 000 metric tonnes
  avg_cpo_price_usd  : USD/tonne          (World Bank)
  avg_cpo_price_myr  : USD × USD/MYR rate

NOTE: the MPOB parser column/row keywords (_MPOB_FIELDS) are written against
the standard BEPI layout, but BEPI sheets vary by year.  The loader prints
exactly what it finds so the keywords can be tuned to a real downloaded file.
"""

import re
import io
import datetime as dt
import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.utils.config import load_config, project_root
from src.utils.io import save_parquet

# Headline MPOB report rows we extract → output column. Verified against the real
# "monthly closing stock" download. Add production/export labels here once those
# files are seen (same report layout, different target rows).
_MPOB_TARGETS = {
    "TOTAL PALM OIL": "palm_oil_stocks",         # stocks report — headline closing stock
    "TOTAL CRUDE PALM OIL": "cpo_stocks",        # stocks report
    "MALAYSIA": "cpo_production",                 # production report — national total
}

_MONTHS = {
    m.lower(): i for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)
}

_WB_MARKETS_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
_WB_KNOWN_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)


# ── MPOB monthly Excel ────────────────────────────────────────────────────────

def _month_num(val) -> int | None:
    """Month number (1-12) from a month name/abbrev cell, else None."""
    if not isinstance(val, str):
        return None
    return _MONTHS.get(val.strip().lower()[:3])


def _parse_mpob_file(path, targets=None) -> pd.DataFrame | None:
    """
    Parse one MPOB BEPI report workbook into a monthly frame indexed by date.

    Verified against the real "monthly closing stock" download (2026-06). The
    sheet holds two half-year blocks (Jan-Jun, Jul-Dec); each block has a row of
    month names, a row of YEARS just below it, then product rows. Every month
    spans TWO columns (prior year, current year), and unreported future months are
    zero-filled. For each row named in `targets` we read every (year, month) value
    and drop the zero placeholders.
    """
    targets = {k.upper(): v for k, v in (targets or _MPOB_TARGETS).items()}
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        print(f"  [WARN] cannot open {path.name}: {e}")
        return None

    recs: dict[str, dict[pd.Timestamp, float]] = {}
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        for i in range(len(raw)):
            month_cols = {c: m for c, v in raw.iloc[i].items() if (m := _month_num(v))}
            if len(month_cols) < 3:                       # not a month-header row
                continue
            year_row = raw.iloc[i + 1]                    # years sit right below
            for j in range(i + 2, len(raw)):
                row_j = raw.iloc[j]
                if sum(1 for v in row_j if _month_num(v)) >= 3:   # next block started
                    break
                field = targets.get(str(row_j.iloc[0]).strip().upper())
                if not field:
                    continue
                for c, month in month_cols.items():
                    for col in (c, c + 1):                # prior-year / current-year columns
                        yr = pd.to_numeric(year_row.get(col), errors="coerce")
                        val = pd.to_numeric(row_j.get(col), errors="coerce")
                        if pd.notna(yr) and 1990 <= yr <= 2100 and pd.notna(val) and val > 0:
                            recs.setdefault(field, {})[pd.Timestamp(int(yr), month, 1)] = float(val)

    if not recs:
        print(f"  [WARN] {path.name} — none of {list(targets.values())} found. "
              f"Sheets: {xl.sheet_names}. Share the layout to extend the parser.")
        return None

    df = pd.DataFrame(recs).sort_index()
    df.index.name = "date"
    print(f"  parsed {path.name}: {len(df)} months "
          f"({df.index.min().date()} → {df.index.max().date()}); fields {list(df.columns)}")
    return df


def _parse_exports_file(path) -> pd.DataFrame | None:
    """
    Parse an MPOB monthly EXPORTS workbook (a different, single-year layout):
    one header row of month names (JAN..DEC) across the columns, then product rows
    each split into a 'Tonnes' and an 'RM Mil' sub-row. We take PALM OIL (Tonnes)
    for each month of the file's year. Future months are zero-filled and skipped.
    """
    try:
        raw = pd.ExcelFile(path, engine="openpyxl").parse(0, header=None)
    except Exception as e:
        print(f"  [WARN] cannot open {path.name}: {e}")
        return None

    text = " ".join(str(v) for v in raw.iloc[0] if pd.notna(v)) + " " + path.stem
    ym = re.search(r"(19|20)\d{2}", text)
    if not ym:
        print(f"  [WARN] {path.name} — no year found in title/filename; skipping.")
        return None
    year = int(ym.group(0))

    header_i, month_cols = None, {}
    for i in range(len(raw)):
        # months are exact tokens (JAN, JUNE…); the 'JAN - MAY' total has spaces so len>4 excludes it
        mc = {c: m for c, v in raw.iloc[i].items()
              if isinstance(v, str) and len(v.strip()) <= 4 and (m := _month_num(v))}
        if len(mc) >= 6:
            header_i, month_cols = i, mc
            break
    if header_i is None:
        print(f"  [WARN] {path.name} — no month-header row found.")
        return None

    recs: dict[pd.Timestamp, float] = {}
    for j in range(header_i + 1, len(raw)):
        if (str(raw.iat[j, 0]).strip().upper() == "PALM OIL"
                and str(raw.iat[j, 1]).strip().lower() == "tonnes"):
            row_j = raw.iloc[j]
            for c, month in month_cols.items():
                val = pd.to_numeric(row_j.get(c), errors="coerce")
                if pd.notna(val) and val > 0:
                    recs[pd.Timestamp(year, month, 1)] = float(val)
            break
    if not recs:
        print(f"  [WARN] {path.name} — 'PALM OIL / Tonnes' row not found.")
        return None

    df = pd.DataFrame({"palm_oil_exports": recs}).sort_index()
    df.index.name = "date"
    print(f"  parsed {path.name}: {len(df)} months "
          f"({df.index.min().date()} → {df.index.max().date()}); fields ['palm_oil_exports']")
    return df


def fetch_mpob_monthly() -> pd.DataFrame | None:
    from functools import reduce

    raw_dir = project_root() / "data" / "raw" / "mpob"
    files = sorted(raw_dir.glob("*.xls*"))
    if not files:
        print("[→] MPOB monthly — no files found in data/raw/mpob/")
        _print_mpob_instructions()
        return None

    print(f"[→] MPOB monthly — parsing {len(files)} file(s) …")
    frames = []
    for f in files:                                  # route by report type
        parser = _parse_exports_file if "export" in f.name.lower() else _parse_mpob_file
        df = parser(f)
        if df is not None:
            frames.append(df)
    if not frames:
        return None

    # merge on date across metrics/years (union of columns, fill gaps)
    combined = reduce(lambda a, b: a.combine_first(b), frames).sort_index()

    out = project_root() / "data" / "processed" / "mpob_monthly.parquet"
    save_parquet(combined, out)
    print(f"[OK] MPOB monthly: {len(combined)} months × {list(combined.columns)} → {out}")
    return combined


# ── World Bank Pink Sheet ─────────────────────────────────────────────────────

def _parse_wb_date(val) -> pd.Timestamp:
    """
    The Pink Sheet date column uses 'Jan-70' / 'Feb-80' style strings
    (month abbreviation + 2-digit year).  Handle that plus datetime objects
    and Excel serial numbers as fallbacks.
    """
    if isinstance(val, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(val)
    s = str(val).strip()
    # "MMM-YY"  e.g.  "Jan-70"  →  1970-01-01
    # "MMM-YYYY" e.g. "Jan-1970" →  1970-01-01
    for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
        try:
            return pd.Timestamp(dt.datetime.strptime(s, fmt))
        except ValueError:
            pass
    # "YYYY Mnn" e.g. "2020 M01"
    m = re.match(r"(\d{4})\s*M(\d{1,2})$", s)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    # Excel serial number (float stored as text or actual float)
    try:
        num = float(s)
        if 10_000 < num < 60_000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(num))
    except (ValueError, TypeError):
        pass
    return pd.NaT


def _wb_download_bytes() -> bytes | None:
    for url in [_WB_KNOWN_URL, _scrape_wb_url()]:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 10_000:
                return r.content
        except Exception:
            continue
    return None


def _scrape_wb_url() -> str | None:
    try:
        r = requests.get(_WB_MARKETS_PAGE, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text().lower()
            if href.endswith(".xlsx") and ("historical" in text or "monthly" in text):
                return href if href.startswith("http") else "https://www.worldbank.org" + href
    except Exception:
        pass
    return None


def _parse_pink_sheet(content: bytes) -> pd.Series | None:
    xl = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    sheet = next((s for s in xl.sheet_names if "month" in s.lower()), xl.sheet_names[0])
    raw = xl.parse(sheet, header=None)

    # Find the header row containing "Palm oil"
    header_row = None
    for i, row in raw.iterrows():
        if any("palm oil" in str(v).lower() for v in row):
            header_row = i
            break

    if header_row is None:
        print("  [WARN] 'Palm oil' column not found in Pink Sheet — sheet layout may have changed")
        print(f"  Sheet names available: {xl.sheet_names}")
        print(f"  First few rows of '{sheet}':")
        print(raw.head(10).to_string())
        return None

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(v).strip() for v in raw.iloc[header_row]]

    date_col = df.columns[0]
    cpo_col  = next((c for c in df.columns if "palm oil" in c.lower()), None)
    if not cpo_col:
        return None

    s = df[[date_col, cpo_col]].copy()
    s.columns = ["date", "avg_cpo_price_usd"]
    s["date"] = s["date"].apply(_parse_wb_date)
    s = s.dropna(subset=["date"])
    s["avg_cpo_price_usd"] = pd.to_numeric(s["avg_cpo_price_usd"], errors="coerce")
    s = s.dropna(subset=["avg_cpo_price_usd"])
    return s.set_index("date").sort_index()["avg_cpo_price_usd"]


def fetch_wb_cpo_price() -> pd.Series | None:
    print("[→] World Bank Pink Sheet — CPO price (USD/tonne) …")
    local = project_root() / "data" / "raw" / "mpob" / "wb_pink_sheet.xlsx"
    if local.exists():
        print(f"  using local file {local.name}")
        content = local.read_bytes()
    else:
        content = _wb_download_bytes()

    if content is None:
        _print_wb_instructions()
        return None

    series = _parse_pink_sheet(content)
    if series is not None:
        print(f"  {len(series)} monthly price rows ({series.index[0].date()} → {series.index[-1].date()})")
    return series


def _usd_to_myr(price_usd: pd.Series) -> pd.Series:
    fx_path = project_root() / "data" / "raw" / "related" / "usdmyr.csv"
    if not fx_path.exists():
        print("  [WARN] usdmyr.csv not found — run fetch_related() first; skipping MYR conversion")
        return pd.Series(dtype=float, name="avg_cpo_price_myr")
    fx = pd.read_csv(fx_path, index_col=0, parse_dates=True)["close"]
    fx_monthly = fx.resample("MS").mean()
    converted = price_usd.multiply(fx_monthly).dropna()
    converted.name = "avg_cpo_price_myr"
    return converted


# ── public entry point ────────────────────────────────────────────────────────

def fetch_supply_demand() -> pd.DataFrame | None:
    cfg      = load_config("data")
    out_path = project_root() / cfg["paths"]["processed"] / "mpob_monthly_clean.parquet"

    sd    = fetch_mpob_monthly()
    price = fetch_wb_cpo_price()

    if sd is None and price is None:
        print("[ERROR] both sources failed — cannot build supply/demand parquet")
        return None

    combined = sd.copy() if sd is not None else pd.DataFrame()

    if price is not None:
        combined["avg_cpo_price_usd"] = price
        combined["avg_cpo_price_myr"] = _usd_to_myr(price)

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined.dropna(how="all")

    save_parquet(combined, out_path)
    print(f"[OK] supply & demand: {len(combined)} months → {out_path}")
    return combined


def _print_mpob_instructions() -> None:
    print("""
  How to get MPOB monthly supply/demand data:
    1. Go to: bepi.mpob.gov.my  (Economic & Industry Development Division)
    2. Open the monthly statistics for Palm Oil:
         production, exports, and closing stocks
    3. Download the table(s) as Excel
    4. Save as: data/raw/mpob/mpob_YYYY.xlsx   (one file per year)
    5. Re-run — every mpob_*.xlsx in that folder is parsed automatically

  Tip: download ONE file first.  If the layout isn't recognised the script
  prints the sheet structure so the parser keywords can be tuned to it.
""")


def _print_wb_instructions() -> None:
    print("""
  Manual fallback for World Bank Pink Sheet:
    1. Go to: worldbank.org → Research → Commodity Markets
    2. Download "Monthly Prices" Excel
    3. Save as: data/raw/mpob/wb_pink_sheet.xlsx
    4. Re-run — auto-detected on next run
""")


if __name__ == "__main__":
    fetch_supply_demand()
