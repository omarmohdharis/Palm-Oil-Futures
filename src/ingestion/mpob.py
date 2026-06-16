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

# Keyword → output column.  Matched case-insensitively against MPOB row labels.
_MPOB_FIELDS = {
    "production_kt":     ["production"],
    "exports_kt":        ["export"],
    "closing_stocks_kt": ["closing stock", "stock"],
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

def _coerce_month(val) -> int | None:
    """Map a cell to a month number (1-12) from a name, abbrev, or number."""
    if isinstance(val, (int, float)) and 1 <= int(val) <= 12:
        return int(val)
    s = str(val).strip().lower()[:3]
    return _MONTHS.get(s)


def _parse_mpob_file(path) -> pd.DataFrame | None:
    """
    Parse one MPOB BEPI workbook into a monthly frame indexed by date,
    with columns production_kt / exports_kt / closing_stocks_kt.

    BEPI layouts vary, so this scans every sheet, finds rows whose first
    cells match the _MPOB_FIELDS keywords, and reads the 12 monthly values
    across the row.  When it cannot make sense of a sheet it prints the
    layout so the keywords/orientation can be adjusted to the real file.
    """
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:
        print(f"  [WARN] cannot open {path.name}: {e}")
        return None

    year = _year_from_name(path.name)
    found: dict[str, dict[int, float]] = {f: {} for f in _MPOB_FIELDS}

    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None)
        # locate a row of month headers to map column index → month
        month_cols = _find_month_columns(raw)
        if not month_cols:
            continue
        for _, row in raw.iterrows():
            label = " ".join(str(v) for v in row[:2] if pd.notna(v)).lower()
            for field, keys in _MPOB_FIELDS.items():
                if found[field]:
                    continue
                if any(k in label for k in keys):
                    for col_idx, month in month_cols.items():
                        val = pd.to_numeric(row.get(col_idx), errors="coerce")
                        if pd.notna(val):
                            found[field][month] = float(val)

    if not any(found.values()):
        print(f"  [WARN] {path.name} — no recognisable production/export/stock rows.")
        print(f"         sheets: {xl.sheet_names}  (year guessed: {year})")
        print(f"         → share this file's layout so the parser can be tuned.")
        return None

    if year is None:
        print(f"  [WARN] {path.name} — could not infer year from filename; skipping.")
        return None

    months = sorted({m for d in found.values() for m in d})
    rows = []
    for m in months:
        rows.append({
            "date": pd.Timestamp(year=year, month=m, day=1),
            **{f: found[f].get(m) for f in _MPOB_FIELDS},
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    print(f"  parsed {path.name}: {len(df)} month(s) for {year}")
    return df


def _find_month_columns(raw: pd.DataFrame) -> dict[int, int]:
    """Return {column_index: month_number} from the first row that looks like
    a Jan..Dec header band."""
    for _, row in raw.iterrows():
        mapping = {}
        for col_idx, v in row.items():
            m = _coerce_month(v) if isinstance(v, str) else None
            if m:
                mapping[col_idx] = m
        if len(mapping) >= 6:          # at least half a year of headers
            return mapping
    return {}


def _year_from_name(name: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None


def fetch_mpob_monthly() -> pd.DataFrame | None:
    raw_dir = project_root() / "data" / "raw" / "mpob"
    files = sorted(raw_dir.glob("mpob_*.xls*"))
    if not files:
        print("[→] MPOB monthly — no files found in data/raw/mpob/")
        _print_mpob_instructions()
        return None

    print(f"[→] MPOB monthly — parsing {len(files)} file(s) …")
    frames = [df for f in files if (df := _parse_mpob_file(f)) is not None]
    if not frames:
        return None
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
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
