"""
Manual entry tool for the FCPO (crude palm oil) price.

Everything else — soybean oil, Brent, WTI, USD/MYR, the World Bank price — is
pulled automatically from the internet when you run the code. The ONLY price that
needs hand-entry (until IBKR is connected) is FCPO itself.

This tool:
  1. reads the database (data/raw/fcpo/fcpo_raw.csv),
  2. works out which recent trading days (today + earlier weekdays) have no price,
  3. asks you to type the closing price for each (press Enter to skip holidays),
  4. updates the database and keeps a durable record in fcpo_manual.csv.

After running it, run `python -m src.serving.serve --monitor` to refresh the
signal and dashboard with the new data.

Run: python -m src.ingestion.fcpo_manual
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from src.utils.config import project_root


def _db_path() -> Path:
    return project_root() / "data" / "raw" / "fcpo" / "fcpo_raw.csv"


def _manual_path() -> Path:
    return project_root() / "data" / "raw" / "fcpo" / "fcpo_manual.csv"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def find_missing_days(today=None, db_path: Path | None = None):
    """Weekdays from the day after the last price up to `today` that have no row.

    Weekends are skipped automatically. Public holidays will show up as 'missing'
    (there's no price to know that) — just press Enter to skip them.
    """
    db = _load(db_path or _db_path())
    last = db.index.max()
    today = pd.Timestamp(today or dt.date.today()).normalize()
    weekdays = pd.bdate_range(last + pd.Timedelta(days=1), today)   # Mon–Fri only
    have = set(db.index)
    missing = [d for d in weekdays if d not in have]
    return missing, last, db


def save_entries(entries: dict, db_path: Path | None = None, manual_path: Path | None = None) -> None:
    """Write the typed prices to the durable manual log and into the database."""
    db_path = db_path or _db_path()
    manual_path = manual_path or _manual_path()
    new = pd.Series({pd.Timestamp(k).normalize(): float(v) for k, v in entries.items()},
                    name="close").sort_index()
    new.index.name = "date"

    # durable manual record (so the entries survive a full re-merge from source)
    man = new.to_frame()
    if manual_path.exists():
        old = _load(manual_path)["close"]
        man = pd.concat([old[~old.index.isin(new.index)], new]).sort_index().to_frame()
    man.to_csv(manual_path)

    # update the database itself
    db = _load(db_path)
    for d, price in new.items():
        db.loc[d, "close"] = price          # new days get a close; OHLC stay blank (unused)
    db.sort_index().to_csv(db_path)


def _ask_price(day: pd.Timestamp) -> float | None:
    """Prompt for one day's closing price; return None to skip."""
    label = f"  {day.date()} ({day.strftime('%a')}) — closing price (Enter to skip): "
    while True:
        raw = input(label).strip().replace(",", "")
        if raw == "":
            return None
        try:
            val = float(raw)
            if val <= 0:
                print("    price must be a positive number — try again.")
                continue
            return val
        except ValueError:
            print("    that doesn't look like a number — try again (e.g. 1116).")


def main() -> None:
    missing, last, _ = find_missing_days()
    print("=" * 56)
    print("FCPO manual price entry")
    print("=" * 56)
    print(f"Latest price in database : {last.date()}")
    print(f"Today                    : {dt.date.today()}")

    if not missing:
        print("\n✓ The database is already up to date — nothing to enter.")
        return

    print(f"\n{len(missing)} day(s) need a closing price "
          f"(skip weekends are already excluded; press Enter to skip holidays):\n")
    entries = {d: p for d in missing if (p := _ask_price(d)) is not None}

    if not entries:
        print("\nNothing entered — database unchanged.")
        return

    save_entries(entries)
    print(f"\n✓ Updated the database with {len(entries)} day(s): "
          f"{', '.join(str(d.date()) for d in entries)}")
    print("Next: run  python -m src.serving.serve --monitor  "
          "to refresh the signal and dashboard.")


if __name__ == "__main__":
    main()
