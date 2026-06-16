"""
Phase 4 — Labeling: turn forward FCPO moves into buy/sell/hold targets.

Rule (from config/features.yaml `label`):
  forward return over `horizon_days` FCPO sessions:
      r_fwd(t) = close(t + horizon) / close(t) - 1
  BUY  if r_fwd >  buy_threshold   (+1%)
  SELL if r_fwd <  sell_threshold  (-1%)
  HOLD otherwise

Key correctness points:
  - The forward return is computed on FCPO's OWN trading calendar, so `horizon`
    means 3 real FCPO sessions (the actual holding period), not 3 rows of the
    inner-joined feature panel (which omits days when US markets were shut).
  - The label legitimately looks into the future — that is the TARGET, not a
    feature. It is kept in its own file and never fed back as an input.
  - The final `horizon` dates have no future yet, so they are unlabeled/dropped.

Output:
  data/processed/labels.parquet     date, fwd_ret_3d, label
  data/features/dataset.parquet     feature_matrix joined with label (Phase-5 ready)

Run: python -m src.processing.label
"""

import pandas as pd

from src.utils.config import load_config, project_root
from src.utils.io import save_parquet, load_parquet


def _fcpo_close() -> pd.Series:
    path = project_root() / "data" / "raw" / "fcpo" / "fcpo_raw.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["close"].sort_index()


def build_labels() -> pd.DataFrame:
    cfg = load_config("features")["label"]
    horizon = cfg["horizon_days"]
    buy_th = cfg["buy_threshold"]
    sell_th = cfg["sell_threshold"]

    close = _fcpo_close()
    # forward return over `horizon` FCPO sessions, on FCPO's own calendar
    fwd_ret = close.shift(-horizon) / close - 1.0

    label = pd.Series("HOLD", index=close.index, name="label")
    label[fwd_ret > buy_th] = "BUY"
    label[fwd_ret < sell_th] = "SELL"
    # last `horizon` rows have no future → not labelable
    label[fwd_ret.isna()] = pd.NA

    labels = pd.DataFrame({"fwd_ret_3d": fwd_ret, "label": label}).dropna()

    out_path = project_root() / "data" / "processed" / "labels.parquet"
    save_parquet(labels, out_path)

    dist = labels["label"].value_counts()
    pct = (dist / len(labels) * 100).round(1)
    print(f"[labels] horizon={horizon}d  buy>{buy_th:+.0%}  sell<{sell_th:+.0%}")
    print(f"[labels] {len(labels)} labelled days "
          f"({labels.index[0].date()} → {labels.index[-1].date()})")
    for cls in ["BUY", "HOLD", "SELL"]:
        print(f"         {cls:<5} {dist.get(cls, 0):>6}  ({pct.get(cls, 0):>4}%)")

    _build_dataset(labels)
    return labels


def _build_dataset(labels: pd.DataFrame) -> None:
    """Join features with the label into one Phase-5-ready modeling table.

    Inner join keeps only dates that have BOTH a complete feature row and a
    label, so the tail (unlabelable) and warm-up (no features) both fall away.
    """
    feat_path = project_root() / load_config("data")["paths"]["features"] / "feature_matrix.parquet"
    features = load_parquet(feat_path)

    dataset = features.join(labels, how="inner")
    out_path = project_root() / "data" / "features" / "dataset.parquet"
    save_parquet(dataset, out_path)
    print(f"[dataset] {dataset.shape[0]} rows × {features.shape[1]} features + label "
          f"→ {out_path}")


if __name__ == "__main__":
    build_labels()
