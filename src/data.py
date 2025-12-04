from __future__ import annotations
from pathlib import Path
import pandas as pd

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)

def load_raw(data_raw: Path) -> dict[str, pd.DataFrame]:
    train = _read_csv(data_raw / "train.csv", parse_dates=["date"])
    test = _read_csv(data_raw / "test.csv", parse_dates=["date"])
    stores = _read_csv(data_raw / "stores.csv")
    oil = _read_csv(data_raw / "oil.csv", parse_dates=["date"])
    holidays = _read_csv(data_raw / "holidays_events.csv", parse_dates=["date"])
    sub = _read_csv(data_raw / "sample_submission.csv")
    return {"train": train, "test": test, "stores": stores, "oil": oil, "holidays": holidays, "sample_submission": sub}

def make_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    # Simple per-date flags (keep it robust and explainable)
    h = holidays.copy()
    if "transferred" in h.columns:
        h = h[h["transferred"] == False]
    h["is_holiday"] = h["type"].isin(["Holiday", "Additional", "Bridge"]).astype(int) if "type" in h.columns else 0
    daily = h.groupby("date", as_index=False).agg(
        holiday_count=("is_holiday", "sum"),
        is_holiday=("is_holiday", "max"),
    )
    return daily

def build_base(df: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(stores, on="store_nbr", how="left")
    out = out.merge(oil[["date", "dcoilwtico"]], on="date", how="left")
    out = out.merge(make_holiday_features(holidays), on="date", how="left")
    out["dcoilwtico"] = out["dcoilwtico"].ffill()
    out["holiday_count"] = out["holiday_count"].fillna(0)
    out["is_holiday"] = out["is_holiday"].fillna(0)
    return out

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Let LightGBM treat these as categorical if available
    for c in ["store_nbr", "family", "city", "state", "type", "cluster"]:
        if c in out.columns:
            out[c] = out[c].astype("category")
    if "onpromotion" in out.columns:
        out["onpromotion"] = out["onpromotion"].fillna(0).astype("float32")
    return out
