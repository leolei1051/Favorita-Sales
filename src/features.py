from __future__ import annotations
import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek.astype("int8")
    out["month"] = out["date"].dt.month.astype("int8")
    out["year"] = out["date"].dt.year.astype("int16")
    out["day"] = out["date"].dt.day.astype("int8")
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int16")
    return out

def add_lag_rolling(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    lags=(1, 7, 14, 28),
    windows=(7, 28),
) -> pd.DataFrame:
    out = df.sort_values(group_cols + ["date"]).copy()
    grp = out.groupby(group_cols, sort=False, observed=False)

    # Lags
    for L in lags:
        out[f"{target_col}_lag{L}"] = grp[target_col].shift(L)

    # Rolling stats (shift by 1 to avoid leakage)
    shifted = grp[target_col].shift(1)
    for W in windows:
        out[f"{target_col}_rmean{W}"] = (
            shifted.groupby([out[c] for c in group_cols], sort=False, observed=False)
                   .transform(lambda s: s.rolling(W, min_periods=1).mean())
        )
        out[f"{target_col}_rstd{W}"] = (
            shifted.groupby([out[c] for c in group_cols], sort=False, observed=False)
                   .transform(lambda s: s.rolling(W, min_periods=1).std())
        )

    # Fill early-history NaNs
    feat_cols = [c for c in out.columns if c.startswith(f"{target_col}_lag") or c.startswith(f"{target_col}_r")]
    out[feat_cols] = out[feat_cols].fillna(0)

    return out


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Concatenate to build consistent lag features for test using train history.
    all_df = pd.concat([train, test], axis=0, ignore_index=True)
    all_df = add_time_features(all_df)

    # sales exists only in train; for test it should be NaN (ok)
    if "sales" not in all_df.columns:
        raise ValueError("Expected 'sales' column in train/test concat (test can be NaN).")

    all_df = add_lag_rolling(all_df, group_cols=["store_nbr", "family"], target_col="sales")

    train_f = all_df.iloc[: len(train)].copy()
    test_f = all_df.iloc[len(train) :].copy()
    return train_f, test_f
