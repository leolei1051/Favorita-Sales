from __future__ import annotations
import pandas as pd


def rolling_folds(
    df: pd.DataFrame,
    date_col: str = "date",
    horizon: int = 16,
    n_folds: int = 5,
    step: int = 16,
):
    """
    Create rolling time folds.

    Returns a list of tuples:
        (train_end_date, val_start_date, val_end_date)

    Example (horizon=16):
        Train up to day T, validate on next 16 days.
        Then roll forward by `step` and repeat.
    """
    dates = pd.Index(sorted(df[date_col].unique()))
    if len(dates) < horizon + 2:
        raise ValueError("Not enough unique dates to create folds.")

    last = dates.max()
    end_idx = dates.get_loc(last)

    folds = []
    for k in range(n_folds):
        val_end_idx = end_idx - k * step
        val_start_idx = val_end_idx - horizon + 1
        train_end_idx = val_start_idx - 1

        if train_end_idx <= 0 or val_start_idx <= 0:
            break

        folds.append((dates[train_end_idx], dates[val_start_idx], dates[val_end_idx]))

    return list(reversed(folds))


def split_by_dates(
    df: pd.DataFrame,
    train_end,
    val_start,
    val_end,
    date_col: str = "date",
):
    """
    Split dataframe into train/val by date boundaries.
    """
    tr = df[df[date_col] <= train_end].copy()
    va = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)].copy()
    return tr, va

