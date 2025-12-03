from __future__ import annotations
import numpy as np
import pandas as pd

from src.path import get_paths
from src.data import load_raw, build_base, enforce_types
from src.features import make_features
from src.split import rolling_folds, split_by_dates
from src.metrics import rmsle
from src.model import train_model
from src.utils import save_model
from src.viz import save_basic_plots
from src.train import train_and_submit
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"id", "date", "sales"}  # keep sales only as target
    return [c for c in df.columns if c not in drop]

def backtest(train_f: pd.DataFrame):
    folds = rolling_folds(train_f, horizon=16, n_folds=5, step=16)
    feat_cols = get_feature_cols(train_f)

    scores = []
    for (train_end, val_start, val_end) in folds:
        tr, va = split_by_dates(train_f, train_end, val_start, val_end)
        X_tr, y_tr = tr[feat_cols], np.log1p(np.clip(tr["sales"].values, 0, None))
        X_va, y_va = va[feat_cols], va["sales"].values

        model = train_model(X_tr, y_tr)
        pred_log = model.predict(X_va)
        pred = np.expm1(pred_log)
        score = rmsle(y_va, pred)
        scores.append(score)
        print(f"Fold {val_start} → {val_end} | RMSLE: {score:.5f}")

    print(f"\nMean RMSLE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

def train_and_submit(train_f: pd.DataFrame, test_f: pd.DataFrame, sample_sub: pd.DataFrame, out_path: str):
    feat_cols = get_feature_cols(train_f)

    X_tr = train_f[feat_cols]
    y_tr = np.log1p(np.clip(train_f["sales"].values, 0, None))

    model = train_model(X_tr, y_tr)
    pred = np.expm1(model.predict(test_f[feat_cols]))
    pred = np.clip(pred, 0, None)

    sub = sample_sub.copy()
    sub["sales"] = pred
    sub.to_csv(out_path, index=False)
    print("Saved submission:", out_path)

def main():
    P = get_paths()
    raw = load_raw(P.data_raw)

    train = enforce_types(build_base(raw["train"], raw["stores"], raw["oil"], raw["holidays"]))
    test = enforce_types(build_base(raw["test"], raw["stores"], raw["oil"], raw["holidays"]))

    # Build lags/rolling consistently for train & test
    train_f, test_f = make_features(train, test)

    # Optional: save processed features
    train_f.to_parquet(P.data_processed / "features_train.parquet", index=False)
    test_f.to_parquet(P.data_processed / "features_test.parquet", index=False)

    print("Train features shape:", train_f.shape, "Test features shape:", test_f.shape)

    # 1)backtest
    try:
        backtest(train_f)
    except Exception as e:
        print("Backtest failed, skipping:", repr(e))

    # 2) train on full data + save submission (and optionally save model)
    train_and_submit(
        train_f=train_f,
        test_f=test_f,
        sample_sub=raw["sample_submission"],
        out_path=str(P.submissions / "submission.csv"),
        P=p,  # pass paths so we can save model/plots
    )

    save_basic_plots(train, P.plots)
    print("Saved plots to:", P.plots)


if __name__ == "__main__":
    main()
