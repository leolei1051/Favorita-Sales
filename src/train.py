from __future__ import annotations
import numpy as np
import pandas as pd

from src.model import train_model
from src.utils import save_model
from typing import Any


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"id", "date", "sales"}
    return [c for c in df.columns if c not in drop]




def train_and_submit(
    train_f: pd.DataFrame,
    test_f: pd.DataFrame,
    sample_sub: pd.DataFrame,
    out_path: str,
    p: Any,  # or Paths if you have a Paths dataclass
    model_name: str = "lgbm_favorita.joblib",
) -> None:

    feat_cols = get_feature_cols(train_f)

    X_tr = train_f[feat_cols]
    y_tr = np.log1p(np.clip(train_f["sales"].values, 0, None))

    model = train_model(X_tr, y_tr)

    # save model
    save_model(model, P.models / model_name)
    print("Saved model:", P.models / model_name)

    # predict test
    pred = np.expm1(model.predict(test_f[feat_cols]))
    pred = np.clip(pred, 0, None)

    sub = sample_sub.copy()
    sub["sales"] = pred
    sub.to_csv(out_path, index=False)
    print("Saved submission:", out_path)
