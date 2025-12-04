from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

def _try_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def train_model(X_train: pd.DataFrame, y_train: np.ndarray):
    lgb = _try_lightgbm()
    if lgb is not None:
        # LightGBM can handle pandas categorical dtype
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train, categorical_feature="auto")
        return model




# ...

# Fallback: sklearn pipeline with one-hot categorical
    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category" or X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
    )

    model = Pipeline(steps=[
        ("pre", pre),
        ("hgb", HistGradientBoostingRegressor(random_state=42, max_depth=10)),
    ])
    model.fit(X_train, y_train)
    return model
