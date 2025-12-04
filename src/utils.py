from __future__ import annotations
from pathlib import Path
import joblib

def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
