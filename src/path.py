from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    outputs: Path
    plots: Path
    models: Path
    submissions: Path

def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    outputs = root / "outputs"
    plots = outputs / "plots"
    models = outputs / "models"
    submissions = outputs / "submissions"
    for p in [data_processed, plots, models, submissions]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(root, data_raw, data_processed, outputs, plots, models, submissions)
