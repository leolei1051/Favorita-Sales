from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_basic_plots(train: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Total sales over time (aggregate)
    daily = train.groupby("date", as_index=False)["sales"].sum()
    plt.figure(figsize=(10,4))
    plt.plot(daily["date"], daily["sales"])
    plt.title("Total Sales Over Time")
    plt.xlabel("Date"); plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(out_dir / "total_sales_over_time.png", dpi=200)
    plt.close()

    # 2) Sales distribution (log scale helps)
    plt.figure(figsize=(6,4))
    plt.hist(train["sales"].clip(lower=0), bins=60)
    plt.title("Sales Distribution")
    plt.xlabel("Sales"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "sales_distribution.png", dpi=200)
    plt.close()

    # 3) Promo effect (if exists)
    if "onpromotion" in train.columns:
        tmp = train.copy()
        tmp["promo_flag"] = (tmp["onpromotion"].fillna(0) > 0).astype(int)
        promo = tmp.groupby("promo_flag")["sales"].mean()
        plt.figure(figsize=(5,4))
        plt.bar(["No Promo", "Promo"], promo.values)
        plt.title("Avg Sales: Promo vs No Promo")
        plt.ylabel("Mean Sales")
        plt.tight_layout()
        plt.savefig(out_dir / "promo_effect.png", dpi=200)
        plt.close()
