"""Step 5: population-normalized borough comparison.

Raw counts favor big boroughs. Complaints per 100k residents is the
honest comparison — and often flips the ranking, which is one of your
best portfolio talking points.

Run:
    python src/per_capita_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.settings import DATA_DIR, PROCESSED_DIR, VISUAL_DIR


def main() -> None:
    counts = pd.read_parquet(PROCESSED_DIR / "borough_totals.parquet")
    population = pd.read_csv(DATA_DIR / "borough_population.csv")

    merged = counts.merge(population, on="borough", validate="one_to_one")
    merged["complaints_per_100k"] = (
        merged["complaint_count"] / merged["population"] * 100_000
    ).round(0)

    merged = merged.sort_values("complaints_per_100k", ascending=False)

    print("Raw vs population-adjusted borough comparison:\n")
    print(
        merged[["borough", "complaint_count", "population", "complaints_per_100k"]]
        .to_string(index=False)
    )

    output = PROCESSED_DIR / "borough_per_capita.parquet"
    merged.to_parquet(output, index=False)
    print(f"\nSaved to {output}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    merged_raw = merged.sort_values("complaint_count", ascending=True)
    axes[0].barh(merged_raw["borough"], merged_raw["complaint_count"], color="steelblue")
    axes[0].set_title("Raw complaint totals")

    merged_pc = merged.sort_values("complaints_per_100k", ascending=True)
    axes[1].barh(merged_pc["borough"], merged_pc["complaints_per_100k"], color="darkorange")
    axes[1].set_title("Complaints per 100k residents")

    plt.tight_layout()
    chart_path = VISUAL_DIR / "borough_raw_vs_per_capita.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
