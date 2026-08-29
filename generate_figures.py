"""Generate Figures 2--6 directly from the reproduced experiment tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"


def read(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finish(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    benchmark = read("benchmark_main_generated.csv")
    families = [r["family"] for r in benchmark]
    x = np.arange(len(families))
    width = 0.37
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - width / 2, [float(r["labeled_variables"]) for r in benchmark], width, label="Vehicle-labeled", color=ORANGE)
    plt.bar(x + width / 2, [float(r["aggregate_variables"]) for r in benchmark], width, label="Aggregate", color=BLUE)
    plt.xticks(x, families)
    plt.ylabel("Decision variables")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    finish("Figure_2.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - width / 2, [float(r["labeled_time_seconds"]) for r in benchmark], width, label="Vehicle-labeled", color=ORANGE)
    plt.bar(x + width / 2, [float(r["aggregate_time_seconds"]) for r in benchmark], width, label="Aggregate", color=BLUE)
    plt.xticks(x, families)
    plt.ylabel("Median construction + solve time (s)")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    finish("Figure_3.png")

    clone = read("clone_sensitivity_generated.csv")
    multiplicity = np.asarray([int(r["vehicles_per_class"]) for r in clone])
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(multiplicity, [float(r["labeled_time_seconds"]) for r in clone], "o-", color=ORANGE, label="Vehicle-labeled")
    plt.plot(multiplicity, [float(r["aggregate_time_seconds"]) for r in clone], "s-", color=BLUE, label="Aggregate")
    plt.xlabel("Vehicles per class")
    plt.ylabel("Construction + solve time (s)")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    finish("Figure_4.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(multiplicity, [float(r["log10_symmetry_group"]) for r in clone], "o-", color=PURPLE)
    plt.xlabel("Vehicles per class")
    plt.ylabel(r"$\log_{10}((m!)^2)$")
    plt.grid(alpha=0.25)
    finish("Figure_5.png")

    frontier = read("triobjective_frontier_generated.csv")
    economic = np.asarray([float(r["economic_cost"]) for r in frontier])
    environmental = np.asarray([float(r["environmental_impact"]) for r in frontier])
    transit = np.asarray([float(r["transit_time_burden"]) for r in frontier])
    normalized = (transit - transit.min()) / max(transit.max() - transit.min(), 1e-12)
    plt.figure(figsize=(7.2, 4.8))
    scatter = plt.scatter(economic, environmental, c=normalized, cmap="viridis", s=34, edgecolor="black", linewidth=0.25)
    plt.xlabel("Economic cost")
    plt.ylabel("Environmental-impact index")
    plt.grid(alpha=0.20)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Normalized transit-time burden")
    finish("Figure_6.png")


if __name__ == "__main__":
    main()
