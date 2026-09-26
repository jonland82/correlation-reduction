"""Plot the aggregate synthetic ReLU dependence results without importing torch."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path(__file__).parent / "data" / "relu_dependence"


def load_rows() -> list[dict]:
    with (RESULTS / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_dependence(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.3), layout="constrained")
    colors = {
        "linear strong": "#2266aa",
        "U-shaped": "#dc7132",
        "linear proxy": "#709e26",
        "interaction A": "#9a5ab0",
        "interaction B": "#9a5ab0",
        "linear weak": "#bd9a25",
        "noise A": "#888888",
        "noise B": "#888888",
    }
    for row in rows:
        x = float(row["corr"])
        y = float(row["mi"])
        loss = float(row["ablation"])
        size = 75 + 1000 * max(loss, 0)
        ax.scatter(x, y, s=size, color=colors[row["name"]],
                   edgecolor="white", linewidth=1, zorder=3)
    labels = {
        "linear strong": (8, -15),
        "U-shaped": (12, 8),
        "linear proxy": (-115, 8),
        "interaction A": (12, 28),
        "interaction B": (12, 12),
        "linear weak": (8, 8),
        "noise A": (18, -4),
    }
    for row in rows:
        name = row["name"]
        if name in labels:
            ax.annotate("noise A/B" if name == "noise A" else name,
                        (float(row["corr"]), float(row["mi"])),
                        xytext=labels[name], textcoords="offset points",
                        fontsize=9, arrowprops={"arrowstyle": "-", "alpha": 0.45}
                        if name.startswith("interaction") else None)
    ax.set(xlabel="Absolute Pearson correlation with sampled class",
           ylabel="Mutual information with sampled class (nats)",
           title="Baseline feature-output dependence")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.012, 0.495)
    ax.set_ylim(-0.005, 0.109)
    ax.text(0.98, 0.04,
            "Bubble area scales with test cross-entropy\nincrease after feature shuffling",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92})
    fig.savefig(RESULTS / "baseline_dependence.png", dpi=180)
    plt.close(fig)


def plot_energy(rows: list[dict]) -> None:
    selected = [row for row in rows if int(row["success"]) > 0]
    names = ["U-shaped", "Linear proxy", "Weak linear"]
    energies = [float(row["energy"]) for row in selected]
    errors = [float(row["energy_sd"]) for row in selected]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    bars = ax.bar(names, energies, yerr=errors, capsize=5,
                  color=["#dc7132", "#709e26", "#bd9a25"],
                  edgecolor="#333333", linewidth=0.6)
    for bar, value, error in zip(bars, energies, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + error + 0.16, f"{value:.2f} ± {error:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Net CPU package + DRAM energy (J)",
           title="Energy of selected dependence-reducing update")
    ax.set_ylim(0, 6.1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.96,
            "Bars: mean of three seeds\nError: SD across seeds\nStrong linear: no feasible update",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92})
    fig.savefig(RESULTS / "energy_to_target.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    aggregate = load_rows()
    plot_dependence(aggregate)
    plot_energy(aggregate)
