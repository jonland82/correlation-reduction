"""Make portrait SVG versions of the manuscript figures for narrow screens."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optical_spring_sweep import log_gaussian_ratio, steady


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
COLORS = {1.0: "#1b9e77", 10.0: "#d95f02", 430.0: "#7570b3"}
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
})


def save_svg(fig, path):
    fig.savefig(path, format="svg")
    # Matplotlib leaves spaces at the ends of a few path-data lines.
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def ratio_figure():
    grid = np.linspace(-3, 3, 301)
    xx, yy = np.meshgrid(grid, grid)
    fig, axs = plt.subplots(4, 1, figsize=(4.3, 15.5))
    fig.subplots_adjust(left=.20, right=.88, top=.97, bottom=.045, hspace=.45)
    cases = [
        (0, 2, 85, r"$x_1$ and $x_2$", r"$x_2$", .004),
        (0, 2, 58, r"$x_1$ and $x_2$", r"$x_2$", .004),
        (0, 3, 85, r"$x_1$ and $v_2$", r"$v_2$", .48),
        (0, 3, 58, r"$x_1$ and $v_2$", r"$v_2$", .48),
    ]
    for ax, (a, b, hz, pair, second, limit) in zip(axs, cases):
        c = steady(-2 * hz / 400_000, 430)
        rho = c[a, b] / np.sqrt(c[a, a] * c[b, b])
        field = log_gaussian_ratio(xx, yy, rho)
        plot = ax.imshow(
            field, extent=(-3, 3, -3, 3), origin="lower",
            cmap="RdBu_r", vmin=-limit, vmax=limit,
            interpolation="bilinear", aspect="equal",
        )
        ax.set(xlim=(-3, 3), ylim=(-3, 3),
               title=f"{hz} Hz · {pair}   " + rf"$\rho={rho:.5f}$",
               xlabel=r"standardized $x_1$",
               ylabel=f"standardized {second}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        colorbar = fig.colorbar(plot, ax=ax, fraction=.055, pad=.05, extend="both")
        colorbar.ax.set_title(r"$\log R$", pad=5, fontsize=10)
        colorbar.ax.tick_params(labelsize=8)
    save_svg(fig, OUT / "pointwise_ratio_mobile.svg")
    plt.close(fig)


def work_figure():
    with (ROOT / "source" / "data" / "optical_spring_results.csv").open(newline="", encoding="utf-8") as file:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(file)]
    fig, axs = plt.subplots(2, 1, figsize=(4.4, 8.0))
    fig.subplots_adjust(left=.20, right=.96, top=.96, bottom=.18, hspace=.49)
    for ratio, color in COLORS.items():
        selected = [row for row in rows if row["ratio"] == ratio]
        ramp = [row for row in selected if row["duration_ms"] > 0]
        x = [row["duration_ms"] for row in ramp]
        axs[0].plot(x, [row["work_kBTlow"] for row in ramp], "o-",
                    color=color, markersize=4.3, label=rf"$T_H/T_L={ratio:g}$")
        axs[0].axhline(selected[0]["work_kBTlow"], color=color,
                       linestyle=":", alpha=.4)
        axs[1].plot(x, [row["excess_work_kBTlow"] for row in ramp], "o-",
                    color=color, markersize=4.3)
    equal_t = [row for row in rows if row["ratio"] == 1.0][0]
    axs[0].axhline(equal_t["quasistatic_work_kBTlow"], color="#26354a",
                   linestyle="--", linewidth=1.2, label="equal-T reversible")
    axs[0].set(title="Total mechanical work", ylabel=r"work ($k_B T_L$)")
    axs[1].set(title="Work above steady-state path", ylabel=r"excess ($k_B T_L$)")
    for ax in axs:
        ax.set(xscale="log", yscale="log", xlabel="ramp duration (ms)")
        ax.grid(alpha=.25)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .025), frameon=False,
               handlelength=1.7, columnspacing=1.2)
    save_svg(fig, OUT / "work_vs_duration_mobile.svg")
    plt.close(fig)


if __name__ == "__main__":
    ratio_figure()
    work_figure()
    print("Wrote portrait SVG figures")
