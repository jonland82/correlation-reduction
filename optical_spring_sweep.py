"""Finite-time work for Yang et al.'s full cavity-mediated optical spring.

Run: python optical_spring_sweep.py
Outputs: optical_spring_results/results.csv and two PNG figures.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

from finite_time_sweep import W0, GAMMA1, GAMMA2, G0, G1, KB, T_LOW, gaussian_mi

OUT = Path(__file__).with_name("optical_spring_results")
OUT.mkdir(exist_ok=True)
RATIOS = (1.0, 10.0, 430.0)
DURATIONS_MS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
N_STEPS = 400


def matrices(h, ratio):
    """u=omega_0*x/sqrt(k_B*T_L/m), v=xdot/sqrt(k_B*T_L/m)."""
    a = np.array([
        [0, W0, 0, 0],
        [-W0 * (1 + h), -GAMMA1, -W0 * h, 0],
        [0, 0, 0, W0],
        [-W0 * h, 0, -W0 * (1 + h), -GAMMA2],
    ])
    d = np.diag([0, 2 * GAMMA1 * ratio, 0, 2 * GAMMA2])
    return a, d


def steady(h, ratio):
    a, d = matrices(h, ratio)
    c = solve_continuous_lyapunov(a, -d)
    return (c + c.T) / 2


def propagate(c, h, ratio, dt):
    if dt == 0:
        return c
    a, _ = matrices(h, ratio)
    css = steady(h, ratio)
    e = expm(a * dt)
    result = css + e @ (c - css) @ e.T
    return (result + result.T) / 2


def generalized_force(c):
    """<dH/dh> in units k_B*T_L; includes both diagonal spring shifts."""
    return (c[0, 0] + c[2, 2] + 2 * c[0, 2]) / 2


def info(c):
    iphase = gaussian_mi(c, ([0, 1], [2, 3]))
    position = c[np.ix_([0, 2], [0, 2])]
    ipos = gaussian_mi(position, ([0], [1]))
    return iphase, ipos


def log_gaussian_ratio(x, y, rho):
    """log[p(x,y)/(p(x)p(y))] for standardized Gaussian coordinates."""
    return -0.5 * np.log1p(-rho**2) + (
        rho * x * y - 0.5 * rho**2 * (x**2 + y**2)
    ) / (1 - rho**2)


def plot_ratio_landscapes():
    """Two-dimensional marginals of the pointwise dependence ratio."""
    grid = np.linspace(-3, 3, 301)
    xx, yy = np.meshgrid(grid, grid)
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.7), sharex=True,
                            sharey=True, layout="constrained")
    pairs = [
        (0, 2, r"$x_1$ versus $x_2$", 0.004),
        (0, 3, r"$x_1$ versus $v_2$", 0.48),
    ]
    for row, (a, b, label, limit) in enumerate(pairs):
        for col, hz in enumerate((85, 58)):
            c = steady(-2 * hz / 400_000, 430)
            rho = c[a, b] / np.sqrt(c[a, a] * c[b, b])
            field = log_gaussian_ratio(xx, yy, rho)
            plot = axs[row, col].contourf(
                xx, yy, field, levels=np.linspace(-limit, limit, 41),
                cmap="RdBu_r", extend="both",
            )
            axs[row, col].set_title(
                f"{hz} Hz; correlation $\\rho={rho:.5f}$", fontsize=10
            )
            axs[row, col].set_xlabel("Standardized first coordinate")
            if col == 0:
                axs[row, col].set_ylabel(label + "\nStandardized second coordinate")
        fig.colorbar(plot, ax=axs[row, :], shrink=0.78, pad=0.02,
                     label=r"$\log [p_{12}/(p_1p_2)]$")
    fig.suptitle(r"Pointwise dependence ratios at $T_H/T_L=430$")
    fig.savefig(OUT / "pointwise_ratio_landscapes.png", dpi=180)
    plt.close(fig)


def quasistatic_work(ratio):
    """Steady-state generalized-force integral using Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    midpoint = (G0 + G1) / 2
    halfwidth = (G1 - G0) / 2
    return halfwidth * sum(
        weight * generalized_force(steady(midpoint + halfwidth * node, ratio))
        for node, weight in zip(nodes, weights)
    )


def run(ratio, duration_ms):
    c = steady(G0, ratio)
    initial_info = info(c)
    work = 0.0
    if duration_ms == 0:
        work = generalized_force(c) * (G1 - G0)
    else:
        dt = duration_ms / (1000 * N_STEPS)
        h_previous = G0
        for j in range(N_STEPS):
            h_mid = G0 + (G1 - G0) * (j + 0.5) / N_STEPS
            work += generalized_force(c) * (h_mid - h_previous)
            c = propagate(c, h_mid, ratio, dt)
            h_previous = h_mid
        work += generalized_force(c) * (G1 - h_previous)
    end_info = info(c)
    final_info = info(steady(G1, ratio))
    return {
        "ratio": ratio, "duration_ms": duration_ms,
        "work_kBTlow": work, "work_J": work * KB * T_LOW,
        "quasistatic_work_kBTlow": quasistatic_work(ratio),
        "excess_work_kBTlow": work - quasistatic_work(ratio),
        "Iphase_initial": initial_info[0], "Iphase_end": end_info[0],
        "Iphase_final_steady": final_info[0],
        "Ipos_initial": initial_info[1], "Ipos_end": end_info[1],
        "Ipos_final_steady": final_info[1],
    }


def plot(rows):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    colors = {1.0: "#1b9e77", 10.0: "#d95f02", 430.0: "#7570b3"}
    for ratio in RATIOS:
        rr = [r for r in rows if r["ratio"] == ratio]
        x = [r["duration_ms"] for r in rr if r["duration_ms"] > 0]
        y = [r["work_kBTlow"] for r in rr if r["duration_ms"] > 0]
        axs[0].plot(x, y, "o-", color=colors[ratio], label=f"$T_H/T_L={ratio:g}$")
        axs[0].axhline(rr[0]["work_kBTlow"], color=colors[ratio], ls=":", alpha=0.4)
        axs[1].plot(x, [r["excess_work_kBTlow"] for r in rr if r["duration_ms"] > 0],
                    "o-", color=colors[ratio])
    rev = 0.5 * np.log((1 + 2 * G1) / (1 + 2 * G0))
    axs[0].axhline(rev, color="black", ls="--", lw=1.2, label="equal-T reversible")
    axs[0].set(xscale="log", yscale="log", xlabel="Ramp duration (ms)",
               ylabel=r"Total work ($k_B T_L$)", title="Full optical spring")
    axs[1].set(xscale="log", yscale="log", xlabel="Ramp duration (ms)",
               ylabel=r"Work above quasistatic ($k_B T_L$)",
               title="Finite-speed excess")
    axs[0].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "work_vs_duration.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ratio in RATIOS:
        rr = [r for r in rows if r["ratio"] == ratio]
        x = [r["duration_ms"] for r in rr if r["duration_ms"] > 0]
        for ax, name in zip(axs, ("Iphase_end", "Ipos_end")):
            ax.plot(x, [r[name] for r in rr if r["duration_ms"] > 0],
                    "o-", color=colors[ratio], label=f"$T_H/T_L={ratio:g}$")
    axs[0].set(xscale="log", yscale="log", xlabel="Ramp duration (ms)",
               ylabel="Phase-space MI (nats)")
    axs[1].set(xscale="log", yscale="log", xlabel="Ramp duration (ms)",
               ylabel="Position MI (nats)")
    for ax in axs:
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "information_vs_duration.png", dpi=180)
    plt.close(fig)


def main():
    rows = [run(r, d) for r in RATIOS for d in DURATIONS_MS]
    with (OUT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    plot_ratio_landscapes()
    for r in rows:
        print(f"ratio={r['ratio']:g} duration={r['duration_ms']:g} ms "
              f"W={r['work_kBTlow']:.9g} kBTL "
              f"Wexcess={r['excess_work_kBTlow']:.9g} kBTL "
              f"Iphase={r['Iphase_end']:.8g} Ipos={r['Ipos_end']:.8g}")
    print("Equal-T reversible work:", 0.5 * np.log((1 + 2 * G1) / (1 + 2 * G0)))
    print("Steady-state values at T_H/T_L=430:")
    for hz in (6, 8, 58, 85):
        c = steady(-2 * hz / 400_000, 430)
        current = GAMMA2 * KB * T_LOW * (c[3, 3] - 1)
        phase, pos = info(c)
        print(f"{hz} Hz: T1/TL={c[1,1]:.6g}, T2/TL={c[3,3]:.6g}, "
              f"J={current:.6g} W, Iphase={phase:.6g}, Ipos={pos:.6g}")


if __name__ == "__main__":
    main()
