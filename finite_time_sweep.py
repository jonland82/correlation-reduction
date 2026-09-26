"""Finite-time coupling sweep for two classical Langevin oscillators.

Run from this directory: python finite_time_sweep.py
Writes figures and machine-readable results to finite_time_results/.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


OUT = Path(__file__).with_name("finite_time_results")
OUT.mkdir(exist_ok=True)
KB = 1.380649e-23
T_LOW = 300.0
W0 = 2 * np.pi * 400_000.0
GAMMA1 = 2 * np.pi * 12.0
GAMMA2 = 2 * np.pi * 6.0
G0 = -2 * 85 / 400_000.0
G1 = -2 * 58 / 400_000.0
N_STEPS = 400
RATIOS = (1.0, 2.0, 10.0)
DURATIONS_MS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)


def matrices(g, ratio):
    """State (u1,v1,u2,v2), with energy in k_B T_low and u=omega*x/sqrt(k_B T_low)."""
    a = np.array([
        [0, W0, 0, 0],
        [-W0, -GAMMA1, -W0 * g, 0],
        [0, 0, 0, W0],
        [-W0 * g, 0, -W0, -GAMMA2],
    ])
    d = np.diag([0, 2 * GAMMA1 * ratio, 0, 2 * GAMMA2])
    return a, d


def steady(g, ratio):
    a, d = matrices(g, ratio)
    c = solve_continuous_lyapunov(a, -d)
    return (c + c.T) / 2


def propagate(c, g, ratio, dt):
    """Exact covariance evolution at fixed g, using its steady covariance."""
    if dt == 0:
        return c
    a, _ = matrices(g, ratio)
    css = steady(g, ratio)
    e = expm(a * dt)
    result = css + e @ (c - css) @ e.T
    return (result + result.T) / 2


def gaussian_mi(c, indices):
    a = c[np.ix_(indices[0], indices[0])]
    b = c[np.ix_(indices[1], indices[1])]
    signa, loga = np.linalg.slogdet(a)
    signb, logb = np.linalg.slogdet(b)
    signc, logc = np.linalg.slogdet(c)
    assert min(signa, signb, signc) > 0
    return max(0.0, (loga + logb - logc) / 2)


def metrics(c, g):
    ip = gaussian_mi(c, ([0, 1], [2, 3]))
    ix = gaussian_mi(c[np.ix_([0, 2], [0, 2])], ([0], [1]))
    energy = 0.5 * np.trace(c) + g * c[0, 2]
    return ip, ix, energy


def run(ratio, duration_ms):
    c = steady(G0, ratio)
    ip0, ix0, e0 = metrics(c, G0)
    work = 0.0
    trace = [(0.0, G0, ip0, ix0, work)]
    if duration_ms == 0:
        work = c[0, 2] * (G1 - G0)
        ip, ix, ef = metrics(c, G1)
        trace.append((0.0, G1, ip, ix, work))
    else:
        dt = duration_ms / 1000 / N_STEPS
        g_previous = G0
        for j in range(N_STEPS):
            g_mid = G0 + (G1 - G0) * (j + 0.5) / N_STEPS
            work += c[0, 2] * (g_mid - g_previous)
            c = propagate(c, g_mid, ratio, dt)
            g_previous = g_mid
            if (j + 1) % 4 == 0:
                ip, ix, _ = metrics(c, g_mid)
                trace.append(((j + 1) * dt * 1000, g_mid, ip, ix, work))
        work += c[0, 2] * (G1 - g_previous)
        ip, ix, ef = metrics(c, G1)
        trace.append((duration_ms, G1, ip, ix, work))
    css_final = steady(G1, ratio)
    ip_ss, ix_ss, ess = metrics(css_final, G1)
    return {
        "ratio": ratio,
        "duration_ms": duration_ms,
        "work_kBTlow": work,
        "work_J": work * KB * T_LOW,
        "Iphase_initial": ip0,
        "Iphase_at_end": ip,
        "Iphase_final_steady": ip_ss,
        "Ipos_initial": ix0,
        "Ipos_at_end": ix,
        "Ipos_final_steady": ix_ss,
        "energy_change_at_end_kBTlow": ef - e0,
        "energy_change_after_relaxation_kBTlow": ess - e0,
    }, np.array(trace)


def plot_results(rows, traces):
    colors = {1.0: "#1b9e77", 2.0: "#d95f02", 10.0: "#7570b3"}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for ratio in RATIOS:
        rr = [r for r in rows if r["ratio"] == ratio]
        x = [r["duration_ms"] for r in rr if r["duration_ms"] > 0]
        y = [r["work_kBTlow"] * 1e8 for r in rr if r["duration_ms"] > 0]
        ax.plot(x, y, "o-", color=colors[ratio], label=f"$T_H/T_L={ratio:g}$")
        ax.axhline(rr[0]["work_kBTlow"] * 1e8, color=colors[ratio], alpha=0.35, ls=":")
    reversible = 0.5 * np.log((1 - G1**2) / (1 - G0**2)) * 1e8
    ax.axhline(reversible, color="black", ls="--", lw=1.3, label="equal-T reversible")
    ax.set(xscale="log", xlabel="Ramp duration (ms)", ylabel=r"Work on system ($10^{-8} k_B T_L$)",
           title="Work depends on ramp speed and bath temperatures")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "work_vs_duration.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    for ratio in RATIOS:
        rr = [r for r in rows if r["ratio"] == ratio]
        x = [r["duration_ms"] for r in rr if r["duration_ms"] > 0]
        axs[0].plot(x, [r["Iphase_at_end"] for r in rr if r["duration_ms"] > 0], "o-", color=colors[ratio], label=f"$T_H/T_L={ratio:g}$")
        axs[0].axhline(rr[0]["Iphase_final_steady"], color=colors[ratio], alpha=0.35, ls=":")
        axs[1].plot(x, [r["Ipos_at_end"] * 1e8 for r in rr if r["duration_ms"] > 0], "o-", color=colors[ratio])
        axs[1].axhline(rr[0]["Ipos_final_steady"] * 1e8, color=colors[ratio], alpha=0.35, ls=":")
    axs[0].set(xscale="log", yscale="log", xlabel="Ramp duration (ms)", ylabel="Phase-space MI (nats)", title="Full oscillator state")
    axs[1].set(xscale="log", xlabel="Ramp duration (ms)", ylabel=r"Position MI ($10^{-8}$ nats)", title="Positions only")
    for ax in axs:
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "information_vs_duration.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    for duration in (0.1, 1.0, 10.0, 100.0, 1000.0):
        t = traces[(10.0, duration)]
        axs[0].plot(t[:, 0] / duration, t[:, 2], label=f"{duration:g} ms")
        axs[1].plot(t[:, 0] / duration, t[:, 4] * 1e8, label=f"{duration:g} ms")
    axs[0].set(xlabel="Fraction of ramp completed", ylabel="Phase-space MI (nats)", title="$T_H/T_L=10$: information trajectory")
    axs[1].set(xlabel="Fraction of ramp completed", ylabel=r"Accumulated work ($10^{-8} k_B T_L$)", title="$T_H/T_L=10$: work trajectory")
    for ax in axs:
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories_hot_ratio_10.png", dpi=180)
    plt.close(fig)


def main():
    rows, traces = [], {}
    for ratio in RATIOS:
        for duration in DURATIONS_MS:
            row, trace = run(ratio, duration)
            rows.append(row)
            traces[(ratio, duration)] = trace
    with (OUT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, traces)
    for row in rows:
        print(f"T_H/T_L={row['ratio']:>4g}  ramp={row['duration_ms']:>6g} ms  "
              f"W={row['work_kBTlow']*1e8:>8.4f}e-8 kBT_L  "
              f"Iphase(end)={row['Iphase_at_end']:.6g}  "
              f"Ipos(end)={row['Ipos_at_end']:.6g}")


if __name__ == "__main__":
    main()
