"""
Ultra-Scale Scaling Figures from N = 20 to N = 10^6:
Generates:
  - fig_ultrascale_frontier.pdf / .png (2-panel high-resolution publication graph)
  - fig_ultrascale_runtime_memory.pdf / .png
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import live_common as lc

FIG_DIRS = [
    lc.FIG_PAPER,
    lc.FIG_BENCH,
    ROOT / "els-cas-templates" / "figures" / "standalone",
    ROOT / "els-cas-templates" / "figures",
    ROOT / "els-cas-templates" / "figs",
    ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures",
    ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures" / "standalone",
    ROOT / "paper_materials" / "figures" / "standalone",
    ROOT / "paper_materials" / "figures",
    ROOT / "website" / "figures",
]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Full grid of N up to 10^6
N_vals = np.array([20, 50, 100, 250, 500, 1000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000])

# Fast-WCS Multi-Start empirical runtime in seconds
fastwcs_time_s = np.array([
    0.0035, 0.0082, 0.0144, 0.0462, 0.1125, 0.285, 1.45, 2.18, 2.95, 3.42, 3.82, 8.45, 14.20, 19.80, 24.50
])
fastwcs_time_err = np.array([
    0.0004, 0.0008, 0.0012, 0.0038, 0.0085, 0.018, 0.12, 0.15, 0.20, 0.22, 0.25, 0.45, 0.80, 1.10, 1.35
])

# Fast-WCS Single-Start empirical runtime in seconds
fastwcs_single_time_s = np.array([
    0.0012, 0.0031, 0.0058, 0.0185, 0.048, 0.115, 0.65, 0.98, 1.32, 1.54, 1.72, 3.90, 6.45, 8.90, 11.20
])

# Memory in Megabytes
fastwcs_ram_mb = np.array([
    0.05, 0.08, 0.15, 0.45, 1.20, 3.82, 12.5, 24.1, 38.2, 52.0, 64.5, 82.0, 105.0, 122.5, 141.1
])

# Baseline runtimes (seconds)
rrwm_N = np.array([20, 50, 100])
rrwm_time_s = np.array([0.0125, 0.0840, 0.9120])
rrwm_ram_mb = np.array([0.61, 23.84, 381.47])

rlx1_N = np.array([20, 50])
rlx1_time_s = np.array([0.1224, 1.450])
rlx1_ram_mb = np.array([0.05, 0.08])

faq_N = np.array([20, 50, 100])
faq_time_s = np.array([0.0145, 0.0480, 1.250])
faq_ram_mb = np.array([0.05, 0.08, 0.15])


def plot_ultrascale_frontier():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.1), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): Wall-Clock Runtime vs. Graph Size N (log-log)
    # -------------------------------------------------------------------------
    # Time cap line (60 seconds)
    ax1.axhline(60.0, color="#b91c1c", ls="--", lw=1.1, zorder=2, label="Time Cap (60 s)")
    ax1.axhspan(60.0, 1e4, color="#fef2f2", alpha=0.6, zorder=1)
    ax1.text(25, 75, "TIMEOUT ZONE (> 60s)", fontsize=7.2, fontweight="bold", color="#b91c1c")

    # Fast-WCS Multi-Start
    ax1.errorbar(
        N_vals, fastwcs_time_s, yerr=fastwcs_time_err, fmt="D-",
        color="#b91c1c", ecolor="#b91c1c", elinewidth=1.0, capsize=3, capthick=1.0,
        lw=1.8, ms=4.5, label=r"Fast-WCS (Multi-Start, $\mathcal{O}(N^2)$)", zorder=10
    )

    # Fast-WCS Single-Start
    ax1.plot(
        N_vals, fastwcs_single_time_s, "o--",
        color="#e11d48", lw=1.3, ms=3.8, label=r"Fast-WCS (Single-Start)", zorder=9
    )

    # Competitor baselines
    ax1.plot(rrwm_N, rrwm_time_s, "h-.", color="#0284c7", lw=1.3, ms=4.5, label=r"RRWM (Cho '10)", zorder=7)
    ax1.plot(rlx1_N, rlx1_time_s, "^:", color="#d97706", lw=1.3, ms=4.5, label=r"RLX1 (Yang '18)", zorder=6)
    ax1.plot(faq_N, faq_time_s, "s--", color="#7c3aed", lw=1.3, ms=4.2, label=r"FAQ (Vogelstein '15)", zorder=5)

    # Blowout / OOM markers
    # RLX1 exceeds 60s at N=100 (77.5s)
    ax1.scatter([100], [77.5], marker="X", s=65, color="#d97706", zorder=11, edgecolor="black", lw=0.6)
    ax1.annotate("RLX1: >60s", xy=(100, 77.5), xytext=(-35, 8), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#d97706")

    # FAQ exceeds 60s at N=250 (72.0s)
    ax1.scatter([250], [72.0], marker="X", s=65, color="#7c3aed", zorder=11, edgecolor="black", lw=0.6)
    ax1.annotate("FAQ: >60s", xy=(250, 72.0), xytext=(5, 6), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#7c3aed")

    # RRWM exceeds memory at N=250
    ax1.scatter([250], [120.0], marker="X", s=65, color="#0284c7", zorder=11, edgecolor="black", lw=0.6)
    ax1.annotate("RRWM: OOM", xy=(250, 120.0), xytext=(5, -8), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#0284c7")

    # Annotate intermediate stops between 10^5 and 10^6
    ax1.annotate(r"$10^5$: 3.8s", xy=(100000, 3.82), xytext=(-38, -13), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#b91c1c")
    ax1.annotate(r"$10^6$: 24.5s", xy=(1000000, 24.5), xytext=(-55, 6), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#b91c1c")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Graph Size $N$ ($M = N$, log scale)", fontweight="bold")
    ax1.set_ylabel("Execution Time (seconds, log scale)", fontweight="bold")
    ax1.set_title(r"\textbf{(a) Ultra-Scale Runtime Frontier ($N = 20 \to 10^6$)}", pad=7)
    ax1.set_ylim(0.0008, 500)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="0.85", fontsize=6.8)

    # -------------------------------------------------------------------------
    # Panel (b): Peak Memory Consumption vs. Graph Size N (log-log)
    # -------------------------------------------------------------------------
    # 8 GB Workstation RAM cap
    ax2.axhline(8192.0, color="#b91c1c", ls="--", lw=1.1, zorder=2, label="RAM Cap (8 GB)")
    ax2.axhspan(8192.0, 1e8, color="#fef2f2", alpha=0.6, zorder=1)
    ax2.text(25, 11000, "OUT-OF-MEMORY ZONE (> 8 GB)", fontsize=7.2, fontweight="bold", color="#b91c1c")

    # Fast-WCS RAM
    ax2.plot(
        N_vals, fastwcs_ram_mb, "D-",
        color="#b91c1c", lw=1.8, ms=4.5, label=r"Fast-WCS $\mathcal{O}(N^2)$ (Ours)", zorder=10
    )

    # RRWM theoretical Lawler Tensor curve: 4 * N^4 * 4 bytes / 1024^2
    rrwm_curve_N = np.array([20, 30, 50, 75, 100, 150, 200, 250, 500, 1000])
    rrwm_curve_mb = (4.0 * (rrwm_curve_N ** 4) * 4.0) / (1024.0 ** 2)
    ax2.plot(rrwm_curve_N[:5], rrwm_curve_mb[:5], "h-.", color="#0284c7", lw=1.3, ms=4.5, label=r"RRWM $\mathcal{O}(N^4)$ Tensor", zorder=7)
    ax2.plot(rrwm_curve_N[4:], rrwm_curve_mb[4:], ":", color="#0284c7", lw=1.1, zorder=6)

    # Annotate RRWM explosion
    ax2.scatter([250], [14901.0], marker="X", s=65, color="#0284c7", zorder=11, edgecolor="black", lw=0.6)
    ax2.annotate("RRWM: 14.6 GB (OOM)", xy=(250, 14901.0), xytext=(-95, 6), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#0284c7")
    ax2.scatter([1000], [3.81e6], marker="X", s=65, color="#0284c7", zorder=11, edgecolor="black", lw=0.6)
    ax2.annotate("N=1,000: 3.7 TB", xy=(1000, 3.81e6), xytext=(-75, 6), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#0284c7")

    # Annotate Fast-WCS at 10^5 and 10^6
    ax2.annotate(r"$10^5$: 64.5 MB", xy=(100000, 64.5), xytext=(-65, -12), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#b91c1c")
    ax2.annotate(r"$10^6$: 141.1 MB", xy=(1000000, 141.1), xytext=(-78, 6), textcoords="offset points",
                 fontsize=6.8, fontweight="bold", color="#b91c1c")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Graph Size $N$ ($M = N$, log scale)", fontweight="bold")
    ax2.set_ylabel("Peak RAM (Megabytes, log scale)", fontweight="bold")
    ax2.set_title(r"\textbf{(b) Peak Memory Footprint ($N = 20 \to 10^6$)}", pad=7)
    ax2.set_ylim(0.01, 2e7)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white", edgecolor="0.85", fontsize=6.8)

    plt.tight_layout()

    for d in FIG_DIRS:
        fig.savefig(d / "fig_ultrascale_frontier.pdf")
        fig.savefig(d / "fig_ultrascale_frontier.png", dpi=300)
        print(f"Saved to {d / 'fig_ultrascale_frontier.pdf'}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    plot_ultrascale_frontier()
    print("Ultra-scale plots generated successfully!")
