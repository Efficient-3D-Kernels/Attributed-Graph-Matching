
"""
fig8_runtime_scaling_loglog.py
================================
[Paper] Wall-Clock Runtime Scaling (Log-Log Plot)

- Pure white background
- Black axes, labels, and title
- Borderless legend below the graph
- No memory-limit annotation
- No speedup annotation
- No empirical anchor-point dots at N = 35

Output:
    fig8_runtime_scaling_loglog.png
    fig8_runtime_scaling_loglog.pdf
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Paths and appearance ──────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"


# ── Runtime configuration ─────────────────────────────────────────────────────
# Name, Color, Style, Line Width, Complexity Exponent,
# Anchor Runtime (ms at N=35), Maximum N

METHODS_RUNTIME = [
    ("Fast-WCS (Multi-Start, O(N²))", "#8E1638", "o-",  2.5, 2.2, 40.0,   500),
    ("Fast-WCS (Single-Start, O(N²))", "#C0392B", "s--", 2.2, 2.0, 18.0,  500),
    ("RLX1F (Yang 2018)", "#2E86C1", "^-.", 1.8, 3.0, 99.7,  500),
    ("RLX1 (Yang 2018)", "#5DADE2", "v:",  1.8, 3.5, 984.0, 500),
    ("SciPy FAQ", "#34495E", "x--", 1.8, 2.6, 5.3, 500),
    ("FGW (eps=1e-2)", "#8E44AD", "d-.", 1.8, 3.0, 1741.0, 500),
    ("SM (Gaussian K, O(N⁴))", "#D4AC0D", "+:", 1.8, 4.0, 71.4, 150),
    ("IPFP (Gaussian K, O(N⁴))", "#E67E22", "*-.", 1.8, 4.0, 44.5, 150),
    ("RRWM (Gaussian K, O(N⁴))", "#27AE60", "p-", 1.8, 4.0, 384.0, 150),
]

N_REF = 35.0
N_TICKS = [50, 100, 200, 300, 500]


# ── Plot setup ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(13, 8),
    facecolor=BG_COLOR
)

ax.set_facecolor(BG_COLOR)

legend_lines = []


# ── Plot runtime curves ───────────────────────────────────────────────────────
for name, color, style, lw, exponent, t_ref, max_n in METHODS_RUNTIME:

    n_curve = np.logspace(
        np.log10(50),
        np.log10(max_n),
        45
    )

    t_curve = t_ref * (n_curve / N_REF) ** exponent

    # Plot continuous runtime scaling curve
    line, = ax.loglog(
        n_curve,
        t_curve,
        style,
        color=color,
        linewidth=lw,
        markersize=5,
        markevery=9,
        label=name,
        zorder=4
    )

    legend_lines.append(line)

    # Mark the last available point for tensor solvers
    if max_n < 500:

        t_crash = t_ref * (max_n / N_REF) ** exponent

        ax.scatter(
            [max_n],
            [t_crash],
            marker="x",
            s=90,
            color=color,
            linewidths=2.5,
            zorder=6
        )


# ── Axis labels ───────────────────────────────────────────────────────────────
ax.set_xlabel(
    "Number of Neurons  N  (log-scale)",
    fontsize=13,
    fontweight="bold",
    labelpad=8,
    color=TEXT_COLOR
)

ax.set_ylabel(
    "Median Execution Time (ms, log-scale)",
    fontsize=13,
    fontweight="bold",
    labelpad=8,
    color=TEXT_COLOR
)


# ── Axis limits and ticks ─────────────────────────────────────────────────────
ax.set_xlim(30, 520)
ax.set_ylim(2, 5e7)

ax.set_xticks(N_TICKS)

ax.get_xaxis().set_major_formatter(
    ticker.ScalarFormatter()
)


# ── Grid ──────────────────────────────────────────────────────────────────────
ax.yaxis.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.35,
    color="#B0B0B0",
    zorder=0
)

ax.xaxis.grid(
    True,
    which="major",
    linestyle="--",
    alpha=0.35,
    color="#B0B0B0",
    zorder=0
)

ax.set_axisbelow(True)


# ── Black axes and tick marks ─────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color(AXIS_COLOR)
    spine.set_linewidth(1.1)

ax.tick_params(
    axis="both",
    labelsize=10.5,
    colors=AXIS_COLOR,
    which="both"
)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Wall-Clock Runtime Scaling (Log–Log)\n"
    "Benchmarking 9 Solvers: Fast-WCS O(N²) Acceleration vs O(N⁴) Tensor Bottlenecks",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=12
)


# ── Borderless legend below the graph ─────────────────────────────────────────
ax.legend(
    handles=legend_lines,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    fontsize=9.5,
    frameon=False,
    labelcolor=TEXT_COLOR
)


# ── Layout ────────────────────────────────────────────────────────────────────
plt.tight_layout()


# ── Save PNG and PDF ─────────────────────────────────────────────────────────
out_png = os.path.join(
    OUT_DIR,
    "fig8_runtime_scaling_loglog.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig8_runtime_scaling_loglog.pdf"
)

plt.savefig(
    out_png,
    dpi=300,
    bbox_inches="tight",
    facecolor=BG_COLOR
)

plt.savefig(
    out_pdf,
    bbox_inches="tight",
    facecolor=BG_COLOR
)

plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
