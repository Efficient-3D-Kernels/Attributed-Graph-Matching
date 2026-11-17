
"""
fig7_accuracy_vs_subgraph_size.py
====================================
[Paper] Matching Accuracy vs. Subgraph Size (L / N Scaling)

Features:
  - Pure white background
  - Black axes, labels, and title
  - Green cross at N = 200 indicating OOM / unavailable results
  - Borderless legend placed below the graph
  - All 9 graph matching solvers compared

Output:
    fig7_accuracy_vs_subgraph_size.png
    fig7_accuracy_vs_subgraph_size.pdf
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Paths and appearance ──────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"
GREEN_COLOR = "#00A000"


# ── Subgraph sizes ────────────────────────────────────────────────────────────
N_VALS = np.array([50, 100, 200, 300, 500])


# ── Accuracy Data (%) ────────────────────────────────────────────────────────
# None indicates that no result is available at that subgraph size.

METHODS_SIZE = [
    ("Fast-WCS (Multi-Start, O(N²))", "#8E1638", "o-", 2.5,
     [91.2, 93.5, 95.8, 96.4, 97.2]),

    ("Fast-WCS (Single-Start, O(N²))", "#C0392B", "s--", 2.0,
     [42.0, 48.5, 54.0, 58.2, 62.1]),

    ("RLX1F (Yang 2018)", "#2E86C1", "^-.", 1.8,
     [33.5, 35.8, 38.0, 39.5, 40.2]),

    ("RLX1 (Yang 2018)", "#5DADE2", "v:", 1.8,
     [27.0, 30.5, 34.2, 36.0, 37.5]),

    ("SciPy FAQ", "#34495E", "x--", 1.8,
     [11.0, 11.5, 12.0, 12.5, 12.8]),

    ("FGW (eps=1e-2)", "#8E44AD", "d-.", 1.8,
     [68.0, 75.4, 82.0, 85.5, 88.0]),

    ("SM (Gaussian K, O(N⁴))", "#D4AC0D", "+:", 1.8,
     [62.0, 68.5, None, None, None]),

    ("IPFP (Gaussian K, O(N⁴))", "#E67E22", "*-.", 1.8,
     [99.8, 100.0, None, None, None]),

    ("RRWM (Gaussian K, O(N⁴))", "#27AE60", "p-", 1.8,
     [99.7, 99.9, None, None, None]),
]


# ── Plot setup ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(
    figsize=(13, 8),
    facecolor=BG_COLOR
)

ax.set_facecolor(BG_COLOR)

legend_lines = []


# ── Plot all methods ──────────────────────────────────────────────────────────
for name, color, style, lw, acc_list in METHODS_SIZE:

    # Filter available accuracy values
    valid_mask = np.array([
        accuracy is not None
        for accuracy in acc_list
    ])

    valid_n = N_VALS[valid_mask]

    valid_acc = [
        accuracy
        for accuracy in acc_list
        if accuracy is not None
    ]

    # Plot method accuracy
    line, = ax.plot(
        valid_n,
        valid_acc,
        style,
        color=color,
        linewidth=lw,
        markersize=7.5,
        label=name,
        zorder=4
    )

    legend_lines.append(line)


# ── Single green cross at N = 200 ─────────────────────────────────────────────
# Indicates that SM, IPFP, and RRWM have no available results at N >= 200.

ax.scatter(
    [200],
    [0],
    marker="x",
    s=100,
    color=GREEN_COLOR,
    linewidths=2.5,
    zorder=6
)


# ── Axis labels ───────────────────────────────────────────────────────────────
ax.set_xlabel(
    "Number of Neurons  N  (Subgraph Scale)",
    fontsize=13,
    fontweight="bold",
    labelpad=8,
    color=TEXT_COLOR
)

ax.set_ylabel(
    "Matching Accuracy (%)",
    fontsize=13,
    fontweight="bold",
    labelpad=8,
    color=TEXT_COLOR
)


# ── Axis limits and ticks ─────────────────────────────────────────────────────
ax.set_xlim(30, 520)
ax.set_ylim(-3, 106)

ax.set_xticks(N_VALS)

ax.tick_params(
    axis="both",
    labelsize=10.5,
    colors=AXIS_COLOR
)


# ── Grid ──────────────────────────────────────────────────────────────────────
ax.yaxis.grid(
    True,
    linestyle="--",
    alpha=0.35,
    color="#B0B0B0",
    zorder=0
)

ax.xaxis.grid(
    True,
    linestyle="--",
    alpha=0.35,
    color="#B0B0B0",
    zorder=0
)

ax.set_axisbelow(True)


# ── Black axes ────────────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_color(AXIS_COLOR)
    spine.set_linewidth(1.1)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Matching Accuracy vs. Subgraph Size (L / N Scaling)\n"
    "Scalability Frontier Across 9 Solvers on Dense Relational Networks",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=12
)


# ── Green cross legend entry ──────────────────────────────────────────────────
oom_legend = Line2D(
    [0],
    [0],
    marker="x",
    color=GREEN_COLOR,
    linestyle="None",
    markersize=9,
    markeredgewidth=2.5,
    label="OOM / No Result at N ≥ 200"
)


# ── Borderless legend below the graph ─────────────────────────────────────────
ax.legend(
    handles=legend_lines + [oom_legend],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    fontsize=9.5,
    frameon=False,
    labelcolor=TEXT_COLOR
)


# ── Layout ────────────────────────────────────────────────────────────────────
plt.tight_layout()


# ── Save PNG and PDF ──────────────────────────────────────────────────────────
out_png = os.path.join(
    OUT_DIR,
    "fig7_accuracy_vs_subgraph_size.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig7_accuracy_vs_subgraph_size.pdf"
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
