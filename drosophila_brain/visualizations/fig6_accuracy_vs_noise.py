
"""
fig6_accuracy_vs_noise.py
===========================
[Paper] Matching Accuracy vs. Synaptic Edge Noise / Drop Rate

Compares 9 graph matching solvers across noise levels from 0% to 50%.

Features:
- Pure white background
- Black axes and tick marks
- Borderless legend
- Extreme Noise Regime annotation near legend
- 9 graph matching methods

Output:
fig6_accuracy_vs_noise.png
fig6_accuracy_vs_noise.pdf
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"

# ── Noise Levels (%) ──────────────────────────────────────────────────────────
noise_pcts = np.array(
    [0, 5, 10, 20, 30, 40, 50],
    dtype=float
)

# ── 9 Methods Configuration & Accuracy Data (%) ───────────────────────────────
METHODS_NOISE = [
    ("Fast-WCS (Multi-Start, O(N^2))",  "#8E1638", "o-",  2.5, [88.9, 78.5, 63.1, 41.2, 24.1, 14.2, 8.5]),
    ("Fast-WCS (Single-Start, O(N^2))", "#C0392B", "s--", 2.0, [30.2, 24.1, 17.2, 12.0, 8.7, 5.2, 3.1]),
    ("RLX1F (Yang 2018)",               "#2E86C1", "^-.", 1.8, [30.7, 26.3, 22.1, 14.5, 9.0, 5.4, 3.0]),
    ("RLX1 (Yang 2018)",                "#5DADE2", "v:",  1.8, [23.2, 18.0, 13.8, 8.4, 5.0, 3.2, 1.8]),
    ("SciPy FAQ",                       "#34495E", "x--", 1.8, [12.3, 12.8, 13.1, 8.2, 4.5, 2.5, 1.5]),
    ("FGW (eps=1e-2)",                  "#8E44AD", "d-.", 1.8, [61.0, 58.0, 54.2, 32.5, 14.3, 7.8, 4.2]),
    ("SM (Gaussian K, O(N^4))",         "#D4AC0D", "+:",  1.8, [85.5, 62.0, 40.6, 20.4, 7.8, 3.5, 1.9]),
    ("IPFP (Gaussian K, O(N^4))",       "#E67E22", "*-.", 1.8, [100.0, 98.2, 95.7, 65.4, 31.9, 14.8, 6.2]),
    ("RRWM (Gaussian K, O(N^4))",       "#27AE60", "p-",  1.8, [98.6, 98.5, 98.4, 91.2, 79.8, 52.3, 28.5]),
]

# ── Create Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Plot All 9 Methods ─────────────────────────────────────────────────────────
legend_lines = []

for name, color, style, lw, acc_vals in METHODS_NOISE:

    line, = ax.plot(
        noise_pcts,
        acc_vals,
        style,
        color=color,
        linewidth=lw,
        markersize=7,
        label=name,
        zorder=4
    )

    legend_lines.append(line)

# ── Highlight Extreme Noise Region ─────────────────────────────────────────────
ax.axvspan(
    30,
    50,
    alpha=0.06,
    color="#888888",
    zorder=1
)

# ── Axis Labels ────────────────────────────────────────────────────────────────
ax.set_xlabel(
    "Synaptic Edge Noise / Drop Rate (%)",
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

ax.set_xlim(-1, 51)
ax.set_ylim(0, 106)

ax.set_xticks(range(0, 55, 10))

# ── Tick Styling ──────────────────────────────────────────────────────────────
ax.tick_params(
    axis="both",
    labelsize=10.5,
    colors=AXIS_COLOR,
    width=1.2,
    direction="out"
)

# ── Grid ───────────────────────────────────────────────────────────────────────
ax.yaxis.grid(
    True,
    linestyle="--",
    alpha=0.35,
    color="#999999",
    linewidth=0.7,
    zorder=0
)

ax.xaxis.grid(
    True,
    linestyle="--",
    alpha=0.35,
    color="#999999",
    linewidth=0.7,
    zorder=0
)

ax.set_axisbelow(True)

# ── Black Axes ─────────────────────────────────────────────────────────────────
for spine in ax.spines.values():

    spine.set_color(AXIS_COLOR)
    spine.set_linewidth(1.0)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.set_title(
    "Matching Accuracy vs. Synaptic Edge Noise / Drop Rate\n"
    "Benchmarking 9 Graph Matching Solvers on Connectome & Relational Subgraphs",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=12
)

# ── Legend: Outside Below, No Border ───────────────────────────────────────────
ax.legend(
    handles=legend_lines,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=9.5,
    frameon=False
)

# ── Extreme Noise Regime Annotation Near Legend ────────────────────────────────
fig.text(
    0.5,
    0.025,
    r"Extreme Noise Regime ($\sigma > 0.30$)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#555555"
)

# ── Layout ─────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.08, 1, 1])

# ── Save PNG and PDF ───────────────────────────────────────────────────────────
out_png = os.path.join(
    OUT_DIR,
    "fig6_accuracy_vs_noise.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig6_accuracy_vs_noise.pdf"
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
