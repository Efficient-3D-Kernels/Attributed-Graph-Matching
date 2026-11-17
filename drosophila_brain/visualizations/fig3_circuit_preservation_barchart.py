
"""
fig3_circuit_preservation_barchart.py
=======================================
[Poster & Paper] Circuit / Cell-Type Preservation Bar Chart

Pure white background, black axes and borderless legend.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Methods and Color Palette ──────────────────────────────────────────────────
METHODS = [
    ("Fast-WCS (Multi-Start, O(N^2))",  "#8E1638", 1.0),
    ("Fast-WCS (Single-Start, O(N^2))", "#C0392B", 0.9),
    ("RLX1F (Yang 2018)",               "#2E86C1", 0.85),
    ("RLX1 (Yang 2018)",                "#5DADE2", 0.8),
    ("SciPy FAQ",                       "#34495E", 0.85),
    ("FGW (eps=1e-2)",                  "#8E44AD", 0.85),
    ("SM (Gaussian K, O(N^4))",         "#D4AC0D", 0.85),
    ("IPFP (Gaussian K, O(N^4))",       "#E67E22", 0.85),
    ("RRWM (Gaussian K, O(N^4))",       "#27AE60", 0.85),
]

# Pure white background
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"

# ── Circuit Preservation Data (%) ──────────────────────────────────────────────
DATA = {
    "Fast-WCS (Multi-Start, O(N^2))":  [95.4, 91.2, 88.6],
    "Fast-WCS (Single-Start, O(N^2))": [82.1, 78.4, 74.0],
    "RLX1F (Yang 2018)":               [68.4, 62.1, 57.3],
    "RLX1 (Yang 2018)":                [54.2, 48.6, 44.1],
    "SciPy FAQ":                       [42.5, 38.0, 35.2],
    "FGW (eps=1e-2)":                  [74.8, 69.5, 64.2],
    "SM (Gaussian K, O(N^4))":         [71.0, 64.3, 59.8],
    "IPFP (Gaussian K, O(N^4))":       [86.5, 81.0, 76.5],
    "RRWM (Gaussian K, O(N^4))":       [89.2, 84.7, 80.1],
}

circuits = [
    "Kenyon Cells (KCs)",
    "MBONs",
    "Projection Neurons (PNs)"
]

n_circuits = len(circuits)
n_methods = len(METHODS)

# ── Plot Layout ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Calculate bar widths and positions
group_width = 0.82
bar_width = group_width / n_methods
x_centers = np.arange(n_circuits)

# ── Plot bars ──────────────────────────────────────────────────────────────────
for i, (name, color, alpha) in enumerate(METHODS):
    offset = (i - (n_methods - 1) / 2) * bar_width
    vals = DATA[name]

    edge_col = "#5A0E23" if "Fast-WCS" in name else "white"
    lw = 1.0 if "Fast-WCS (Multi-Start" in name else 0.5

    ax.bar(
        x_centers + offset,
        vals,
        bar_width * 0.92,
        label=name,
        color=color,
        alpha=alpha,
        edgecolor=edge_col,
        linewidth=lw,
        zorder=3
    )

# ── Annotate selected bars ─────────────────────────────────────────────────────
for i, (name, color, _) in enumerate(METHODS):
    if name in [
        "Fast-WCS (Multi-Start, O(N^2))",
        "RRWM (Gaussian K, O(N^4))",
        "SciPy FAQ"
    ]:
        offset = (i - (n_methods - 1) / 2) * bar_width
        vals = DATA[name]

        for c_idx, val in enumerate(vals):
            ax.text(
                x_centers[c_idx] + offset,
                val + 1.2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=color,
                rotation=90
            )

# ── Grid and styling ──────────────────────────────────────────────────────────
ax.yaxis.grid(
    True,
    linestyle="--",
    alpha=0.4,
    color="#999999",
    zorder=0
)
ax.set_axisbelow(True)

ax.set_ylim(0, 115)

ax.set_ylabel(
    "% Circuit Subgraph Preserved",
    fontsize=13,
    color=TEXT_COLOR,
    fontweight="bold",
    labelpad=8
)

ax.set_xticks(x_centers)
ax.set_xticklabels(
    circuits,
    fontsize=12,
    fontweight="bold",
    color=TEXT_COLOR
)

ax.tick_params(
    axis="both",
    colors=AXIS_COLOR,
    labelsize=10,
    width=1.2
)

# All four axes/spines black
for spine in ax.spines.values():
    spine.set_color(AXIS_COLOR)
    spine.set_linewidth(1.0)

ax.set_title(
    "Circuit / Cell-Type Subgraph Preservation Across 9 Graph Matching Methods\n"
    "Drosophila Mushroom Body: Hemibrain ↔ FlyWire Partial Graph Matching",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=14
)

# ── Legend OUTSIDE BELOW the graph (NO BORDER) ─────────────────────────────────
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=9.5,
    frameon=False,
    edgecolor="none"
)

# ── Save outputs ───────────────────────────────────────────────────────────────
plt.tight_layout()

out_png = os.path.join(OUT_DIR, "fig3_circuit_barchart.png")
out_pdf = os.path.join(OUT_DIR, "fig3_circuit_barchart.pdf")

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
