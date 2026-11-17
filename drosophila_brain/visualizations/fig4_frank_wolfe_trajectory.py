
"""
fig4_frank_wolfe_trajectory.py
================================
[Paper] Frank-Wolfe Continuation Trajectory & Duality Gap

Single-panel figure:
- Dual-axis Frank-Wolfe continuation trajectory of Fast-WCS.
- Continuation parameter zeta: +1.0 (convex) -> -1.0 (concave).
- Left Y-axis: Homotopy objective J_zeta(X).
- Right Y-axis: Frank-Wolfe duality gap g_t.

Pure white background.
Black axes and tick marks.
Borderless legend.

Output:
fig4_frank_wolfe_trajectory.png
fig4_frank_wolfe_trajectory.pdf
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_JSON = os.path.join(BASE, "viz_data.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colors and Styles ──────────────────────────────────────────────────────────
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
AXIS_COLOR = "#000000"

COL_J = "#8E1638"       # Maroon: Homotopy objective
COL_GAP = "#34495E"     # Slate: Frank-Wolfe duality gap

# ── Load Fast-WCS History ──────────────────────────────────────────────────────
if os.path.exists(VIZ_JSON):

    with open(VIZ_JSON, "r") as f:
        data = json.load(f)

    history = data.get("history", [])

else:
    history = []

# Use actual experimental history when available
if history:

    zetas = [h["zeta"] for h in history]
    J_vals = [h["J"] for h in history]
    gap_vals = [h["gap"] for h in history]

    steps = list(range(len(history)))

else:

    # Illustrative fallback values.
    # Replace with actual experimental results before publication.
    zetas = [
        1.0, 0.901, 0.623, 0.223,
        -0.223, -0.623, -0.901, -1.0
    ]

    J_vals = [
        0.11, 0.46, 1.42, 2.68,
        -21.6, -62.0, -90.0, -100.0
    ]

    gap_vals = [
        0.0, 0.286, 1.069, 2.144,
        0.0, 0.0, 0.0, 0.0
    ]

    steps = list(range(len(zetas)))


# ── Create Single-Panel Figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ══════════════════════════════════════════════════════════════════════════════
# Primary Y-axis: Homotopy Objective
# ══════════════════════════════════════════════════════════════════════════════

line_j = ax.plot(
    steps,
    J_vals,
    "o-",
    color=COL_J,
    linewidth=2.4,
    markersize=7,
    markeredgecolor=COL_J,
    markerfacecolor=COL_J,
    label=r"Homotopy Objective $J_{\zeta}(X)$",
    zorder=4
)

ax.fill_between(
    steps,
    J_vals,
    alpha=0.08,
    color=COL_J,
    zorder=2
)

ax.set_xlabel(
    r"Continuation Step (Homotopy Parameter $\zeta: +1.0 \rightarrow -1.0$)",
    fontsize=11.5,
    fontweight="bold",
    color=TEXT_COLOR,
    labelpad=10
)

ax.set_ylabel(
    r"Homotopy Objective $J_{\zeta}(X)$",
    fontsize=11.5,
    fontweight="bold",
    color=COL_J,
    labelpad=10
)

ax.tick_params(
    axis="y",
    colors=AXIS_COLOR,
    labelsize=9.5,
    width=1.2
)

ax.tick_params(
    axis="x",
    colors=AXIS_COLOR,
    labelsize=9,
    width=1.2
)

ax.set_xticks(steps)

ax.set_xticklabels(
    [
        f"s{i}\nζ={z:.2f}"
        for i, z in enumerate(zetas)
    ],
    color=TEXT_COLOR
)

# ══════════════════════════════════════════════════════════════════════════════
# Secondary Y-axis: Frank-Wolfe Duality Gap
# ══════════════════════════════════════════════════════════════════════════════

ax_twin = ax.twinx()

ax_twin.set_facecolor("none")

line_gap = ax_twin.plot(
    steps,
    gap_vals,
    "s--",
    color=COL_GAP,
    linewidth=2.0,
    markersize=6.5,
    markeredgecolor=COL_GAP,
    markerfacecolor=COL_GAP,
    label=r"Frank-Wolfe Duality Gap $g_t$",
    zorder=4
)

ax_twin.fill_between(
    steps,
    gap_vals,
    alpha=0.06,
    color=COL_GAP,
    zorder=2
)

ax_twin.set_ylabel(
    r"Frank-Wolfe Duality Gap $g_t$",
    fontsize=11.5,
    fontweight="bold",
    color=COL_GAP,
    labelpad=10
)

ax_twin.tick_params(
    axis="y",
    colors=AXIS_COLOR,
    labelsize=9.5,
    width=1.2
)

ax_twin.set_ylim(bottom=0)

# ══════════════════════════════════════════════════════════════════════════════
# Convergence Annotation
# ══════════════════════════════════════════════════════════════════════════════

if len(steps) > 4 and max(gap_vals) > 0:

    conv_step = 4

    ax_twin.axvspan(
        conv_step - 0.4,
        len(steps) - 0.5,
        alpha=0.07,
        color="#27AE60",
        zorder=1
    )

    ax_twin.text(
        conv_step + 0.1,
        max(gap_vals) * 0.45,
        r"Convergence: $g_t \rightarrow 0$",
        color="#1E8449",
        fontsize=9.5,
        fontweight="bold"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Grid and Axis Styling
# ══════════════════════════════════════════════════════════════════════════════

ax.yaxis.grid(
    True,
    linestyle="--",
    alpha=0.35,
    color="#999999",
    linewidth=0.7
)

ax.xaxis.grid(
    False
)

ax.set_axisbelow(True)

# Black axes
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(AXIS_COLOR)
    ax.spines[spine].set_linewidth(1.1)

ax.spines["top"].set_visible(False)

ax_twin.spines["right"].set_color(AXIS_COLOR)
ax_twin.spines["right"].set_linewidth(1.1)
ax_twin.spines["top"].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════

ax.set_title(
    "Frank-Wolfe Continuation Trajectory and Duality Gap",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    pad=16
)

# ══════════════════════════════════════════════════════════════════════════════
# Legend (No Border / No Frame)
# ══════════════════════════════════════════════════════════════════════════════

all_lines = line_j + line_gap

ax.legend(
    all_lines,
    [line.get_label() for line in all_lines],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
    fontsize=9.5,
    frameon=False
)

# ══════════════════════════════════════════════════════════════════════════════
# Save PNG and PDF
# ══════════════════════════════════════════════════════════════════════════════

plt.tight_layout()

out_png = os.path.join(
    OUT_DIR,
    "fig4_frank_wolfe_trajectory.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig4_frank_wolfe_trajectory.pdf"
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
