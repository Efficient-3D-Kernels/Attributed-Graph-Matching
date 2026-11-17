
"""
fig1_bipartite_alignment.py
============================
2D Anatomical Bipartite Alignment Graph

Hemibrain neurons (left, blue) <--> FlyWire neurons (right, maroon)
Green lines = predicted matched pairs
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

# ── Paths ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATCHES_CSV = os.path.join(BASE, "predicted_neuron_matches_wcs_2d.csv")
COORDS_A_CSV = os.path.join(BASE, "flyA_hemibrain_MB_coords.csv")
COORDS_B_CSV = os.path.join(BASE, "flyB_fafb_MB_coords.csv")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ────────────────────────────────────────────────────

COL_A_MATCH = "#3D4B5C"
COL_A_UNMATCH = "#A8B5C0"

COL_B_MATCH = "#8E1638"
COL_B_UNMATCH = "#D4909A"

COL_EDGE = "#2A7D4F"
BG_COLOR = "#FFFFFF"

# ── Load data ─────────────────────────────────────────────────────────

matches = pd.read_csv(MATCHES_CSV)
coords_a = pd.read_csv(COORDS_A_CSV)
coords_b = pd.read_csv(COORDS_B_CSV)

matched_a = set(matches["fly_A_neuron"].tolist())
matched_b = set(matches["fly_B_neuron"].tolist())

all_a = coords_a["bodyId"].tolist()
all_b = coords_b["bodyId"].tolist()

# ── Select neurons ────────────────────────────────────────────────────

MAX_SHOW = 60

if len(all_a) > MAX_SHOW:

    matched_a_list = [
        x for x in all_a if x in matched_a
    ][:MAX_SHOW]

    unmatched_a = [
        x for x in all_a if x not in matched_a
    ]

    n_fill = max(0, MAX_SHOW - len(matched_a_list))

    rng = np.random.default_rng(42)

    unmatched_a_sel = rng.choice(
        unmatched_a,
        size=min(n_fill, len(unmatched_a)),
        replace=False
    ).tolist()

    show_a = matched_a_list + unmatched_a_sel

else:
    show_a = all_a


if len(all_b) > MAX_SHOW:

    matched_b_list = [
        x for x in all_b if x in matched_b
    ][:MAX_SHOW]

    unmatched_b = [
        x for x in all_b if x not in matched_b
    ]

    n_fill = max(0, MAX_SHOW - len(matched_b_list))

    rng = np.random.default_rng(42)

    unmatched_b_sel = rng.choice(
        unmatched_b,
        size=min(n_fill, len(unmatched_b)),
        replace=False
    ).tolist()

    show_b = matched_b_list + unmatched_b_sel

else:
    show_b = all_b


n_a = len(show_a)
n_b = len(show_b)

# ── Assign node positions ─────────────────────────────────────────────

pos_a = {
    nid: (0.0, 1.0 - i / max(n_a - 1, 1))
    for i, nid in enumerate(show_a)
}

pos_b = {
    nid: (1.0, 1.0 - i / max(n_b - 1, 1))
    for i, nid in enumerate(show_b)
}

# ── Figure setup ──────────────────────────────────────────────────────

fig, ax = plt.subplots(
    figsize=(14, 10),
    facecolor=BG_COLOR
)

ax.set_facecolor(BG_COLOR)

ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.05, 1.12)

ax.axis("off")

# ── Main title (separated from column labels) ─────────────────────────

fig.suptitle(
    "Neuron Correspondence: Hemibrain ↔ FlyWire "
    "(WCS-GNCCP Partial Matching)",
    fontsize=16,
    fontweight="bold",
    color="#1a1a2e",
    y=0.965
)

# ── Draw matching lines ───────────────────────────────────────────────

visible_matches = matches[
    matches["fly_A_neuron"].isin(pos_a)
    & matches["fly_B_neuron"].isin(pos_b)
]

for _, row in visible_matches.iterrows():

    xa, ya = pos_a[row["fly_A_neuron"]]
    xb, yb = pos_b[row["fly_B_neuron"]]

    cx = 0.5

    cy = (
        (ya + yb) / 2
        + np.random.default_rng(
            int(row["fly_A_neuron"]) % 1000
        ).uniform(-0.02, 0.02)
    )

    verts = [
        (xa, ya),
        (cx, cy),
        (xb, yb)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.LINETO
    ]

    path = Path(verts, codes)

    patch = mpatches.PathPatch(
        path,
        facecolor="none",
        edgecolor=COL_EDGE,
        linewidth=0.9,
        alpha=0.55,
        zorder=1
    )

    ax.add_patch(patch)


# ── Draw Hemibrain nodes ──────────────────────────────────────────────

for nid, (x, y) in pos_a.items():

    matched = nid in matched_a

    color = COL_A_MATCH if matched else COL_A_UNMATCH
    size = 90 if matched else 35
    alpha = 1.0 if matched else 0.55

    ax.scatter(
        x, y,
        s=size,
        c=color,
        alpha=alpha,
        zorder=3,
        edgecolors="white" if matched else "none",
        linewidths=0.6
    )


# ── Draw FlyWire nodes ────────────────────────────────────────────────

for nid, (x, y) in pos_b.items():

    matched = nid in matched_b

    color = COL_B_MATCH if matched else COL_B_UNMATCH
    size = 90 if matched else 35
    alpha = 1.0 if matched else 0.55

    ax.scatter(
        x, y,
        s=size,
        c=color,
        alpha=alpha,
        zorder=3,
        edgecolors="white" if matched else "none",
        linewidths=0.6
    )


# ── Column labels (positioned below main title) ────────────────────────

ax.text(
    0.0,
    1.045,
    "Hemibrain\n(Fly A)",
    transform=ax.transData,
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color=COL_A_MATCH
)

ax.text(
    1.0,
    1.045,
    "FlyWire\n(Fly B)",
    transform=ax.transData,
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color=COL_B_MATCH
)


# ── Legend ────────────────────────────────────────────────────────────

legend_elements = [

    mpatches.Patch(
        facecolor=COL_A_MATCH,
        label=f"Hemibrain matched ({len(visible_matches)})"
    ),

    mpatches.Patch(
        facecolor=COL_A_UNMATCH,
        label="Hemibrain unmatched"
    ),

    mpatches.Patch(
        facecolor=COL_B_MATCH,
        label=f"FlyWire matched ({len(visible_matches)})"
    ),

    mpatches.Patch(
        facecolor=COL_B_UNMATCH,
        label="FlyWire unmatched"
    ),

    mpatches.Patch(
        facecolor=COL_EDGE,
        label="Predicted correspondence"
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=10,
    framealpha=0.85,
    edgecolor="#ccc",
    bbox_to_anchor=(0.5, -0.04)
)

# ── Adjust layout ─────────────────────────────────────────────────────

# Reserve space at the top for the title and column labels.
# Reserve space at the bottom for the legend.

fig.subplots_adjust(
    top=0.87,
    bottom=0.12,
    left=0.08,
    right=0.92
)

# ── Save outputs ──────────────────────────────────────────────────────

out_png = os.path.join(
    OUT_DIR,
    "fig1_bipartite_alignment.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig1_bipartite_alignment.pdf"
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
