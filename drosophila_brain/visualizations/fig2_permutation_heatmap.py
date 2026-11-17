
"""
fig2_permutation_heatmap.py
============================
[Research Paper] Permutation / Correspondence Matrix Heatmap

Zero values       = White
Low values        = Light blue
Medium values     = Blue
High values       = Dark navy

Background        = White
Text              = Black

Data source:
  - viz_data.json (contains X matrix, M, N and matches)

Output:
  - fig2_permutation_heatmap.png
  - fig2_permutation_heatmap.pdf
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# ── Paths ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIZ_JSON = os.path.join(BASE, "viz_data.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data ─────────────────────────────────────────────────────────

print("Loading viz_data.json ...")

with open(VIZ_JSON) as f:
    data = json.load(f)

M = data["M"]
N = data["N"]

matches = data["matches"]

X = np.array(data["X"], dtype=float)

print(f"Full matrix shape: {X.shape}")
print(f"Total matches: {len(matches)}")

# ── Validate matrix ───────────────────────────────────────────────────

if X.ndim != 2:
    raise ValueError("X must be a two-dimensional matrix.")

if not np.all(np.isfinite(X)):
    raise ValueError("X contains NaN or infinite values.")

if X.shape != (M, N):
    raise ValueError(
        f"Matrix shape {X.shape} does not match M={M}, N={N}"
    )

# ── Build reordered subset ────────────────────────────────────────────

SUBSET = 100

# Sort matches by residual (best first)

matches_sorted = sorted(
    matches,
    key=lambda m: m["residual"]
)[:SUBSET]

if len(matches_sorted) == 0:
    raise ValueError("No matches found in viz_data.json.")

row_idx = [m["i"] for m in matches_sorted]
col_idx = [m["j"] for m in matches_sorted]

# Extract reordered submatrix

X_sub = X[np.ix_(row_idx, col_idx)]

print(f"Displayed matrix shape: {X_sub.shape}")

# ── Custom colormap: White -> Blue -> Dark Navy ───────────────────────

cmap_colors = [
    "#FFFFFF",   # Zero - pure white
    "#DCEAF5",   # Very light blue
    "#91B9D8",   # Light blue
    "#4682B4",   # Medium blue
    "#174A73",   # Dark blue
    "#082B49"    # Deep navy
]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_blue",
    cmap_colors,
    N=512
)

# Set zero values to pure white.
# This also ensures missing values appear white.

cmap.set_under("#FFFFFF")
cmap.set_bad("#FFFFFF")

# ── Figure setup: Pure white background ───────────────────────────────

fig, ax = plt.subplots(
    figsize=(9, 8),
    facecolor="white"
)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Heatmap ───────────────────────────────────────────────────────────

max_value = X_sub.max()

if max_value <= 0:
    raise ValueError("X_sub has no positive correspondence values.")

im = ax.imshow(
    X_sub,
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=1e-12,
    vmax=max_value
)

# ── Colorbar ──────────────────────────────────────────────────────────

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.038,
    pad=0.02
)

cbar.ax.set_facecolor("white")

cbar.set_label(
    "Match Score",
    color="black",
    fontsize=11,
    labelpad=10
)

cbar.ax.tick_params(
    colors="black",
    labelsize=9
)

cbar.outline.set_edgecolor("black")

# ── Axis labels ───────────────────────────────────────────────────────

ax.set_xlabel(
    "FlyWire Neurons (Fly B)",
    color="black",
    fontsize=12,
    labelpad=10
)

ax.set_ylabel(
    "Hemibrain Neurons (Fly A)",
    color="black",
    fontsize=12,
    labelpad=10
)

# ── Axis ticks ────────────────────────────────────────────────────────

ax.tick_params(
    axis="both",
    colors="black",
    labelsize=9
)

ax.xaxis.set_major_locator(
    MaxNLocator(integer=True, nbins=6)
)

ax.yaxis.set_major_locator(
    MaxNLocator(integer=True, nbins=6)
)

# ── Plot borders ──────────────────────────────────────────────────────

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(0.8)

# ── Diagonal reference line ───────────────────────────────────────────

ax.plot(
    [0, len(col_idx) - 1],
    [0, len(row_idx) - 1],
    color="black",
    linewidth=0.8,
    linestyle="--",
    alpha=0.65,
    label="True-match diagonal"
)

ax.legend(
    loc="upper right",
    fontsize=9,
    facecolor="white",
    edgecolor="black",
    labelcolor="black",
    framealpha=1
)

# ── Title ─────────────────────────────────────────────────────────────

ax.set_title(
    f"Permutation Matrix X ({len(row_idx)} × {len(col_idx)} Matched Subset)\n"
    "WCS-GNCCP Correspondence Score Distribution",
    color="black",
    fontsize=13,
    fontweight="bold",
    pad=15
)

# ── Layout ────────────────────────────────────────────────────────────

plt.tight_layout()

# ── Output paths ──────────────────────────────────────────────────────

out_png = os.path.join(
    OUT_DIR,
    "fig2_permutation_heatmap.png"
)

out_pdf = os.path.join(
    OUT_DIR,
    "fig2_permutation_heatmap.pdf"
)

# ── Save high-resolution PNG ──────────────────────────────────────────

plt.savefig(
    out_png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="white",
    transparent=False
)

# ── Save publication-quality PDF ──────────────────────────────────────

plt.savefig(
    out_pdf,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="white",
    transparent=False
)

plt.close()

# ── Completion message ────────────────────────────────────────────────

print("\nFigure generation completed successfully!")
print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")

print("\nColor interpretation:")
print("  0          = White")
print("  Low scores = Light blue")
print("  Mid scores = Medium blue")
print("  High scores= Dark navy")
