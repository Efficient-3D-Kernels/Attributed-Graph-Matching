"""Generates the Space-Time-Accuracy 3-panel frontier figure with crisp, publication-grade styling."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Repository root and destination directories
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIRS = [
    REPO_ROOT / "benchmarks" / "figures" / "standalone",
    REPO_ROOT / "paper" / "figures" / "standalone",
    REPO_ROOT / "els-cas-templates" / "figures" / "standalone",
    REPO_ROOT / "els-cas-templates" / "figures",
    REPO_ROOT / "els-cas-templates" / "figs",
    REPO_ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures",
    REPO_ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures" / "standalone",
    REPO_ROOT / "paper_materials" / "figures" / "standalone",
    REPO_ROOT / "paper_materials" / "figures",
    REPO_ROOT / "website" / "figures",
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.6), dpi=300)
plt.subplots_adjust(wspace=0.28, bottom=0.18, left=0.06, right=0.98, top=0.90)

# -------------------------------------------------------------
# Panel (a): Memory Wall - Space Complexity O(N^2) vs O(N^4)
# -------------------------------------------------------------
ax1 = axes[0]
Ns = np.array([10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500])

# O(N^4) float32 memory in Megabytes: (N^2 * N^2 * 4 bytes) / (1024^2)
mem_tensor_mb = (Ns**4 * 4.0) / (1024.0**2)
# O(N^2) float32 memory in Megabytes: (4 * N^2 * 4 bytes) / (1024^2)
mem_fastwcs_mb = (4 * Ns**2 * 4.0) / (1024.0**2)

ax1.plot(Ns, mem_tensor_mb, "r-o", lw=2.2, ms=5, label=r"Lawler Tensor QAP $\mathcal{O}(N^4)$ (IPFP, RRWM, SM)", zorder=5)
ax1.plot(Ns, mem_fastwcs_mb, "b-s", lw=2.2, ms=5, label=r"Fast-WCS $\mathcal{O}(N^2)$ Adjacency (Ours)", zorder=6)

# Workstation 16 GB RAM line
ax1.axhline(16 * 1024, color="darkred", ls="--", lw=1.2, label="Workstation RAM Limit (16 GB)")
ax1.axhspan(16 * 1024, 2e7, color="red", alpha=0.08, label="OOM / Crash Zone")

ax1.annotate("N=150: 4.06 GB", (150, mem_tensor_mb[Ns == 150][0]), xytext=(-95, -18), textcoords="offset points",
             fontsize=8.5, fontweight="bold", color="darkred", arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0))
ax1.annotate("N=500: 1.04 TB\n(Out of Memory)", (500, mem_tensor_mb[Ns == 500][0]), xytext=(-115, 14), textcoords="offset points",
             fontsize=8.5, fontweight="bold", color="darkred", arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0))
ax1.annotate("N=500: < 4 MB", (500, mem_fastwcs_mb[Ns == 500][0]), xytext=(-80, 12), textcoords="offset points",
             fontsize=8.5, fontweight="bold", color="blue", arrowprops=dict(arrowstyle="->", color="blue", lw=1.0))

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Graph Size ($M=N$ Vertices)", fontweight="bold")
ax1.set_ylabel("Peak Memory Consumption (MB, log-scale)", fontweight="bold")
ax1.set_title(r"(a) The $\mathcal{O}(N^4)$ Memory Wall vs. $\mathcal{O}(N^2)$ Fast-WCS", fontweight="bold", pad=10)
ax1.grid(True, which="both", ls=":", alpha=0.3)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8", fontsize=7.8)

# -------------------------------------------------------------
# Panel (b): Real Image Keypoints: Accuracy vs Latency
# -------------------------------------------------------------
ax2 = axes[1]
methods_img = [
    ("Fast-WCS (Ours)", 17.2, 89.1, "#b2182b", "o", 85, True),
    ("SciPy FAQ", 11.2, 28.9, "#2ca02c", "v", 55, False),
    ("RLX1F", 204.5, 70.3, "#053061", "^", 65, False),
    ("RLX1", 2047.4, 69.4, "#4393c3", "s", 65, False),
    ("RRWM", 520.1, 80.5, "#9467bd", "P", 75, False),
    ("FGW (eps=1e-2)", 442.0, 75.6, "#ff7f0e", "h", 65, False),
    ("SM", 358.4, 33.7, "#7f7f7f", "*", 65, False)
]

# Plot Pareto frontier
ax2.plot([11.2, 17.2], [28.9, 89.1], color="0.4", ls="--", lw=1.3, zorder=2, label="Pareto Optimal Frontier")

img_offsets = {
    "Fast-WCS (Ours)": (8, 2),
    "SciPy FAQ": (8, -2),
    "RLX1F": (8, -3),
    "RLX1": (-42, -12),
    "RRWM": (8, 3),
    "FGW (eps=1e-2)": (8, -9),
    "SM": (8, -3)
}

for name, ms, acc, col, mk, sz, is_ours in methods_img:
    ax2.scatter(ms, acc, color=col, marker=mk, s=sz, zorder=5)
    xy_off = img_offsets.get(name, (8, 2))
    ax2.annotate(name, (ms, acc), fontsize=8.0, xytext=xy_off, textcoords="offset points",
                 fontweight="bold" if is_ours else "normal",
                 color=col if is_ours else "black")

ax2.set_xscale("log")
ax2.set_xlabel("Execution Runtime (ms, log-scale) - Lower is Faster", fontweight="bold")
ax2.set_ylabel("Matching Accuracy (%) - Higher is Better", fontweight="bold")
ax2.set_title("(b) Real Image Keypoints: Accuracy vs. Latency", fontweight="bold", pad=10)
ax2.set_xlim(6.0, 4500)
ax2.set_ylim(18, 98)
ax2.grid(True, ls=":", alpha=0.35)
ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8", fontsize=7.8)

# -------------------------------------------------------------
# Panel (c): Dynamic Social Networks: Accuracy vs Latency (N=100, 30% Noise)
# -------------------------------------------------------------
ax3 = axes[2]
# Using highest performing variant V4 (WL-OA augmented) labeled Fast-WCS (Ours)
methods_sn = [
    ("Fast-WCS (Ours)", 126.1, 100.0, "#b2182b", "o", 85, True),
    ("WL-OA (Hungarian)", 58.3, 98.7, "#388e3c", "D", 75, False),
    ("Feature-Only Hungarian", 0.45, 72.0, "#f57c00", "v", 65, False),
    ("SciPy FAQ", 8.8, 29.3, "#7b1fa2", "P", 65, False),
    ("Yang 2018 (RLX1F)", 388.2, 17.3, "#616161", "X", 65, False)
]

# Plot Pareto frontier
ax3.plot([0.45, 58.3, 126.1], [72.0, 98.7, 100.0], color="0.4", ls="--", lw=1.3, zorder=2, label="Pareto Optimal Frontier")

# Careful offsets so NO label collides with data points, lines, or axes
p3_offsets = {
    "Fast-WCS (Ours)": (10, 4),               # Clear of Pareto line which ends at (126.1, 100)
    "WL-OA (Hungarian)": (-65, 9),            # Positioned cleanly above point and above Pareto line
    "Feature-Only Hungarian": (8, 2),         # To the right of point, below Pareto line slope
    "SciPy FAQ": (8, 2),                      # To the right of point
    "Yang 2018 (RLX1F)": (-110, -3)           # To the left of point, well clear of right border
}

for name, ms, acc, col, mk, sz, is_ours in methods_sn:
    ax3.scatter(ms, acc, color=col, marker=mk, s=sz, zorder=5)
    xy_off = p3_offsets.get(name, (8, 2))
    ax3.annotate(name, (ms, acc), fontsize=8.0, xytext=xy_off, textcoords="offset points",
                 fontweight="bold" if is_ours else "normal",
                 color=col if is_ours else "black")

ax3.set_xscale("log")
ax3.set_xlabel("Execution Runtime (ms, log-scale) - Lower is Faster", fontweight="bold")
ax3.set_ylabel("Matching Accuracy (%) - Higher is Better", fontweight="bold")
ax3.set_title("(c) Dynamic Social Networks: Accuracy vs. Latency", fontweight="bold", pad=10)
ax3.set_xlim(0.2, 1500)
ax3.set_ylim(10, 115)
ax3.grid(True, ls=":", alpha=0.35)
ax3.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8", fontsize=7.8)

# Save figure in all target locations
for out_dir in OUTPUT_DIRS:
    fig.savefig(out_dir / "fig_space_time_complexity_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_space_time_complexity_frontier.pdf", bbox_inches="tight")

print("[OK] Saved fig_space_time_complexity_frontier (.pdf and .png) to all target locations successfully!")
plt.close(fig)
