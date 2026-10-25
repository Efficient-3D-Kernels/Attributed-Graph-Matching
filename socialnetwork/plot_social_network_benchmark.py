"""
Publication-Quality Plotting for Social Network Benchmarks
Generates clean, aesthetic figures for the paper and deliverables:
  1. results/plots/social_network_accuracy_frontier.png
  2. results/plots/social_network_runtime_pareto.png
  3. results/plots/social_network_objective_collapse.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DELIVERABLES_ROOT = PROJECT_ROOT.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PAPER_FIGS = DELIVERABLES_ROOT / "paper" / "figures" / "standalone"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

df_methods = pd.read_csv(TABLES_DIR / "social_network_comprehensive_methods.csv")
df_kernels = pd.read_csv(TABLES_DIR / "social_network_global_kernels.csv")

# Color palette & styling
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

color_map = {
    "Fast-WCS v4 (WL-OA Augmented)": "#D32F2F",   # Strong Crimson
    "Fast-WCS v3 (Ours)": "#1976D2",               # Royal Blue
    "Fast-WCS v3 (No 2-Opt)": "#0288D1",           # Light Blue
    "WL-OA (Hungarian)": "#388E3C",                # Forest Green
    "Feature-Only Hungarian": "#F57C00",           # Amber
    "SciPy FAQ (Topology Only)": "#7B1FA2",        # Purple
    "Yang 2018 (RLX1F)": "#616161"                 # Gray
}

marker_map = {
    "Fast-WCS v4 (WL-OA Augmented)": "o",
    "Fast-WCS v3 (Ours)": "s",
    "Fast-WCS v3 (No 2-Opt)": "^",
    "WL-OA (Hungarian)": "D",
    "Feature-Only Hungarian": "v",
    "SciPy FAQ (Topology Only)": "P",
    "Yang 2018 (RLX1F)": "X"
}

# =========================================================================
# FIGURE 1: ACCURACY VS NOISE (N=40 and N=100 Side-by-Side)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for ax, n_val, title in zip(axes, [40, 100], ["(a) Subgraph Size N = 40 (L = 30)", "(b) Subgraph Size N = 100 (L = 75)"]):
    df_sub = df_methods[df_methods["N"] == n_val]
    for method, grp in df_sub.groupby("Method"):
        grp = grp.sort_values("Noise_pct")
        c = color_map.get(method, "#333333")
        m = marker_map.get(method, "o")
        lw = 2.5 if "v4" in method or "v3" in method else 1.8
        ls = "-" if "v4" in method or "v3" in method or "WL-OA" in method else "--"
        ax.plot(grp["Noise_pct"], grp["Accuracy_pct"], label=method, color=c, marker=m, linewidth=lw, linestyle=ls, markersize=7)
    
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Structural Perturbation Noise (%)")
    ax.set_ylim(-2, 105)
    ax.set_xticks([0, 5, 10, 20, 30])

axes[0].set_ylabel("Ground-Truth Matching Accuracy (%)")
axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, facecolor="#fafafa")

def save_all_formats(fig, name):
    for base in [PLOTS_DIR, PAPER_FIGS]:
        fig.savefig(base / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(base / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {name}.png and {name}.pdf")

save_all_formats(fig, "social_network_accuracy_frontier")
plt.close(fig)

# =========================================================================
# FIGURE 2: PARETO FRONTIER: ACCURACY VS RUNTIME (N=100, 30% NOISE)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))
df_stress = df_methods[(df_methods["N"] == 100) & (df_methods["Noise_pct"] == 30)]

custom_offsets = {
    "Fast-WCS v4 (WL-OA Augmented)": (1.15, 2.0),
    "Fast-WCS v3 (Ours)": (1.15, -2.0),
    "Fast-WCS v3 (No 2-Opt)": (0.38, -5.0),
    "WL-OA (Hungarian)": (0.32, 3.5),
    "Feature-Only Hungarian": (1.20, 1.0),
    "SciPy FAQ (Topology Only)": (1.18, 1.5),
    "Yang 2018 (RLX1F)": (1.15, -2.5)
}

for _, row in df_stress.iterrows():
    m = row["Method"]
    c = color_map.get(m, "#333333")
    marker = marker_map.get(m, "o")
    ax.scatter(row["Runtime_ms"], row["Accuracy_pct"], color=c, marker=marker, s=140, label=m, zorder=5)
    scale_x, delta_y = custom_offsets.get(m, (1.15, 1.5))
    ax.annotate(
        f" {m}", 
        (row["Runtime_ms"] * scale_x, row["Accuracy_pct"] + delta_y), 
        fontsize=9, 
        fontweight="bold" if "Fast-WCS" in m else "normal",
        color="#111111"
    )

ax.set_xscale("log")
ax.set_xlabel("Wall-Clock Latency (ms, log scale)")
ax.set_ylabel("Ground-Truth Accuracy (%) at 30% Noise")
ax.set_title("Space-Time-Accuracy Pareto Trade-off (Stanford CollegeMsg, N = 100)", fontweight="bold")
ax.set_ylim(-5, 110)
ax.set_xlim(0.8, 2000)

plt.tight_layout()
save_all_formats(fig, "social_network_runtime_pareto")
plt.close(fig)

# =========================================================================
# FIGURE 3: OBJECTIVE VALUE STABILITY (LOG SCALE)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
df_n100 = df_methods[df_methods["N"] == 100].dropna(subset=["Objective"])

for method, grp in df_n100.groupby("Method"):
    grp = grp.sort_values("Noise_pct")
    c = color_map.get(method, "#333333")
    m = marker_map.get(method, "o")
    # Plot objective + 1e-4 for log scale
    ax.plot(grp["Noise_pct"], grp["Objective"] + 1e-4, label=method, color=c, marker=m, linewidth=2.2, markersize=7)

ax.set_yscale("log")
ax.set_xlabel("Structural Perturbation Noise (%)")
ax.set_ylabel("WCS Objective Energy $F^*(X)$ (log scale)")
ax.set_title("Optimization Robustness: Residual Energy vs Noise (N = 100)", fontweight="bold")
ax.set_xticks([0, 5, 10, 20, 30])
ax.legend(frameon=True, facecolor="#fafafa")

plt.tight_layout()
save_all_formats(fig, "social_network_objective_collapse")
plt.close(fig)
print("All social network benchmark plots successfully generated.")
