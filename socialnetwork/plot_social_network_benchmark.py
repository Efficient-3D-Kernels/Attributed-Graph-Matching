"""
Publication-Quality Plotting for Social Network Benchmarks
Generates clean, aesthetic figures for the paper and deliverables:
  1. results/plots/social_network_accuracy_frontier.png / .pdf
  2. results/plots/social_network_runtime_pareto.png / .pdf
  3. results/plots/social_network_objective_collapse.png / .pdf

Updated to adhere strictly to:
  - Fast-WCS uniform naming: "Fast-WCS (Ours)" (highest performing V4 retained, v3/no-2opt removed)
  - Zero collision between lines, axes, markers, and text labels.
  - Exported to all paper, benchmark, and website directories.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DELIVERABLES_ROOT = PROJECT_ROOT.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

OUTPUT_DIRS = [
    PLOTS_DIR,
    DELIVERABLES_ROOT / "paper" / "figures" / "standalone",
    DELIVERABLES_ROOT / "benchmarks" / "figures" / "standalone",
    DELIVERABLES_ROOT / "els-cas-templates" / "figures" / "standalone",
    DELIVERABLES_ROOT / "els-cas-templates" / "figures",
    DELIVERABLES_ROOT / "els-cas-templates" / "figs",
    DELIVERABLES_ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures",
    DELIVERABLES_ROOT / "els-cas-templates" / "fastwcs_jocs_paper_package" / "figures" / "standalone",
    DELIVERABLES_ROOT / "paper_materials" / "figures" / "standalone",
    DELIVERABLES_ROOT / "paper_materials" / "figures",
    DELIVERABLES_ROOT / "website" / "figures",
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

raw_df = pd.read_csv(TABLES_DIR / "social_network_comprehensive_methods.csv")

# Filter: Retain only highest-performing Fast-WCS version (V4 WL-OA Augmented)
# Rename to unified 'Fast-WCS (Ours)' and drop intermediate v3 / No 2-Opt variants
df_filtered = raw_df[~raw_df["Method"].isin(["Fast-WCS v3 (Ours)", "Fast-WCS v3 (No 2-Opt)"])].copy()
df_filtered["Method"] = df_filtered["Method"].replace({"Fast-WCS v4 (WL-OA Augmented)": "Fast-WCS (Ours)"})

# Color palette & styling
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

color_map = {
    "Fast-WCS (Ours)": "#D32F2F",               # Crimson
    "WL-OA (Hungarian)": "#388E3C",            # Forest Green
    "Feature-Only Hungarian": "#F57C00",       # Amber
    "SciPy FAQ (Topology Only)": "#7B1FA2",    # Purple
    "Yang 2018 (RLX1F)": "#616161"             # Slate Gray
}

marker_map = {
    "Fast-WCS (Ours)": "o",
    "WL-OA (Hungarian)": "D",
    "Feature-Only Hungarian": "v",
    "SciPy FAQ (Topology Only)": "P",
    "Yang 2018 (RLX1F)": "X"
}

method_order = [
    "Fast-WCS (Ours)",
    "WL-OA (Hungarian)",
    "Feature-Only Hungarian",
    "SciPy FAQ (Topology Only)",
    "Yang 2018 (RLX1F)"
]


def save_all_formats(fig, name):
    for base in OUTPUT_DIRS:
        fig.savefig(base / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(base / f"{name}.pdf", bbox_inches="tight")
    print(f"[OK] Saved {name} (.png and .pdf) to all figure directories")


# =========================================================================
# FIGURE 1: ACCURACY VS NOISE (N=40 and N=100 Side-by-Side)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True, dpi=300)

for ax, n_val, title in zip(axes, [40, 100], ["(a) Subgraph Size N = 40 (L = 30)", "(b) Subgraph Size N = 100 (L = 75)"]):
    df_sub = df_filtered[df_filtered["N"] == n_val]
    for method in method_order:
        grp = df_sub[df_sub["Method"] == method]
        if grp.empty:
            continue
        grp = grp.sort_values("Noise_pct")
        c = color_map.get(method, "#333333")
        m = marker_map.get(method, "o")
        is_ours = "Fast-WCS" in method
        lw = 2.4 if is_ours else 1.8
        ls = "-" if (is_ours or "WL-OA" in method) else "--"
        z = 10 if is_ours else 5
        ax.plot(grp["Noise_pct"], grp["Accuracy_pct"], label=method, color=c, marker=m,
                linewidth=lw, linestyle=ls, markersize=6.5, zorder=z)
    
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Structural Perturbation Noise (%)", labelpad=6)
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks([0, 5, 10, 20, 30])

axes[0].set_ylabel("Ground-Truth Matching Accuracy (%)", labelpad=6)

# Unified horizontal legend below both subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5,
           frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.5, columnspacing=1.4)

plt.tight_layout()
fig.subplots_adjust(bottom=0.20)
save_all_formats(fig, "social_network_accuracy_frontier")
plt.close(fig)


# =========================================================================
# FIGURE 2: PARETO FRONTIER: ACCURACY VS RUNTIME (N=100, 30% NOISE)
# =========================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=300)
df_stress = df_filtered[(df_filtered["N"] == 100) & (df_filtered["Noise_pct"] == 30)]

# Plot Pareto frontier line: Feature-Only -> WL-OA -> Fast-WCS
fo_row = df_stress[df_stress["Method"] == "Feature-Only Hungarian"].iloc[0]
wl_row = df_stress[df_stress["Method"] == "WL-OA (Hungarian)"].iloc[0]
fw_row = df_stress[df_stress["Method"] == "Fast-WCS (Ours)"].iloc[0]

pareto_x = [fo_row["Runtime_ms"], wl_row["Runtime_ms"], fw_row["Runtime_ms"]]
pareto_y = [fo_row["Accuracy_pct"], wl_row["Accuracy_pct"], fw_row["Accuracy_pct"]]
ax.plot(pareto_x, pareto_y, color="0.45", ls="--", lw=1.4, zorder=2, label="Pareto Optimal Frontier")

custom_offsets = {
    "Fast-WCS (Ours)": (1.15, 3.5),             # To the right and above (Pareto line terminates here)
    "WL-OA (Hungarian)": (0.32, 4.0),           # To the left and above (above Pareto line, no collision)
    "Feature-Only Hungarian": (1.20, 1.2),      # To the right (below positive slope)
    "SciPy FAQ (Topology Only)": (1.18, 1.5),   # To the right
    "Yang 2018 (RLX1F)": (0.35, -2.5)           # To the left (well away from right plot border)
}

for _, row in df_stress.iterrows():
    m = row["Method"]
    c = color_map.get(m, "#333333")
    marker = marker_map.get(m, "o")
    is_ours = "Fast-WCS" in m
    ax.scatter(row["Runtime_ms"], row["Accuracy_pct"], color=c, marker=marker, s=130 if is_ours else 100,
               zorder=6, edgecolors="#1a202c" if is_ours else "none", linewidth=1.0)
    scale_x, delta_y = custom_offsets.get(m, (1.15, 1.5))
    ax.annotate(
        f" {m}", 
        (row["Runtime_ms"] * scale_x, row["Accuracy_pct"] + delta_y), 
        fontsize=9.2, 
        fontweight="bold" if is_ours else "normal",
        color=c if is_ours else "#222222"
    )

ax.set_xscale("log")
ax.set_xlabel("Wall-Clock Latency (ms, log scale)", labelpad=6)
ax.set_ylabel("Ground-Truth Accuracy (%) at 30% Noise", labelpad=6)
ax.set_title("Space-Time-Accuracy Pareto Trade-off (Stanford CollegeMsg, N = 100)", fontweight="bold", pad=10)
ax.set_ylim(-5, 115)
ax.set_xlim(0.4, 2500)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

plt.tight_layout()
save_all_formats(fig, "social_network_runtime_pareto")
plt.close(fig)


# =========================================================================
# FIGURE 3: OBJECTIVE VALUE STABILITY (LOG SCALE)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
df_n100 = df_filtered[df_filtered["N"] == 100].dropna(subset=["Objective"])

for method in method_order:
    grp = df_n100[df_n100["Method"] == method]
    if grp.empty:
        continue
    grp = grp.sort_values("Noise_pct")
    c = color_map.get(method, "#333333")
    m = marker_map.get(method, "o")
    is_ours = "Fast-WCS" in method
    ax.plot(grp["Noise_pct"], grp["Objective"] + 1e-4, label=method, color=c, marker=m,
            linewidth=2.4 if is_ours else 1.8, markersize=6.5, zorder=10 if is_ours else 5)

ax.set_yscale("log")
ax.set_xlabel("Structural Perturbation Noise (%)", labelpad=6)
ax.set_ylabel("WCS Objective Energy $F^*(X)$ (log scale)", labelpad=6)
ax.set_title("Optimization Robustness: Residual Energy vs Noise (N = 100)", fontweight="bold", pad=10)
ax.set_xticks([0, 5, 10, 20, 30])
ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

plt.tight_layout()
save_all_formats(fig, "social_network_objective_collapse")
plt.close(fig)

print("All social network benchmark plots successfully generated and synchronized!")
