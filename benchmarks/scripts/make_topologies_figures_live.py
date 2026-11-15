"""
Generate standalone publication-grade figures and LaTeX table for the Multi-Topology benchmark:
  - fig_topologies_noise.pdf / .png
  - fig_topologies_outliers.pdf / .png
  - fig_topologies_size.pdf / .png
  - tab_topologies.tex

Updated:
  - Only show Fast-WCS (Multi-Start) and Fast-WCS (Single-Start) for our method.
  - V4 is strictly reserved for social analysis, removed here.
  - Clean styling with no line or text collisions.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import live_common as lc

CSV_RESULTS = lc.LIVE / "csv" / "topologies_results.csv"
FIG_DIRS = [
    lc.FIG_PAPER,
    lc.FIG_BENCH,
    ROOT / "els-cas-templates" / "figures" / "standalone",
    ROOT / "els-cas-templates" / "figures",
    ROOT / "els-cas-templates" / "figs",
    ROOT / "paper_materials" / "figures" / "standalone",
    ROOT / "paper_materials" / "figures",
    ROOT / "website" / "figures",
]
for d in FIG_DIRS:
    d.mkdir(parents=True, exist_ok=True)
TABLE_DIR = ROOT / "paper" / "live"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette: Only highest Fast-WCS variants (Multi-Start & Single-Start)
STYLE = {
    "Fast-WCS (Multi-Start, 10 paths)": dict(color="#D32F2F", marker="D", lw=2.2, ms=6.5, ls="-", label="Fast-WCS (Multi-Start)", zorder=10),
    "Fast-WCS (Single-Start)": dict(color="#E57373", marker="o", lw=1.6, ms=5.5, ls="--", label="Fast-WCS (Single-Start)", zorder=9),
    "RLX1F (Yang 2018)": dict(color="#FB8C00", marker="^", lw=1.5, ms=5, ls=":", label=r"RLX1F (Yang '18)", zorder=7),
    "FAQ": dict(color="#8E24AA", marker="x", lw=1.3, ms=5, ls="--", label=r"FAQ", zorder=6),
    "SM": dict(color="#43A047", marker="v", lw=1.3, ms=5, ls="-", label=r"SM", zorder=5),
    "IPFP": dict(color="#00897B", marker="p", lw=1.5, ms=5, ls="-", label=r"IPFP", zorder=4),
    "RRWM": dict(color="#00ACC1", marker="h", lw=1.5, ms=5, ls="-", label=r"RRWM", zorder=3),
}

TOPO_TITLES = {
    "shared_er": "Erdős–Rényi (ER)",
    "random_regular": "Random Regular (Uniform Degree $d$)",
    "barabasi_albert": "Barabási–Albert (Scale-Free Hubs)",
}


def plot_sweep(df: pd.DataFrame, scenario: str, x_col: str, xlabel: str, out_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True, dpi=300)
    
    sub = df[df.scenario == scenario]
    topos = ["shared_er", "random_regular", "barabasi_albert"]
    
    for ax, topo in zip(axes, topos):
        t_sub = sub[sub.topology == topo]
        for m, st in STYLE.items():
            m_sub = t_sub[t_sub.method == m].sort_values(x_col)
            if len(m_sub) == 0:
                continue
            xs = m_sub[x_col].values
            ys = m_sub["acc_mean"].values
            y_err_lo = np.clip(ys - m_sub["acc_ci95_lo"].values, 0, None)
            y_err_hi = np.clip(m_sub["acc_ci95_hi"].values - ys, 0, None)
            ax.errorbar(
                xs, ys, yerr=[y_err_lo, y_err_hi],
                fmt=st["marker"], color=st["color"], ecolor=st["color"],
                elinewidth=1.1, capsize=3.5, capthick=1.2,
                ls=st["ls"], lw=st["lw"], ms=st["ms"],
                label=st["label"], zorder=st["zorder"],
            )
            
        ax.set_title(TOPO_TITLES[topo], fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=6)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(-3, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="both", labelsize=9.5)
        
    axes[0].set_ylabel("Matching Accuracy (%)", fontsize=11, labelpad=6)
    
    # Unified legend at bottom - 4 columns, clean spacing
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.5,
               handletextpad=0.5, columnspacing=1.5)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    for d in FIG_DIRS:
        pdf_path = d / f"{out_name}.pdf"
        png_path = d / f"{out_name}.png"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved {out_name} (.pdf, .png) to all figure directories", flush=True)
    plt.close(fig)


def make_latex_table(df: pd.DataFrame):
    """Generate summary LaTeX table tab_topologies.tex."""
    methods = [
        "Fast-WCS (Multi-Start, 10 paths)",
        "Fast-WCS (Single-Start)",
        "RLX1F (Yang 2018)",
        "FAQ",
        "SM",
        "IPFP",
        "RRWM",
    ]
    
    topos = ["shared_er", "random_regular", "barabasi_albert"]
    
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Multi-Topology Benchmark: Matching accuracy (\%) across Erdős–Rényi, Random Regular, and Barabási–Albert scale-free graphs (20 Monte Carlo trials per setting). Bold: proposed Fast-WCS formulations.}",
        r"\label{tab:topologies}",
        r"\fontsize{7.2}{8.8}\selectfont",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Erdős–Rényi (ER)}} & \multicolumn{3}{c}{\textbf{Random Regular (Degree $d$)}} & \multicolumn{3}{c}{\textbf{Barabási–Albert (Scale-Free)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"Method & $\sigma{=}0$ & $M{-}L{=}0$ & $M{=}60$ & $\sigma{=}0$ & $M{-}L{=}0$ & $M{=}60$ & $\sigma{=}0$ & $M{-}L{=}0$ & $M{=}60$ \\",
        r"\midrule",
    ]
    
    for m in methods:
        m_label = m.replace("Fast-WCS (Multi-Start, 10 paths)", r"\textbf{Fast-WCS (Multi-Start)}")
        m_label = m_label.replace("Fast-WCS (Single-Start)", r"\textbf{Fast-WCS (Single-Start)}")
        m_label = m_label.replace("RLX1F (Yang 2018)", r"RLX1F~\cite{yang2018algorithm}")
        m_label = m_label.replace("FAQ", r"FAQ~\cite{vogelstein2015fast}")
        m_label = m_label.replace("SM", r"SM~\cite{leordeanu2005}")
        m_label = m_label.replace("IPFP", r"IPFP~\cite{leordeanu2009ipfp}")
        m_label = m_label.replace("RRWM", r"RRWM~\cite{cho2010rrwm}")
        
        vals = []
        for topo in topos:
            s0 = df[(df.topology == topo) & (df.scenario == "A_noise") & (df.x == 0.0) & (df.method == m)]
            v_s0 = f"{s0['acc_mean'].values[0]:.1f}" if len(s0) else "--"
            o0 = df[(df.topology == topo) & (df.scenario == "B_outliers") & (df.x == 0) & (df.method == m)]
            v_o0 = f"{o0['acc_mean'].values[0]:.1f}" if len(o0) else "--"
            m60 = df[(df.topology == topo) & (df.scenario == "C_size") & (df.x == 60) & (df.method == m)]
            v_m60 = f"{m60['acc_mean'].values[0]:.1f}" if len(m60) else "--"
            vals.extend([v_s0, v_o0, v_m60])
            
        lines.append(f"{m_label} & " + " & ".join(vals) + r" \\")
        if m == "Fast-WCS (Single-Start)":
            lines.append(r"\midrule")
            
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ])
    
    for t_dir in [TABLE_DIR, ROOT / "els-cas-templates" / "live", ROOT / "paper_materials" / "live"]:
        t_dir.mkdir(parents=True, exist_ok=True)
        tex_path = t_dir / "tab_topologies.tex"
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Saved LaTeX table to {tex_path}", flush=True)


def main():
    if not CSV_RESULTS.exists():
        print(f"Results CSV not found at {CSV_RESULTS}!", flush=True)
        return
    df = pd.read_csv(CSV_RESULTS)
    print(f"Loaded {len(df)} aggregated rows from {CSV_RESULTS}", flush=True)
    
    plot_sweep(df, "A_noise", "x", r"Coordinate Noise $\sigma$", "fig_topologies_noise")
    plot_sweep(df, "B_outliers", "x", r"Outliers Count $M - L$", "fig_topologies_outliers")
    plot_sweep(df, "C_size", "x", r"Graph Size $M$ ($N=M+5, L=M-5$)", "fig_topologies_size")
    make_latex_table(df)
    print("Multi-Topology figures & tables generated successfully!", flush=True)


if __name__ == "__main__":
    main()
