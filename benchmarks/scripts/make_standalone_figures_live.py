"""Standalone publication figures from the LIVE benchmark CSVs only (one experiment per figure).

Each figure is written as 300-dpi PNG and vector PDF to benchmarks/figures/standalone/ and
paper/figures/standalone/. json/figures_manifest.json records, per figure, the source CSV files and
their SHA-256, so every plotted value is traceable to a live run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = lc.LIVE / "csv"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "lines.markersize": 4,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
ONE_COL, TWO_COL = 3.5, 7.16

STYLE = {  # method -> (color, marker, linestyle, linewidth, zorder, label)
    "Fast-WCS (Multi-Start, 10 paths)": ("#b22222", "D", "-", 2.5, 20, r"Fast-WCS (Multi-Start, $\mathcal{O}(N^2)$)"),
    "Fast-WCS v3 (default)": ("#d62728", "o", "--", 1.7, 15, r"Fast-WCS (Single-Start, $\mathcal{O}(N^2)$)"),
    "RLX1F (Yang 2018)": ("#008080", "^", "--", 1.4, 8, "RLX1F (Yang 2018)"),
    "RLX1 (Yang 2018)": ("#1f77b4", "s", "-", 1.4, 7, "RLX1 (Yang 2018)"),
    "FAQ": ("#2ca02c", "v", "-.", 1.3, 6, "SciPy FAQ"),
    "FGW (eps=1e-2)": ("#ff7f0e", "h", "-.", 1.3, 5, "FGW (eps=1e-2)"),
    "SM": ("#7f7f7f", "*", ":", 1.2, 4, r"SM (Gaussian $K$, $\mathcal{O}(N^4)$)"),
    "IPFP": ("#8c564b", "D", "--", 1.3, 5, r"IPFP (Gaussian $K$, $\mathcal{O}(N^4)$)"),
    "RRWM": ("#9467bd", "P", ":", 1.5, 5, r"RRWM (Gaussian $K$, $\mathcal{O}(N^4)$)"),
}
MANIFEST = {}


def save(fig, name, sources):
    for d in (lc.FIG_BENCH, lc.FIG_PAPER):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=300)
        fig.savefig(d / f"{name}.pdf")
    plt.close(fig)
    MANIFEST[name] = {"sources": {str(Path(s).relative_to(lc.ROOT)): lc.sha256(Path(s)) for s in sources},
                      "outputs": [f"{d.relative_to(lc.ROOT)}/{name}.{e}" for d in (lc.FIG_BENCH, lc.FIG_PAPER) for e in ("png", "pdf")]}
    print("figure", name, flush=True)


def _acc_plot(df, scen, xlabel, name, src, legend_cols=1):
    d = df[df.scenario == scen]
    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)
    for m in STYLE:
        g = d[d.method == m].sort_values("x")
        if g.empty:
            continue
        c, mk, ls, lw, z, lab = STYLE[m]
        is_ours = "Fast-WCS" in m
        ax.plot(g.x, g.acc_mean, color=c, marker=mk, ls=ls, lw=lw, ms=5.5 if is_ours else 4.0, zorder=z, label=lab)
        if is_ours:
            ax.fill_between(g.x, g.acc_ci95_lo, g.acc_ci95_hi, color=c, alpha=0.12, lw=0, zorder=z - 1)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Matching accuracy (%)", fontweight="bold")
    ax.set_ylim(-2, 104)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, ncol=2, fontsize=7.0,
              handlelength=2.2, columnspacing=1.0, handletextpad=0.4)
    save(fig, name, [src])


def fig_accuracy():
    src = CSV / "synthetic_results.csv"
    df = pd.read_csv(src)
    _acc_plot(df, "A_noise", r"Deformation noise $\sigma$", "fig_matching_accuracy_vs_noise", src)
    _acc_plot(df, "C_outliers", r"Outliers $M-L$", "fig_matching_accuracy_vs_outliers", src)
    _acc_plot(df, "B_size", r"Graph size $M$ ($N=M+5$, $L=M-5$)", "fig_synthetic_accuracy_vs_size", src)
    _acc_plot(df, "D_density", r"Edge density $\rho$", "fig_synthetic_accuracy_vs_density", src)


def fig_image():
    src = CSV / "image_results.csv"
    if not src.exists():
        return
    df = pd.read_csv(src).rename(columns={"group": "scenario"})
    _acc_plot(df, "outliers", r"Outliers $M-L$ (real images, $M=35$, $N=40$)", "fig_image_accuracy_vs_outliers", src)
    _acc_plot(df, "size", r"Graph size $M$ (real images, $N=M+5$, $L=M-5$)", "fig_image_accuracy_vs_size", src)


GRID_METHODS = [("fast", "Fast-WCS v3 (default)"), ("rlx1f", "RLX1F (Yang 2018)"), ("rlx1", "RLX1 exact-LP (Yang 2018)")]
GRID_COL = {"fast": "#b2182b", "rlx1f": "#053061", "rlx1": "#4393c3"}
RATIO_MK = {0.4: "o", 0.6: "s", 0.8: "^", 1.0: "D"}
RATIO_LS = {0.4: ":", 0.6: "-.", 0.8: "--", 1.0: "-"}


def fig_grid():
    src = CSV / "large_scale_grid.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    # runtime vs N
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    for key, lab in GRID_METHODS:
        for r in sorted(df.ratio.unique()):
            g = df[df.ratio == r].sort_values("N")
            ax.plot(g.N, g[f"{key}_ms_median"], color=GRID_COL[key], marker=RATIO_MK[r], ls=RATIO_LS[r], lw=1.1, ms=3.2)
            ax.fill_between(g.N, g[f"{key}_ms_q25"], g[f"{key}_ms_q75"], color=GRID_COL[key], alpha=0.08, lw=0)
    ax.set_yscale("log"); ax.set_xlabel(r"Graph size $N$ ($M=N$)"); ax.set_ylabel("Wall-clock runtime (ms)")
    h1 = [plt.Line2D([], [], color=GRID_COL[k], lw=1.6, label=l.replace(" (Yang 2018)", "")) for k, l in GRID_METHODS]
    h2 = [plt.Line2D([], [], color="0.3", marker=RATIO_MK[r], ls=RATIO_LS[r], lw=1, label=f"$L/N={r:.1f}$") for r in sorted(RATIO_MK)]
    l1 = ax.legend(handles=h1, loc="upper left", frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=h2, loc="lower right", frameon=False, ncol=2)
    save(fig, "fig_scaling_runtime_vs_n", [src])
    # speed-up vs N
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    for r in sorted(df.ratio.unique()):
        g = df[df.ratio == r].sort_values("N")
        ax.plot(g.N, g.speedup_vs_rlx1f_median, color="#053061", marker=RATIO_MK[r], ls=RATIO_LS[r], ms=3.2, lw=1.1)
        ax.plot(g.N, g.speedup_vs_rlx1_median, color="#4393c3", marker=RATIO_MK[r], ls=RATIO_LS[r], ms=3.2, lw=1.1)
    ax.axhline(1.0, color="0.4", lw=0.7)
    ax.set_yscale("log"); ax.set_xlabel(r"Graph size $N$ ($M=N$)"); ax.set_ylabel("Speed-up of Fast-WCS v3 (median, ×)")
    h1 = [plt.Line2D([], [], color="#053061", lw=1.6, label="vs. RLX1F"), plt.Line2D([], [], color="#4393c3", lw=1.6, label="vs. RLX1 (exact LP)")]
    l1 = ax.legend(handles=h1, loc="upper left", frameon=False); ax.add_artist(l1)
    ax.legend(handles=h2, loc="lower right", frameon=False, ncol=2)
    save(fig, "fig_speedup_factor_vs_baseline", [src])
    # empirical complexity slopes (log-log fit, N >= 40)
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    slopes = {}
    for key, lab in GRID_METHODS:
        g = df[df.ratio == 0.8].sort_values("N")
        ax.loglog(g.N, g[f"{key}_ms_median"], color=GRID_COL[key], marker="o", ls="none", ms=3.5)
        fit = g[g.N >= 40]
        b, a = np.polyfit(np.log(fit.N), np.log(fit[f"{key}_ms_median"]), 1)
        slopes[key] = float(b)
        xs = np.linspace(fit.N.min(), fit.N.max(), 50)
        ax.loglog(xs, np.exp(a) * xs ** b, color=GRID_COL[key], lw=1.2, label=f"{lab.replace(' (Yang 2018)', '')}: slope {b:.2f}")
    ax.set_xlabel(r"Graph size $N$ ($L=0.8N$, log scale)"); ax.set_ylabel("Runtime (ms, log scale)")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_empirical_complexity_slopes", [src])
    lc.write_json(lc.LIVE / "json" / "empirical_complexity_slopes.json",
                  {"fit": "log(ms) = a + b log(N), N >= 40, L/N = 0.8, medians of large_scale_grid.csv", "slopes": slopes})


def fig_ablation():
    src = CSV / "ablation_linesearch_calls.csv"
    src2 = CSV / "ablation_trials.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    df = df[df.stage.str.startswith(("S2", "S3"))]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.7))
    regimes = sorted(df.N.unique())
    pos, data, cols = [], [], []
    for i, N in enumerate(regimes):
        for j, (kind, col) in enumerate((("11-point grid (full objective evaluations)", "#2166ac"), ("Cardano quartic (Theorem 3)", "#b2182b"))):
            v = df[(df.N == N) & (df.line_search == kind)].ms.values * 1e3
            data.append(v); pos.append(i * 3 + j); cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.75, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1))
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c); b.set_alpha(0.55)
    ax.set_yscale("log"); ax.set_xticks([i * 3 + 0.5 for i in range(len(regimes))]); ax.set_xticklabels([f"$N={n}$" for n in regimes])
    ax.set_ylabel(r"Latency per line-search call ($\mu$s)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2166ac", alpha=0.55, label="11-point grid (Yang 2018)"),
                       plt.Rectangle((0, 0), 1, 1, color="#b2182b", alpha=0.55, label="Cardano quartic (Thm. 3)")],
              loc="upper left", frameon=False)
    save(fig, "fig_cardano_vs_gridsearch_latency", [src])
    t = pd.read_csv(src2)
    t = t[t.stage.str.startswith("S5")]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.8))
    for N, c in zip(sorted(t.N.unique()), ("#1b7837", "#762a83", "#b2182b")):
        g = t[t.N == N]
        ax.scatter(g.F_pre_2opt, g.F, s=9, color=c, alpha=0.75, label=f"$N={N}$", lw=0)
    lim = [min(t.F.min(), t.F_pre_2opt.min()) * 0.8, max(t.F.max(), t.F_pre_2opt.max()) * 1.2]
    ax.plot(lim, lim, color="0.3", lw=0.8, ls="--", label="no change")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"WCS objective before 2-opt, $F(X_{\mathrm{disc}})$"); ax.set_ylabel(r"After 2-opt (Prop. 2)")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_2opt_refinement_objective_drop", [src2])


STYLE3D = {
    "Fast-WCS v3 hierarchical": ("#d62728", "-", 2.4, "Fast-WCS v3 (Hierarchical, Ours)"),
    "IPFP + Kabsch": ("#8c564b", "--", 1.3, "IPFP + Kabsch"),
    "RRWM + Kabsch": ("#9467bd", ":", 1.4, "RRWM + Kabsch"),
    "FGW (eps=1e-2) + Kabsch": ("#ff7f0e", "-.", 1.3, "FGW (eps=1e-2) + Kabsch"),
    "FAQ + Kabsch": ("#2ca02c", "-", 1.2, "SciPy FAQ + Kabsch"),
    "ICP (24-start)": ("#333333", "-.", 1.3, "ICP (24-start)"),
    "ICP (single start)": ("#888888", ":", 1.1, "Single-Start ICP"),
    "RANSAC + FPFH + ICP": ("#1f77b4", "--", 1.2, "RANSAC + FPFH"),
    "CPD (rigid)": ("#17becf", ":", 1.2, "CPD (rigid)"),
}


def fig_3d():
    src = CSV / "superposition_3d_trials.csv"
    src2 = CSV / "superposition_3d_results.csv"
    if not src.exists():
        return
    t = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(3.7, 3.2), dpi=300)
    for m in STYLE3D:
        c, ls, lw, lab = STYLE3D[m]
        v = np.sort(t[(t.method == m) & t.rmsd.notna()].rmsd.values)
        if len(v) == 0:
            continue
        n_all = len(t[t.method == m])
        ax.step(v, np.arange(1, len(v) + 1) / n_all * 100, where="post", color=c, ls=ls, lw=lw, label=lab)
    for x in (0.1, 1.0):
        ax.axvline(x, color="0.5", lw=0.7, ls="--")
    ax.set_xscale("log"); ax.set_xlabel("RMSD over all ground-truth core atoms (Å, log scale)", fontweight="bold")
    ax.set_ylabel("Trials with RMSD ≤ x (%)", fontweight="bold"); ax.set_ylim(0, 102)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False, ncol=3, fontsize=6.5)
    save(fig, "fig_rmsd_cumulative_distribution", [src])

    # Pareto: median latency vs success < 1 A, pooled over scenarios
    rows = []
    for m in STYLE3D:
        g = t[t.method == m]
        if g.empty:
            continue
        rows.append((m, float(np.median(g.ms)), 100.0 * float(np.mean(g.rmsd < 1.0))))
    fig, ax = plt.subplots(figsize=(3.7, 3.0), dpi=300)
    front = [r for r in rows if not any((o[1] <= r[1] and o[2] >= r[2]) and (o[1] < r[1] or o[2] > r[2]) for o in rows)]
    fr = sorted(front, key=lambda r: r[1])
    ax.plot([r[1] for r in fr], [r[2] for r in fr], color="0.5", lw=1.0, ls="--", zorder=1, label="Pareto front")
    
    offsets = {
        "Fast-WCS v3 hierarchical": (10, 4),
        "FAQ + Kabsch": (10, -5),
        "IPFP + Kabsch": (-45, 5),
        "RRWM + Kabsch": (-55, -12),
        "FGW (eps=1e-2) + Kabsch": (-95, -4),
        "ICP (single start)": (10, -3),
        "ICP (24-start)": (10, 3),
        "RANSAC + FPFH + ICP": (10, -10),
        "CPD (rigid)": (10, 4),
    }

    for m, x, y in rows:
        if m not in STYLE3D:
            continue
        c = STYLE3D[m][0]
        is_ours = m.startswith("Fast-WCS")
        ax.scatter(x, y, color=c, s=55 if is_ours else 28, zorder=4,
                   edgecolor="k" if (m, x, y) in front else "none", lw=0.6)
        label_text = STYLE3D[m][3].replace(" + Kabsch", "+K").replace("Fast-WCS v3 ", "Fast-WCS ")
        xy_off = offsets.get(m, (8, 2))
        ax.annotate(label_text, (x, y), fontsize=6.8, xytext=xy_off, textcoords="offset points",
                    fontweight="bold" if is_ours else "normal",
                    color=c if is_ours else "black")
    ax.set_xscale("log"); ax.set_xlabel("Median runtime per alignment (ms, log scale)", fontweight="bold")
    ax.set_ylabel(r"Success rate, RMSD $<1$ Å (%)", fontweight="bold"); ax.set_ylim(-3, 114)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    save(fig, "fig_pareto_latency_vs_rmsd", [src, src2])


def main():
    fig_accuracy()
    fig_image()
    fig_grid()
    fig_ablation()
    fig_3d()
    lc.write_json(lc.LIVE / "json" / "figures_manifest.json", {"figures": MANIFEST, "provenance": lc.provenance([Path(__file__)])})


if __name__ == "__main__":
    main()
