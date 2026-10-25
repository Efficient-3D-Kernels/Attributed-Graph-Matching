"""Generate every number the manuscript quotes from the LIVE CSV/JSON outputs.

Writes
  paper/live/live_values.tex   \\LVdef{key}{value} for every number quoted in the text
  paper/live/tab_*.tex         the empirical tables (I synthetic, II image, III grid, IV 3D, V ablation, VI gradient)
The manuscript uses \\LV{key}; an undefined key is a hard LaTeX error, so text and tables cannot
silently drift from the CSV files. The same formatting functions are used for tables and text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV, JS = lc.LIVE / "csv", lc.LIVE / "json"
OUT = lc.ROOT / "paper" / "live"
VALUES: dict = {}

MKEY = {
    "Fast-WCS (Multi-Start, 10 paths)": "fwms",
    "Fast-WCS v3 (default)": "fwd", "Fast-WCS v3 (long path)": "fwl", "RLX1 (Yang 2018)": "rlxone",
    "RLX2 (Yang 2018)": "rlxtwo", "RLX1F (Yang 2018)": "rlxonef", "FAQ": "faq", "SM": "sm", "IPFP": "ipfp",
    "RRWM": "rrwm", "NGM (pretrained)": "ngm", "FGW (eps=1e-2)": "fgwa", "FGW (eps=1e-3)": "fgwb",
    "Partial GW": "pgw", "Partial FGW": "pgw",
}
MLABEL = {
    "Fast-WCS (Multi-Start, 10 paths)": r"\textbf{Fast-WCS (Multi-Start)}",
    "Fast-WCS v3 (default)": r"\textbf{Fast-WCS (Single-Start)}", "Fast-WCS v3 (long path)": r"\textbf{Fast-WCS (long path)}",
    "RLX1 (Yang 2018)": r"RLX1~\cite{yang2018algorithm}", "RLX2 (Yang 2018)": r"RLX2~\cite{yang2018algorithm}",
    "RLX1F (Yang 2018)": r"RLX1F~\cite{yang2018algorithm}", "FAQ": r"FAQ~\cite{vogelstein2015fast}",
    "SM": r"SM~\cite{leordeanu2005}", "IPFP": r"IPFP~\cite{leordeanu2009ipfp}", "RRWM": r"RRWM~\cite{cho2010rrwm}",
    "NGM (pretrained)": r"NGM (VOC)~\cite{wang2021ngm}", "FGW (eps=1e-2)": r"FGW, $\varepsilon{=}10^{-2}$~\cite{titouan2019optimal}",
    "FGW (eps=1e-3)": r"FGW, $\varepsilon{=}10^{-3}$~\cite{titouan2019optimal}", "Partial GW": r"Partial GW~\cite{chapel2020partial}",
    "Partial FGW": r"Partial FGW~\cite{chapel2020partial}",
}


# ------------------------------------------------------------------------------------ formatting
def f_acc(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.1f}"


def f_ms(v):
    if v is None or not np.isfinite(v):
        return "--"
    if v < 1:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:,.0f}".replace(",", "{,}")


def f_x(v):
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 1000 else f"{v:,.0f}".replace(",", "{,}"))


def f_rmsd(v):
    return "--" if v is None or not np.isfinite(v) else (f"{v:.3f}" if v < 1 else f"{v:.2f}")


def f_sci(v):
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


def put(key, val):
    if key in VALUES and VALUES[key] != val:
        raise KeyError(f"conflicting value for {key}")
    VALUES[key] = val


def xkey(x):
    s = f"{float(x):g}".replace(".", "p").replace("-", "m")
    return s


def bold_best(vals, higher=True):
    arr = np.array([np.nan if v is None else v for v in vals], float)
    if np.all(np.isnan(arr)):
        return [False] * len(vals)
    best = np.nanmax(arr) if higher else np.nanmin(arr)
    if higher and best <= 0:
        return [False] * len(vals)
    return [bool(np.isfinite(a) and abs(a - best) < 1e-9) for a in arr]


def cell(text, b):
    return rf"\textbf{{{text}}}" if b else text


# ---------------------------------------------------------------------------------- Table I synth
SYN_COLS = [("A_noise", 0.0, r"$\sigma{=}0$"), ("A_noise", 0.04, r"$\sigma{=}.04$"), ("A_noise", 0.1, r"$\sigma{=}.10$"),
            ("B_size", 20, r"$M{=}20$"), ("B_size", 40, r"$M{=}40$"), ("B_size", 60, r"$M{=}60$"),
            ("C_outliers", 0, r"$M{-}L{=}0$"), ("C_outliers", 10, r"$M{-}L{=}10$"),
            ("D_density", 0.2, r"$\rho{=}.2$"), ("D_density", 1.0, r"$\rho{=}1$")]


def synthetic():
    p = CSV / "synthetic_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    tr = pd.read_csv(CSV / "synthetic_trials.csv")
    methods = [m for m in MKEY if m in set(df.method)]
    for _, r in df.iterrows():
        k = f"syn.{r.scenario}.{xkey(r.x)}.{MKEY[r.method]}"
        put(k + ".acc", f_acc(r.acc_mean)); put(k + ".ms", f_ms(r.ms_median))
        put(k + ".fle", f_acc(100 * r.frac_F_le_gt)); put(k + ".fratio", f_x(r.F_ratio_median))
    for m in methods:
        for s in df.scenario.unique():
            put(f"syn.{s}.{MKEY[m]}.accavg", f_acc(df[(df.scenario == s) & (df.method == m)].acc_mean.mean()))
        put(f"syn.all.{MKEY[m]}.accavg", f_acc(df[df.method == m].acc_mean.mean()))
        put(f"syn.all.{MKEY[m]}.fle", f_acc(100 * tr[(tr.method == m) & tr.error.isna()].F_le_gt.mean()))
        put(f"syn.all.{MKEY[m]}.fratio", f_x(tr[(tr.method == m) & tr.error.isna()].F_ratio.median()))
        put(f"syn.all.{MKEY[m]}.ms", f_ms(tr[(tr.method == m) & tr.error.isna()].ms.median()))
        put(f"syn.all.{MKEY[m]}.errors", str(int(tr[tr.method == m].error.notna().sum())))
    # per-trial speed ratios vs Fast-WCS default
    piv = tr.pivot_table(index=["scenario", "setting", "trial"], columns="method", values="ms")
    for m in methods:
        if m != "Fast-WCS v3 (default)":
            put(f"syn.spd.{MKEY[m]}", f_x(float(np.median(piv[m] / piv["Fast-WCS v3 (default)"]))))
    # B_size: speed-ups at M = 60
    b = tr[(tr.scenario == "B_size") & (tr.M == 60)].pivot_table(index="trial", columns="method", values="ms")
    for m in methods:
        if m != "Fast-WCS v3 (default)":
            put(f"syn.spd.msixty.{MKEY[m]}", f_x(float(np.median(b[m] / b["Fast-WCS v3 (default)"]))))
    # paired Wilcoxon: Fast-WCS default vs RLX1F accuracy over all synthetic trials
    a = tr.pivot_table(index=["scenario", "setting", "trial"], columns="method", values="acc")
    for m in ("RLX1F (Yang 2018)", "RLX1 (Yang 2018)", "RRWM", "IPFP"):
        pv = lc.wilcoxon_p(a["Fast-WCS v3 (default)"].values, a[m].values)
        put(f"syn.wilcoxon.{MKEY[m]}", f_sci(pv) if pv is not None and pv > 0 else "1")
    pv = lc.wilcoxon_p(a["Fast-WCS v3 (long path)"].values, a["RLX1F (Yang 2018)"].values)
    put("syn.wilcoxon.fwl.rlxonef", f_sci(pv) if pv is not None and pv > 0 else "1")
    put("syn.ntrials", str(int(df.n_trials.max())))
    put("syn.ninstances", f"{len(tr[['scenario', 'setting', 'trial']].drop_duplicates()):,}".replace(",", "{,}"))
    rows = []
    for m in methods:
        accs, mss = [], []
        for scen, x, _ in SYN_COLS:
            g = df[(df.scenario == scen) & (np.isclose(df.x, x)) & (df.method == m)]
            accs.append(float(g.acc_mean.iloc[0]) if len(g) else None)
        for scen, x in (("A_noise", 0.04), ("B_size", 60)):
            g = df[(df.scenario == scen) & (np.isclose(df.x, x)) & (df.method == m)]
            mss.append(float(g.ms_median.iloc[0]) if len(g) else None)
        rows.append((m, accs, mss))
    best_acc = [bold_best([r[1][j] for r in rows]) for j in range(len(SYN_COLS))]
    best_ms = [bold_best([r[2][j] for r in rows], higher=False) for j in range(2)]
    head = " & ".join(c[2] for c in SYN_COLS)
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Synthetic WCS benchmark (protocol of~\cite{yang2018algorithm}, " + VALUES["syn.ntrials"] +
         r" trials per setting, identical instances for all solvers). Mean matching accuracy (\%) at representative settings of the "
         r"four scenarios and median wall-clock time (ms). Bold: best per column. Full noise and outlier sweeps: Figs.~\ref{fig:acc_noise}--\ref{fig:acc_outliers}.}",
         r"\label{tab:synthetic}", r"\fontsize{6.6}{7.9}\selectfont", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(SYN_COLS) + "cc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Noise ($M{=}30,N{=}35,L{=}25$)} & \multicolumn{3}{c}{Size ($N{=}M{+}5,L{=}M{-}5$)} & "
         r"\multicolumn{2}{c}{Outliers ($M{=}30,N{=}35$)} & \multicolumn{2}{c}{Density} & \multicolumn{2}{c}{Time (ms)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(l){12-13}",
         r"Method & " + head + r" & $\sigma{=}.04$ & $M{=}60$ \\", r"\midrule"]
    for i, (m, accs, mss) in enumerate(rows):
        cells = [cell(f_acc(a), best_acc[j][i]) for j, a in enumerate(accs)] + [cell(f_ms(v), best_ms[j][i]) for j, v in enumerate(mss)]
        L.append(MLABEL[m] + " & " + " & ".join(cells) + r" \\")
        if m in ("Fast-WCS v3 (long path)", "RLX1F (Yang 2018)"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (OUT / "tab_synthetic.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------- Table II image
IMG_COLS = [("size", 20, r"$M{=}20$"), ("size", 30, r"$M{=}30$"), ("size", 40, r"$M{=}40$"),
            ("outliers", 0, r"$0$"), ("outliers", 4, r"$4$"), ("outliers", 10, r"$10$")]


def image():
    p = CSV / "image_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    tr = pd.read_csv(CSV / "image_trials.csv")
    js = json.loads((JS / "image_results.json").read_text())
    put("img.npairs", str(len(js["test_pairs"])))
    put("img.ntrials", str(int(df.n_trials.max())))
    put("img.reproj", f"{js['gt_mean_reprojection_px']:.2f}")
    methods = [m for m in MKEY if m in set(df.method)]
    for _, r in df.iterrows():
        k = f"img.{r.group}.{xkey(r.x)}.{MKEY[r.method]}"
        put(k + ".acc", f_acc(r.acc_mean)); put(k + ".ms", f_ms(r.ms_median))
    for m in methods:
        for g in ("size", "outliers"):
            put(f"img.{g}.{MKEY[m]}.accavg", f_acc(df[(df.group == g) & (df.method == m)].acc_mean.mean()))
        put(f"img.all.{MKEY[m]}.accavg", f_acc(df[df.method == m].acc_mean.mean()))
        put(f"img.all.{MKEY[m]}.ms", f_ms(tr[(tr.method == m) & tr.error.isna()].ms.median()))
        put(f"img.all.{MKEY[m]}.errors", str(int(tr[tr.method == m].error.notna().sum())))
    piv = tr.pivot_table(index=["group", "setting", "pair", "sampling"], columns="method", values="ms")
    a = tr.pivot_table(index=["group", "setting", "pair", "sampling"], columns="method", values="acc")
    for m in methods:
        if m != "Fast-WCS v3 (default)":
            put(f"img.spd.{MKEY[m]}", f_x(float(np.median(piv[m] / piv["Fast-WCS v3 (default)"]))))
            pv = lc.wilcoxon_p(a["Fast-WCS v3 (default)"].values, a[m].values)
            put(f"img.wilcoxon.{MKEY[m]}", f_sci(pv) if pv is not None and pv > 0 else "1")
    rows = []
    for m in methods:
        accs = []
        for g, x, _ in IMG_COLS:
            q = df[(df.group == g) & np.isclose(df.x, x) & (df.method == m)]
            accs.append(float(q.acc_mean.iloc[0]) if len(q) else None)
        ms = float(tr[(tr.method == m) & tr.error.isna()].ms.median())
        rows.append((m, accs, ms, float(df[df.method == m].acc_mean.mean())))
    ba = [bold_best([r[1][j] for r in rows]) for j in range(len(IMG_COLS))]
    bm = bold_best([r[2] for r in rows], higher=False)
    bv = bold_best([r[3] for r in rows])
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Real-image keypoint matching (Oxford Affine~\cite{mikolajczyk2005comparison}, " + VALUES["img.npairs"] +
         r" image pairs, Delaunay graphs, SIFT unary, $\alpha{=}0.5$, " + VALUES["img.ntrials"] +
         r" trials per setting). Mean accuracy (\%), mean over all 11 settings, and median time (ms) over all trials.}",
         r"\label{tab:image}", r"\fontsize{6.2}{7.4}\selectfont", r"\setlength{\tabcolsep}{1.6pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Size ($L{=}M{-}5$)} & \multicolumn{3}{c}{Outliers $M{-}L$} & & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Method & " + " & ".join(c[2] for c in IMG_COLS) + r" & Avg. & ms \\", r"\midrule"]
    for i, (m, accs, ms, avg) in enumerate(rows):
        lab = MLABEL[m].replace("Fast-WCS v3 ", "Fast-WCS ")
        L.append(lab + " & " + " & ".join(cell(f_acc(v), ba[j][i]) for j, v in enumerate(accs)) +
                 f" & {cell(f_acc(avg), bv[i])} & {cell(f_ms(ms), bm[i])}" + r" \\")
        if m in ("Fast-WCS v3 (long path)", "RLX1F (Yang 2018)"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (OUT / "tab_image.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------- Table III grid
def grid():
    p = CSV / "large_scale_grid.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        k = f"grid.{int(r.N)}.{xkey(r.ratio)}"
        for c in ("fast", "long", "rlx1f", "rlx1"):
            put(f"{k}.{c}.ms", f_ms(r[f"{c}_ms_median"])); put(f"{k}.{c}.acc", f_acc(r[f"{c}_acc_mean"]))
        put(f"{k}.spd.rlxonef", f_x(r.speedup_vs_rlx1f_median)); put(f"{k}.spd.rlxone", f_x(r.speedup_vs_rlx1_median))
        put(f"{k}.fle.rlxonef", f_acc(100 * r.fast_F_le_rlx1f_rate)); put(f"{k}.fle.rlxone", f_acc(100 * r.fast_F_le_rlx1_rate))
    for c, k in (("rlx1f", "rlxonef"), ("rlx1", "rlxone")):
        put(f"grid.spd.{k}.min", f_x(df[f"speedup_vs_{c}_median"].min()))
        put(f"grid.spd.{k}.max", f_x(df[f"speedup_vs_{c}_median"].max()))
        put(f"grid.fle.{k}.mean", f_acc(100 * df[f"fast_F_le_{c}_rate"].mean()))
        put(f"grid.dacc.{k}.mean", f"{(df.fast_acc_mean - df[f'{c}_acc_mean']).mean():+.1f}")
    for ratio, g in df.groupby("ratio"):
        rk = xkey(ratio)
        for c in ("fast", "long", "rlx1f", "rlx1"):
            put(f"grid.avg.{rk}.{c}.acc", f_acc(g[f"{c}_acc_mean"].mean()))
            put(f"grid.avg.{rk}.{c}.fratio", f_x(float(np.median(g[f"{c}_F_ratio_median"]))))
        for c, k in (("rlx1f", "rlxonef"), ("rlx1", "rlxone")):
            put(f"grid.avg.{rk}.spd.{k}", f_x(float(np.median(g[f"speedup_vs_{c}_median"]))))
    put("grid.ntrials", str(int(df.n_trials.max())))
    put("grid.ncells", str(len(df)))
    s = JS / "empirical_complexity_slopes.json"
    if s.exists():
        for k, v in json.loads(s.read_text())["slopes"].items():
            put(f"grid.slope.{k}", f"{v:.2f}")
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Large-scale $(N,L)$ grid ($M{=}N$, $\sigma{=}.03$, $\rho{=}.5$, " + VALUES["grid.ntrials"] +
         r" trials per cell). Median time (ms), median per-trial speed-up of Fast-WCS v3 (default) and mean accuracy (\%).}",
         r"\label{tab:grid}", r"\fontsize{5.9}{7.0}\selectfont", r"\setlength{\tabcolsep}{1.3pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}rrrrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Median time (ms)} & \multicolumn{2}{c}{Speed-up ($\times$)} & \multicolumn{3}{c}{Accuracy (\%)} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-10}",
         r"$N$ & $L$ & Fast-WCS & RLX1F & RLX1 & /RLX1F & /RLX1 & Fast & R1F & R1 \\", r"\midrule"]
    for N in sorted(df.N.unique()):
        for _, r in df[df.N == N].sort_values("ratio").iterrows():
            L.append(f"{int(r.N)} & {int(r.L)} & {f_ms(r.fast_ms_median)} & {f_ms(r.rlx1f_ms_median)} & {f_ms(r.rlx1_ms_median)} & "
                     f"{f_x(r.speedup_vs_rlx1f_median)} & {f_x(r.speedup_vs_rlx1_median)} & {f_acc(r.fast_acc_mean)} & "
                     f"{f_acc(r.rlx1f_acc_mean)} & {f_acc(r.rlx1_acc_mean)}" + r" \\")
        if N != df.N.max():
            L.append(r"\addlinespace[1pt]")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (OUT / "tab_grid.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------- Table IV 3D
M3D = {"Fast-WCS v3 hierarchical": "fw", "Fast-WCS v3 (graph stage only)": "fwg", "FAQ + Kabsch": "faq", "SM + Kabsch": "sm",
       "RRWM + Kabsch": "rrwm", "IPFP + Kabsch": "ipfp", "FGW (eps=1e-2) + Kabsch": "fgw", "Partial GW + Kabsch": "pgw",
       "ICP (single start)": "icp", "ICP (24-start)": "icpms", "RANSAC + FPFH + ICP": "ransac", "CPD (rigid)": "cpd"}
L3D = {"Fast-WCS v3 hierarchical": r"\textbf{Fast-WCS v3 (hierarchical)}", "Fast-WCS v3 (graph stage only)": r"\textbf{Fast-WCS v3 (graph only)}",
       "FAQ + Kabsch": r"FAQ~\cite{vogelstein2015fast} + Kabsch", "SM + Kabsch": r"SM~\cite{leordeanu2005} + Kabsch",
       "RRWM + Kabsch": r"RRWM~\cite{cho2010rrwm} + Kabsch", "IPFP + Kabsch": r"IPFP~\cite{leordeanu2009ipfp} + Kabsch",
       "FGW (eps=1e-2) + Kabsch": r"FGW~\cite{titouan2019optimal} + Kabsch", "Partial GW + Kabsch": r"Partial GW~\cite{chapel2020partial} + Kabsch",
       "ICP (single start)": r"ICP~\cite{besl1992icp}", "ICP (24-start)": r"ICP, 24 starts~\cite{besl1992icp}",
       "RANSAC + FPFH + ICP": r"RANSAC+FPFH~\cite{rusu2009fpfh}+ICP", "CPD (rigid)": r"CPD (rigid)~\cite{myronenko2010cpd}"}
SC3D = [("S1_near_rigid", "S1 near-rigid"), ("S2_deformation", "S2 deformation"), ("S3_outlier_heavy", "S3 outlier-heavy")]


def three_d():
    p = CSV / "superposition_3d_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    tr = pd.read_csv(CSV / "superposition_3d_trials.csv")
    js = json.loads((JS / "superposition_3d_results.json").read_text())
    put("td.ntrials", str(int(df.n_trials.max())))
    put("td.nmol", str(len(js["molecules"])))
    for s, _ in SC3D:
        put(f"td.{s}.oracle", f_rmsd(float(df[df.scenario == s].oracle_rmsd_median.iloc[0])))
    for _, r in df.iterrows():
        k = f"td.{r.scenario}.{M3D[r.method]}"
        put(k + ".sone", f_acc(r.success_lt_1A_pct)); put(k + ".stenth", f_acc(r.success_lt_0p1A_pct))
        put(k + ".rmsd", f_rmsd(r.rmsd_median)); put(k + ".ms", f_ms(r.ms_median))
        put(k + ".acc", f_acc(r.match_acc_mean) if pd.notna(r.match_acc_mean) else "--")
    for m, k in M3D.items():
        g = tr[tr.method == m]
        ok = g[g.error.isna()]
        put(f"td.all.{k}.sone", f_acc(100 * float(np.mean(ok.rmsd < 1.0)) * len(ok) / max(len(g), 1)))
        put(f"td.all.{k}.stenth", f_acc(100 * float(np.mean(ok.rmsd < 0.1)) * len(ok) / max(len(g), 1)))
        put(f"td.all.{k}.ms", f_ms(float(ok.ms.median())))
        put(f"td.all.{k}.rmsd", f_rmsd(float(ok.rmsd.median())))
        put(f"td.all.{k}.det", f_acc(100 * float(ok.proper_rotation.mean())))
        put(f"td.all.{k}.refl", f_acc(100 * float(ok.reflection_corrected.mean())))
        put(f"td.all.{k}.errors", str(int(g.error.notna().sum())))
    L = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{3D superposition of all-atom drug conformers (" + VALUES["td.ntrials"] +
         r" trials per scenario, random $\mathrm{SO}(3)$ pose). Success rates (\%) for RMSD $<1$\,\AA{} and $<0.1$\,\AA{}, median RMSD (\AA) over "
         r"\emph{all} ground-truth core atoms (untrimmed) and median time (ms). The last column is the share of returned rotations with "
         r"$\det\mathbf{R}{=}{+}1$ over all 3 scenarios. Oracle RMSD (Kabsch on the true pairs): S1 " + VALUES["td.S1_near_rigid.oracle"] +
         r", S2 " + VALUES["td.S2_deformation.oracle"] + r", S3 " + VALUES["td.S3_outlier_heavy.oracle"] +
         r"\,\AA; the $<$0.1\,\AA{} threshold is therefore unreachable in S2. Bold: best per column.}",
         r"\label{tab:3d}", r"\fontsize{6.4}{7.7}\selectfont", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "cccc" * 3 + "c@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{lab} ($\sigma{{=}}{SCENARIOS_SIG[s]}$\,\AA, $M{{=}}N{{=}}{SCENARIOS_M[s]}$, $L{{=}}36$)}}"
                            for s, lab in SC3D) + r" & \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         r"Method & " + " & ".join([r"$<$1\AA & $<$0.1\AA & RMSD & ms"] * 3) + r" & $\det{=}{+}1$ \\", r"\midrule"]
    meth = [m for m in M3D if m in set(df.method)]
    best = {}
    for s, _ in SC3D:
        g = df[df.scenario == s].set_index("method")
        for col, hi in (("success_lt_1A_pct", True), ("success_lt_0p1A_pct", True), ("rmsd_median", False), ("ms_median", False)):
            b = bold_best([float(g.loc[m, col]) for m in meth], higher=hi)
            best[(s, col)] = dict(zip(meth, b))
    for m in meth:
        cells = []
        for s, _ in SC3D:
            r = df[(df.scenario == s) & (df.method == m)].iloc[0]
            cells += [cell(f_acc(r.success_lt_1A_pct), best[(s, "success_lt_1A_pct")][m]),
                      cell(f_acc(r.success_lt_0p1A_pct), best[(s, "success_lt_0p1A_pct")][m]),
                      cell(f_rmsd(r.rmsd_median), best[(s, "rmsd_median")][m]), cell(f_ms(r.ms_median), best[(s, "ms_median")][m])]
        cells.append(VALUES[f"td.all.{M3D[m]}.det"])
        L.append(L3D[m] + " & " + " & ".join(cells) + r" \\")
        if m in ("Fast-WCS v3 (graph stage only)", "Partial GW + Kabsch"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    (OUT / "tab_3d.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


SCENARIOS_SIG = {"S1_near_rigid": "0.02", "S2_deformation": "0.25", "S3_outlier_heavy": "0.05"}
SCENARIOS_M = {"S1_near_rigid": "40", "S2_deformation": "40", "S3_outlier_heavy": "46"}


# ---------------------------------------------------------------------------------- Table V ablation
STAGE_LAB = {"S0": r"S0 Yang RLX1F~\cite{yang2018algorithm}", "S1": r"S1 + Lemma~\ref{lem:rank1} ($\mathbf{U}{=}\mathbf{r}\mathbf{r}^T$)",
             "S2": r"S2 + Thm.~\ref{thm:gradient} (exact $\nabla$)", "S3": r"S3 + Thm.~\ref{thm:quartic_line_search} (Cardano)",
             "S4": r"S4 + Burkard oracle (Sec.~\ref{subsec:burkard})", "S5": r"S5 + Prop.~\ref{prop:2opt} (2-opt)",
             "REF": r"Fast-WCS v3 defaults"}


def ablation():
    p = CSV / "ablation_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    calls = pd.read_csv(CSV / "ablation_linesearch_calls.csv")
    tr = pd.read_csv(CSV / "ablation_trials.csv")
    solved = (tr.assign(s=tr.acc >= 90).groupby(["N", "stage"]).s.mean() * 100).to_dict()
    put("abl.ntrials", str(int(df.n_trials.max())))
    for _, r in df.iterrows():
        st = r.stage.split()[0]
        k = f"abl.{int(r.N)}.{st}"
        put(k + ".solved", f_acc(solved[(r.N, r.stage)]))
        put(k + ".ms", f_ms(r.ms_median)); put(k + ".spd", f_x(r.cumulative_speedup_median))
        put(k + ".df", f"{r.objective_reduction_vs_S0_pct_median:+.1f}"); put(k + ".acc", f_acc(r.acc_mean))
        put(k + ".fratio", f_x(r.F_ratio_to_gt_median))
        if pd.notna(r.get("stage_speedup_vs_previous", np.nan)):
            put(k + ".step", f_x(r.stage_speedup_vs_previous))
        if st in ("S0", "S1") and pd.notna(r.get("identical_to_OriginalGNCCP_2018_rate", np.nan)):
            put(k + ".identical", f_acc(100 * r.identical_to_OriginalGNCCP_2018_rate))
        if st == "S5":
            put(k + ".twoopt.drop", f_acc(r.twoopt_objective_drop_pct_median)); put(k + ".twoopt.rate", f_acc(100 * r.twoopt_improved_rate))
            put(k + ".twoopt.ms", f_ms(r.twoopt_ms_median))
        if pd.notna(r.get("ls_share_of_runtime_pct", np.nan)):
            put(k + ".lsshare", f_acc(r.ls_share_of_runtime_pct))
    for N in sorted(calls.N.unique()):
        g = calls[(calls.N == N) & calls.stage.str.startswith("S2")].ms.median() * 1e3
        c = calls[(calls.N == N) & calls.stage.str.startswith("S3")].ms.median() * 1e3
        put(f"abl.{N}.ls.grid.us", f_ms(g)); put(f"abl.{N}.ls.cardano.us", f_ms(c)); put(f"abl.{N}.ls.ratio", f_x(g / c))
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Cumulative component ablation (one component added per stage; linear schedule, 10 levels $\times$ 6 FW steps; "
         + VALUES["abl.ntrials"] + r" instances per regime, $M{=}N$, $L{=}0.8N$). Median time, median per-trial speed-up over S0, median "
         r"objective change vs.\ S0 ($+$: lower $F$), mean accuracy and share of instances solved (accuracy $\geq 90\%$).}",
         r"\label{tab:ablation}", r"\fontsize{6.0}{7.2}\selectfont", r"\setlength{\tabcolsep}{1.5pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
         r"Stage & ms & $\times$S0 & $\Delta F$ (\%) & Acc. (\%) & Solved (\%) \\", r"\midrule"]
    for N in sorted(df.N.unique()):
        L.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{df[df.N == N].regime.iloc[0].split()[0]} regime, $N{{=}}{int(N)}$, $L{{=}}{int(round(0.8 * N))}$}}}} \\")
        for _, r in df[df.N == N].iterrows():
            st = r.stage.split()[0]
            L.append(f"{STAGE_LAB[st]} & {f_ms(r.ms_median)} & {f_x(r.cumulative_speedup_median)} & "
                     f"{r.objective_reduction_vs_S0_pct_median:+.1f} & {f_acc(r.acc_mean)} & {f_acc(solved[(r.N, r.stage)])}" + r" \\")
        if N != df.N.max():
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    # Table VI gradient micro-benchmark
    g = pd.read_csv(CSV / "gradient_microbenchmark.csv")
    for _, r in g.iterrows():
        k = f"grad.{int(r.N)}"
        put(k + ".thm", f"{r.ms_theorem1:.3f}"); put(k + ".rlx", f"{r.ms_rlx1_eq11:.3f}"); put(k + ".fd", f_ms(r.ms_central_fd))
        put(k + ".spdfd", f_x(r.speedup_vs_fd)); put(k + ".spdrlx", f_x(r.speedup_vs_rlx1)); put(k + ".err", f_sci(r.theorem1_vs_fd_max_rel_err))
    put("grad.maxerr", f_sci(g.theorem1_vs_fd_max_rel_err.max()))
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{One gradient evaluation (single thread, minimum of 30 repetitions): Theorem~\ref{thm:gradient} vs.\ the "
         r"RLX1 gradient (Eq.~\ref{eq:rlx1_grad}) and central finite differences (FD, $2MN$ objective calls) of the true objective.}",
         r"\label{tab:gradient}", r"\fontsize{6.2}{7.4}\selectfont", r"\setlength{\tabcolsep}{2.0pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}rrrrrrc@{}}", r"\toprule",
         r"$N$ & Thm.~1 (ms) & RLX1 (ms) & FD (ms) & /FD & /RLX1 & max rel.\ err. vs FD \\", r"\midrule"]
    for _, r in g.iterrows():
        L.append(f"{int(r.N)} & {r.ms_theorem1:.3f} & {r.ms_rlx1_eq11:.3f} & {f_ms(r.ms_central_fd)} & {f_x(r.speedup_vs_fd)} & "
                 f"{f_x(r.speedup_vs_rlx1)} & ${f_sci(r.theorem1_vs_fd_max_rel_err)}$" + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (OUT / "tab_gradient.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------- audit / calib
def audit_and_calibration():
    a = JS / "engine_audit.json"
    if a.exists():
        j = json.loads(a.read_text())
        put("audit.grad.relerr", f_sci(j["gradient_check"]["relative_error"]))
        put("audit.parity.cases", str(j["parity"]["cases"]))
        put("audit.parity.identical", str(j["parity"]["bitwise_identical"]))
        put("audit.cubic.n", f"{j['cubic']['n']:,}".replace(",", "{,}"))
        put("audit.cubic.missed", str(j["cubic"]["missed_roots"]))
        put("audit.cubic.worst", f_sci(j["cubic"]["worst_root_error"]))
        put("audit.cubic.argminfail", str(j["cubic"]["quartic_argmin_worse_than_200k_grid"]))
        dg = j["degenerate_graphs"]["cases"]
        put("audit.degenerate.cases", str(len(dg) if isinstance(dg, list) else dg))
        for t in j["timing"]:
            put(f"audit.time.{t['N']}.pre", f_ms(t["pre_audit_median_ms"])); put(f"audit.time.{t['N']}.post", f_ms(t["audited_median_ms"]))
            put(f"audit.time.{t['N']}.spd", f_x(t["speedup"]))
    c = JS / "fastwcs_calibration.json"
    if c.exists():
        cj = json.loads(c.read_text())
        j = cj["long_path"]["config"]
        put("cal.long.levels", str(j["max_iter"])); put("cal.long.fw", str(j["fw_max_iter"]))
        put("cal.fw.ninst", str(cj["protocol"]["n_instances"])); put("cal.fw.nconf", str(cj["all"] if isinstance(cj["all"], int) else len(cj["all"])))
        put("cal.fw.budget", f_ms(cj["protocol"]["latency_budget_ms"]))
        for k in ("default", "long_path"):
            kk = "def" if k == "default" else "long"
            put(f"cal.fw.{kk}.acc", f_acc(cj[k]["acc_mean"])); put(f"cal.fw.{kk}.ms", f_ms(cj[k]["ms_median"]))
    c = JS / "baseline_calibration.json"
    if c.exists():
        bj = json.loads(c.read_text())
        put("cal.bl.ninst", str(bj["protocol"]["n_instances"]))
        names = {"fastwcs": "fw", "rlx1f": "rlxonef", "faq": "faq", "pgw": "pgw", "fgw_cg": "fgwcg"}
        for scen, sk in (("shared_er", "er"), ("threshold", "thr")):
            for k, v in bj["summary"].get(scen, {}).items():
                if "|" in k:
                    m, aff = k.split("|")
                    key = f"{m}.{'wcs' if aff == 'wcs' else 'g' + aff.split(':')[1].replace('.', 'p')}"
                else:
                    key = names.get(k, k)
                put(f"cal.bl.{sk}.{key}.acc", f_acc(v["acc_mean"]))
                put(f"cal.bl.{sk}.{key}.ms", f_ms(v["ms_median"]))
                put(f"cal.bl.{sk}.{key}.fle", f_acc(100 * v["frac_F_le_gt"]))


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        kk = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk + "."))
        elif isinstance(v, (int, float, str, bool)):
            out[kk] = (f_sci(v) if isinstance(v, float) and v != 0 and abs(v) < 1e-3 else str(v)).replace("_", r"\_")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic(); image(); grid(); three_d(); ablation(); audit_and_calibration()
    lines = ["% AUTO-GENERATED by benchmarks/scripts/make_paper_tables_live.py from benchmarks/live_runs -- do not edit"]
    for k in sorted(VALUES):
        lines.append(rf"\LVdef{{{k}}}{{{VALUES[k]}}}")
    (OUT / "live_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lc.write_json(JS / "paper_values.json", {"values": VALUES, "provenance": lc.provenance([Path(__file__)])})
    print(f"{len(VALUES)} live values; tables in {OUT}")


if __name__ == "__main__":
    main()
