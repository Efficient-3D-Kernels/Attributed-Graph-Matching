"""expcommon.py -- solver registry, metrics (Sec. VII-D), bootstrap CIs, Wilcoxon, reproducibility log."""
import glob
import hashlib
import json
import os
import platform
import sys
import numpy as np
import pandas as pd
import fastwcs as fw


def make_solvers(alpha, include=("single", "multi", "rlx1f", "faq", "feature"), multi_starts=10, seed=0):
    S = {}
    if "single" in include:
        S["Fast-WCS (single-start)"] = lambda I: fw.fastwcs(I["AG"], I["AH"], I["C"], I["L"], alpha=alpha, seed=seed)
    if "multi" in include:
        S["Fast-WCS (multi-start)"] = lambda I: fw.fastwcs(I["AG"], I["AH"], I["C"], I["L"], alpha=alpha,
                                                           starts=multi_starts, seed=seed)
    if "rlx1f" in include:
        S["RLX1F (re-impl.)"] = lambda I: fw.rlx1f_baseline(I["AG"], I["AH"], I["C"], I["L"], alpha)
    if "faq" in include:
        S["SciPy FAQ"] = lambda I: fw.faq_baseline(I["AG"], I["AH"], I["C"], I["L"], alpha, seed)
    if "fgw" in include:
        S["FGW (POT)"] = lambda I: fw.fgw_baseline(I["AG"], I["AH"], I["C"], I["L"], alpha)
    if "feature" in include:
        S["Feature-only Hungarian"] = lambda I: fw.feature_only_baseline(I["AG"], I["AH"], I["C"], I["L"], alpha)
    return S


def evaluate(inst, res, alpha):
    L = inst["L"]
    rows, cols = np.asarray(res["rows"]), np.asarray(res["cols"])
    gt = dict(zip(inst["gt_rows"].tolist(), inst["gt_cols"].tolist()))
    acc = sum(gt.get(int(r), -1) == int(c) for r, c in zip(rows, cols)) / L
    C = inst["C"]
    Fgt = fw.wcs_objective_pairs(inst["AG"], inst["AH"], C, inst["gt_rows"], inst["gt_cols"], alpha)
    F = fw.wcs_objective_pairs(inst["AG"], inst["AH"], C, rows, cols, alpha)
    out = dict(acc=100 * acc, F=F, F_gt=Fgt, F_ratio=F / Fgt if Fgt > 1e-12 else np.nan,
               F_le_gt=float(F <= Fgt + 1e-9), time_ms=1000 * res["time"], n_pairs=len(rows))
    if "ZG" in inst:
        out["elem_consistency"] = 100 * float(np.mean(inst["ZG"][rows] == inst["ZH"][cols]))
    if "posG" in inst:                                       # Phase 2: Kabsch RMSD (Angstrom)
        out["rmsd_A"] = fw.kabsch_rmsd(inst["posG"][rows], inst["posH"][cols])
    return out


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarise(df, group_cols, metric_cols):
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        d = dict(zip(group_cols, key)); d["n"] = len(g)
        for m in metric_cols:
            if m in g:
                lo, hi = boot_ci(g[m].values)
                d[m + "_mean"] = g[m].mean(); d[m + "_lo"] = lo; d[m + "_hi"] = hi
                if m == "time_ms":
                    d["time_ms_median"] = g[m].median()
        rows.append(d)
    return pd.DataFrame(rows)


def paired_wilcoxon(df, ref="Fast-WCS (single-start)", metric="acc"):
    from scipy.stats import wilcoxon
    out = []
    piv = df.pivot_table(index="instance", columns="method", values=metric)
    for m in piv.columns:
        if m == ref or ref not in piv:
            continue
        d = (piv[ref] - piv[m]).dropna()
        p = np.nan
        if len(d) > 5 and np.any(d != 0):
            try:
                p = wilcoxon(d).pvalue
            except Exception:
                pass
        out.append(dict(ref=ref, other=m, metric=metric, mean_diff=d.mean(), p_value=p, n=len(d)))
    return pd.DataFrame(out)


def log_environment(outdir, args):
    import scipy, ase
    here = os.path.dirname(os.path.abspath(__file__))
    sha = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
           for p in glob.glob(os.path.join(here, "*.py"))}
    info = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, ase=ase.__version__, pandas=pd.__version__,
                args=vars(args), source_sha256=sha)
    json.dump(info, open(os.path.join(outdir, "environment.json"), "w"), indent=2, default=str)
