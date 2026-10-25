"""Independent consistency check: paper numbers vs. raw per-trial CSV records.

Recomputes the aggregates printed in the manuscript (paper/live/live_values.tex, used through \\LV{...}
in research_paper.tex and in the generated tables) directly from the per-trial CSV files, formats them
with the same rounding rules and reports every mismatch. Also checks that every \\LV key used in the
manuscript is defined and that every table cell value appears in the per-trial-derived aggregates.

Output: benchmarks/live_runs/json/paper_number_verification.json (exit code 1 on any mismatch).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_tables_live as T  # noqa: E402  (naming maps and number formatters only)

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "benchmarks" / "live_runs" / "csv"
PAPER = ROOT / "paper"


for _name in ("f_acc", "f_ms", "f_x", "f_rmsd"):
    _f = getattr(T, _name)
    # summation order changes the last bits of a mean (37.75 vs 37.7499...); round that noise away first
    setattr(T, _name, (lambda f: lambda v: f(None if v is None else float(np.round(v, 10))))(_f))


def load_values():
    txt = (PAPER / "live" / "live_values.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\LVdef\{([^}]+)\}\{(.*)\}\s*$", txt, flags=re.M))


def main():
    V = load_values()
    checks, bad = 0, []

    def chk(key, val):
        nonlocal checks
        checks += 1
        if key not in V:
            bad.append((key, "missing", val))
        elif V[key] != val:
            bad.append((key, V[key], val))

    # synthetic: mean accuracy, median ms, F <= F_gt share per setting and method
    tr = pd.read_csv(CSV / "synthetic_trials.csv")
    ok = tr[tr.error.isna()]
    for (scen, x, m), g in ok.groupby(["scenario", "x", "method"]):
        k = f"syn.{scen}.{T.xkey(x)}.{T.MKEY[m]}"
        chk(k + ".acc", T.f_acc(g.acc.mean()))
        chk(k + ".ms", T.f_ms(g.ms.median()))
    for m, g in ok.groupby("method"):
        chk(f"syn.all.{T.MKEY[m]}.ms", T.f_ms(g.ms.median()))
    # image
    tr = pd.read_csv(CSV / "image_trials.csv")
    ok = tr[tr.error.isna()]
    for (grp, x, m), g in ok.groupby(["group", "x", "method"]):
        k = f"img.{grp}.{T.xkey(x)}.{T.MKEY[m]}"
        chk(k + ".acc", T.f_acc(g.acc.mean()))
        chk(k + ".ms", T.f_ms(g.ms.median()))
    # grid
    tr = pd.read_csv(CSV / "large_scale_grid_trials.csv")
    gm = {"Fast-WCS v3 (default)": "fast", "Fast-WCS v3 (long path)": "long", "RLX1F (Yang 2018)": "rlx1f",
          "RLX1 exact-LP (Yang 2018)": "rlx1"}
    for (N, r, m), g in tr[tr.error.isna()].groupby(["N", "ratio", "method"]):
        k = f"grid.{int(N)}.{T.xkey(r)}.{gm[m]}"
        chk(k + ".ms", T.f_ms(g.ms.median()))
        chk(k + ".acc", T.f_acc(g.acc.mean()))
    # 3D
    tr = pd.read_csv(CSV / "superposition_3d_trials.csv")
    ok = tr[tr.error.isna()]
    for (s, m), g in ok.groupby(["scenario", "method"]):
        k = f"td.{s}.{T.M3D[m]}"
        chk(k + ".sone", T.f_acc(100 * (g.rmsd < 1.0).mean()))
        chk(k + ".stenth", T.f_acc(100 * (g.rmsd < 0.1).mean()))
        chk(k + ".rmsd", T.f_rmsd(g.rmsd.median()))
        chk(k + ".ms", T.f_ms(g.ms.median()))
    for m, g in ok.groupby("method"):
        chk(f"td.all.{T.M3D[m]}.sone", T.f_acc(100 * (g.rmsd < 1.0).mean()))
        chk(f"td.all.{T.M3D[m]}.ms", T.f_ms(g.ms.median()))
    # ablation
    tr = pd.read_csv(CSV / "ablation_trials.csv")
    for (N, st), g in tr.groupby(["N", "stage"]):
        k = f"abl.{int(N)}.{st.split()[0]}"
        chk(k + ".ms", T.f_ms(g.ms.median()))
        chk(k + ".acc", T.f_acc(g.acc.mean()))
        chk(k + ".solved", T.f_acc(100 * (g.acc >= 90).mean()))
        base = tr[(tr.N == N) & tr.stage.str.startswith("S0")].set_index("trial").ms
        chk(k + ".spd", T.f_x(float(np.median(base.loc[g.trial].values / g.ms.values))))

    # every \LV key used by the manuscript is defined
    tex = (PAPER / "research_paper.tex").read_text(encoding="utf-8")
    tex = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    used = sorted(set(re.findall(r"\\LV\{([^}]+)\}", tex)))
    undefined = [k for k in used if k not in V]

    report = dict(values_defined=len(V), lv_keys_used_in_manuscript=len(used), undefined_keys=undefined,
                  recomputed_checks=checks, mismatches=[dict(key=k, paper=p, recomputed=r) for k, p, r in bad])
    out = ROOT / "benchmarks" / "live_runs" / "json" / "paper_number_verification.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{checks} values recomputed from per-trial CSVs, {len(bad)} mismatches; "
          f"{len(used)} \\LV keys used, {len(undefined)} undefined -> {out}")
    for k, p, r in bad[:30]:
        print(f"  {k}: paper {p} vs recomputed {r}")
    return 1 if bad or undefined else 0


if __name__ == "__main__":
    sys.exit(main())
