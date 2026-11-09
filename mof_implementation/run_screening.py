"""run_screening.py -- virtual screening of MOFs with the Fast-WCS kernels (paper Fig. 1 / Fig. 7h analogue).

Task (our design; the paper's screening data is molecular and is not available for MOFs):
  * library = random MOFs from CoRE MOF that have a value for a chosen numeric property
    (e.g. PLD, LCD, AV_VF from the CoRE MOF csv, e.g. ASR_20241119.csv);
  * "active" = property value in the top --active_frac of the library;
  * each MOF is represented by its local fragment (n atoms around the first metal atom);
  * for a query active, every other library MOF is ranked by k_2D, k_3D or k_hybrid(beta);
  * ROC-AUC (actives vs inactives) is averaged over queries (95% bootstrap CI over queries).
Baselines: fragment element-composition cosine similarity, and random (0.5).
gamma_2D, gamma_3D = 1/median over all scored pairs (label-free); beta is swept and ALL values are reported.
Usage:
 python run_screening.py --cif_dir "data/.../CR/ASR" --csv data/ASR_20241119.csv           (lists columns)
 python run_screening.py --cif_dir "..." --csv ... --label_col PLD --library 300 --queries 30 --out results_screen
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
import mof_data as md, pipeline3d as p3, expcommon as ec


def auc(scores, labels):
    r = rankdata(scores); n1 = labels.sum(); n0 = len(labels) - n1
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def find_name_column(df, stems):
    best, bc = 0.0, None
    for c in df.columns:
        v = df[c].astype(str).str.strip().str.replace(r"\.cif$", "", regex=True)
        o = v.isin(stems).mean()
        if o > best:
            best, bc = o, c
    return bc, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True); ap.add_argument("--filter", default=None)
    ap.add_argument("--csv", required=True); ap.add_argument("--label_col", default=None)
    ap.add_argument("--active_frac", type=float, default=0.25)
    ap.add_argument("--library", type=int, default=300); ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--n_atoms", type=int, default=30)
    ap.add_argument("--omega", type=float, default=50.0); ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_screen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); ec.log_environment(args.out, args)
    paths = md.find_cifs(args.cif_dir, args.filter)
    stem2path = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
    df = pd.read_csv(args.csv)
    namecol, ov = find_name_column(df, set(stem2path))
    print(f"CSV columns: {list(df.columns)}\nname column: {namecol} (overlap with CIF names: {ov:.2%})")
    if args.label_col is None or namecol is None:
        print("-> choose a numeric property with --label_col"); return
    lc = {c.lower(): c for c in df.columns}[args.label_col.lower()]
    df["_stem"] = df[namecol].astype(str).str.strip().str.replace(r"\.cif$", "", regex=True)
    df["_y"] = pd.to_numeric(df[lc], errors="coerce")
    df = df[df._stem.isin(stem2path) & df._y.notna()].drop_duplicates("_stem")
    rng = np.random.default_rng(args.seed)
    df = df.sample(frac=1, random_state=args.seed)
    lib = []                                                    # (stem, y, pos, Z)
    for _, r in df.iterrows():
        if len(lib) >= args.library:
            break
        try:
            at = md.load_structure(stem2path[r._stem]); pos, Z = p3.fragment(at, args.n_atoms)
            lib.append((r._stem, r._y, pos, Z))
        except Exception:
            continue
    S = len(lib); y = np.array([l[1] for l in lib])
    thr = np.quantile(y, 1 - args.active_frac); lab = (y >= thr).astype(int)
    print(f"library {S}, actives {lab.sum()} (property '{lc}' >= {thr:.4g})")
    qidx = rng.choice(np.flatnonzero(lab), min(args.queries, lab.sum()), replace=False)
    elems = sorted({int(z) for l in lib for z in l[3]})
    comp = np.array([[np.mean(l[3] == z) for z in elems] for l in lib])
    comp /= np.linalg.norm(comp, axis=1, keepdims=True)
    # all query-library pairs
    FL = np.full((len(qidx), S), np.nan); RM = np.full((len(qidx), S), np.nan)
    for a, q in enumerate(qidx):
        for j in range(S):
            if j == q:
                continue
            m = p3.match_fragments(lib[q][2], lib[q][3], lib[j][2], lib[j][3], args.omega, args.alpha)
            FL[a, j], RM[a, j] = m["F_over_L"], m["rmsd"]
        print(f"query {a + 1}/{len(qidx)}", flush=True)
    g2 = 1.0 / np.nanmedian(FL); g3 = 1.0 / np.nanmedian(RM ** 2)
    print(f"gamma_2D={g2:.4g} gamma_3D={g3:.4g} (1/median)")
    rows = []
    for a, q in enumerate(qidx):
        msk = np.arange(S) != q
        entries = {"composition cosine (baseline)": comp[msk] @ comp[q]}
        for b in args.betas:
            k2, k3, kh = p3.kernels(FL[a, msk], RM[a, msk], g2, g3, b)
            nm = "k_2D (beta=0)" if b == 0 else ("k_3D (beta=1)" if b == 1 else f"k_hybrid (beta={b:g})")
            entries[nm] = kh
        entries["random (reference)"] = rng.random(msk.sum())
        for nm, sc in entries.items():
            rows.append(dict(query=lib[q][0], method=nm, auc=auc(sc, lab[msk])))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(args.out, "per_query_auc.csv"), index=False)
    summ = ec.summarise(res, ["method"], ["auc"]); summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.DataFrame({"stem": [l[0] for l in lib], "property": y, "active": lab}).to_csv(
        os.path.join(args.out, "library.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
