"""run_all_mof.py -- ONE command that runs the whole MOF study (2D + 3D parts of the paper).

  python run_all_mof.py --cif_dir "data/CoREMOF2019_public_v2_20241118/CR/ASR"
  (optional, for the virtual-screening part) add:  --csv data/ASR_20241119.csv

Steps:  1) engine audit   2) 2D benchmark (noise/outlier/size sweeps, 100 MOFs)
        3) 3D conformer alignment + Kabsch RMSD   4) 2D/3D/hybrid kernel virtual screening (needs --csv)
Each step writes its own results folder; finished steps are skipped on re-run.
"""
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--cif_dir", required=True)
ap.add_argument("--csv", default=None, help="CoRE MOF property csv, e.g. data/ASR_20241119.csv")
ap.add_argument("--label_col", default=None, help="property column for screening (default: PLD, else LCD)")
ap.add_argument("--n_structures", type=int, default=100)
a = ap.parse_args()
py = sys.executable


def run(cmd, done_file=None):
    if done_file and os.path.exists(done_file):
        print("SKIP (already done):", done_file); return
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print("step failed with code", r.returncode); sys.exit(r.returncode)


run([py, "selftest.py"])
run([py, "run_mof.py", "--cif_dir", a.cif_dir, "--n_structures", str(a.n_structures), "--out", "results_mof"],
    "results_mof/summary.csv")
conf = [py, "run_conformer.py", "--cif_dir", a.cif_dir, "--n_structures", "60", "--sigma", "0.05", "--out", "results_conf"]
run(conf, "results_conf/summary.csv")
if a.csv:
    import pandas as pd
    cols = {c.lower(): c for c in pd.read_csv(a.csv, nrows=1).columns}
    lc = a.label_col or next((cols[k] for k in ("pld", "lcd") if k in cols), None)
    if lc is None:
        print("could not find PLD/LCD column; columns are:", list(cols.values()), "-> rerun with --label_col")
    else:
        run([py, "run_screening.py", "--cif_dir", a.cif_dir, "--csv", a.csv, "--label_col", lc,
             "--library", "300", "--queries", "30", "--out", "results_screen"], "results_screen/summary.csv")
else:
    print("\n(no --csv given: virtual-screening step skipped)")
print("\nALL DONE. Results: results_mof/  results_conf/  results_screen/  (summary.csv + PNG plots in each)")
