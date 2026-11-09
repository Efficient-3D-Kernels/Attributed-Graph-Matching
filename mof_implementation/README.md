# Fast-WCS on MOFs (CoRE MOF 2019) -- run order
pip install -r requirements.txt
python selftest.py                                   # engine audit (gradient, quartic, Cardano, oracle, 2-opt)
python download_coremof.py data                      # Zenodo record 3677685, CoREMOF2019_public_v2.zip (~81 MB)
python run_mof.py --cif_dir data/CoREMOF2019 --n_structures 100 --out results_mof
python run_mof.py --cif_dir data/CoREMOF2019 --n_structures 100 --omegas 50 0 --out results_mof_omega   # optional ablation
python run_synthetic.py --sizes 50 100 200 500 1000 --rhos 0.1 0.3 0.5 --trials 10 --out results_syn
Set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 for timing runs (paper Sec. VII-B).
Outputs: per_trial.csv, summary.csv (mean, 95% bootstrap CI, median time), wilcoxon_*.csv, PNG plots, environment.json

# Added: 3D phase, kernels, conformer alignment, virtual screening
pip install pot            # optional, enables the FGW baseline
python run_conformer.py --cif_dir "data/CoREMOF2019_public_v2_20241118/CR/ASR" --n_structures 60 --sigma 0.05 --out results_conf
python run_screening.py --cif_dir "data/CoREMOF2019_public_v2_20241118/CR/ASR" --csv data/ASR_20241119.csv      # prints CSV columns
python run_screening.py --cif_dir "data/CoREMOF2019_public_v2_20241118/CR/ASR" --csv data/ASR_20241119.csv --label_col PLD --library 300 --queries 30 --out results_screen
