# Fast-WCS benchmark protocol

## What this runner evaluates

`benchmark_paper_synthetic_wcs.py` runs the base paper's synthetic **WCS and PIW sweep settings**: noise, graph size, planted match count/outliers, and edge density. Every accuracy run is paired: all selected methods receive the same generated instance and seed. The reported task objective is the discrete alpha=1 WCS objective, in addition to planted correspondence accuracy.

The generator is an explicit reconstruction of the published protocol. The paper specifies coordinate sampling, planted correspondences, distance weights, graph noise, and sweep settings, but leaves implementation details such as exact topology coupling, PRNG, and some baseline parameters unspecified. The runner records those assumptions. It must therefore be described as a **reproducible reconstruction**, not bit-for-bit reproduction of the authors' code.

Generic SM, RRWM, IPFP, and GRAMPA return scores rather than native exact-L WCS solutions. The runner builds the same WCS-derived pairwise affinity for PyGMtools and applies an augmented assignment projection that guarantees exactly L matches. These rows are explicitly adapters. The v3, v2, and local 2018-style implementations receive the same alpha=1 structural task; the local 2018-style implementation is not represented as the authors' original binary.

## Full synthetic run

From the project root, run:

```powershell
python benchmarks/benchmark_paper_synthetic_wcs.py `
  --runs 30 `
  --timing-runs 50 `
  --warmups 1 `
  --threads 1 `
  --sweeps all `
  --methods all `
  --out benchmarks/paper_ready_wcs_results.json
```

The default run uses 30 paired accuracy/objective instances and 50 independent timing instances per setting, plus one excluded warm-up per method and setting. Timing order is randomized within each repetition. The output contains raw samples and seeds, 95% bootstrap intervals, per-instance paired accuracy deltas against v3, exact-L validation, failures, environment/package information, and SHA-256 hashes of the benchmark and solver files. The corresponding CSV is written next to the JSON.

The runner refuses to overwrite an existing output path. Choose a new `--out` path for each run.

## Smoke run

Use a limited setting only to confirm imports and output shape:

```powershell
python benchmarks/benchmark_paper_synthetic_wcs.py `
  --sweeps noise `
  --methods "Fast-WCS v3,Fast-WCS v2,Local 2018-style RLX1F" `
  --runs 2 --timing-runs 2 --warmups 1 --limit-settings 1 `
  --out benchmarks/paper_ready_smoke.json
```

The script labels this as a smoke run. Its values are not paper results.

## Submission scope and remaining work

This runner hardens the synthetic WCS/PIW comparison; it does **not** by itself complete a publication-quality empirical study. Before making comparative paper claims:

1. Run the complete sweeps and archive the JSON/CSV with the exact source hashes and environment.
2. Verify the reconstructed generator against the original authors' implementation or supplementary material, especially topology coupling and edge-noise details.
3. Add verified original implementations for the base paper's unavailable methods (or state clearly that those methods could not be reproduced); do not treat a local reconstruction or generic adapter as the original method.
4. Evaluate the base paper's real image datasets and any claimed application datasets with their official ground-truth and scoring procedures. Those data are not included in this synthetic runner.
5. Report all failures, paired uncertainty, and runtime distributions. Do not select hyperparameters on the same test instances; tune on separate development seeds and freeze them before the reported run.
6. Treat molecular point-cloud registration, conformer comparison, and protein-ligand docking as separate tasks. A ligand-derived point set with synthetic outliers is not a docking benchmark; a 3D claim needs chemically valid inputs, known atom correspondences or an explicit docking endpoint, and task-appropriate baselines.

The prior `benchmark_comprehensive_domain.py` is exploratory only. Its CYP19A1-labeled section constructs transformed point sets with synthetic outliers from individual ligand coordinates; it does not evaluate ligand-to-ligand matching or protein docking. Its solver inputs and exact-L adapters also differ, so do not use its results for paper comparisons.

The common-input, version-pinned, objective/runtime reporting approach is consistent with the reproducibility motivation of the [ECCV 2022 graph-matching benchmark](https://vislearn.github.io/gmbench/). This script does not run that benchmark's vision datasets or containerized methods. The base-paper protocol reference is Yang, Qiao & Liu, [IEEE TNNLS 2018](https://doi.org/10.1109/TNNLS.2017.2712794).
