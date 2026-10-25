# Review and results for the pasted graph-matching benchmarks

## What the two pasted scripts intend to benchmark

Both scripts generate **synthetic spatially correlated graph pairs** with a planted partial correspondence. They vary noise, graph size, unmatched nodes, and edge density, then compare GNCCP variants with SciPy FAQ and POT transport methods. The “rigorous” script additionally calls out larger instances up to roughly 100 target nodes; the “comprehensive” script has the broader parameter sweeps.

These scripts do **not** benchmark the 3D molecular/atom-mapping problem. The generators use 2D coordinates and an in-frame Euclidean unary cost. They also do not reproduce the base paper's alpha=1 graph-only experiments: both use `alpha=0.7`, which adds coordinate information to the graph objective.

## Problems found and corrections used for the run

- Both scripts import `gnccp_advanced` and `gnccp_generalized`; the workspace files are `gnccp_advanced_v3.py` and `gnccp_v2_classic.py`.
- The comprehensive generator used Python's randomized `hash()` in seed construction, so results could differ by process. The adapted runner uses explicit deterministic seeds.
- At `sigma=0`, the comprehensive generator still forced at least one graph perturbation and perturbed only `A_G`. The adapted runner performs zero changes at zero noise and perturbs both graphs independently.
- SciPy FAQ minimizes its QAP score. The pasted version passed two positive adjacency matrices, which favors low overlap. The adapted baseline negates one matrix, then selects `L` pairs using unary coordinate cost. It remains an adapter, not an exact partial-QAP method.
- POT and FGW return soft transport plans, not exact `L` bijections. The adapted runner uses an augmented assignment projection to produce exactly `L` unique pairs, and labels these rows as adapters.
- The original local GNCCP baseline's line search and final score omitted the unary term when `alpha < 1`. I corrected it in [gnccp_baseline_2018.py](/C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/gnccp_baseline_2018.py); this preserves its `alpha=1` behavior and makes the feature-assisted run evaluate the objective it claims to optimize.
- The pasted “original paper GNCCP” wrapper actually pointed to the local generalized solver, not original author code. The adapted run labels the in-workspace 2018-style solver as a reconstruction.

## Results: four comprehensive sweeps

The run used 10 paired seeds per setting, 25 settings total, and exact-`L` outputs for all methods. The table reports unweighted averages of setting means; accuracy is planted-pair accuracy and runtime is the average of per-setting mean milliseconds on this machine.

| Method | Accuracy | Common objective | Runtime |
|---|---:|---:|---:|
| Fast-WCS v3 | 53.3% | **8.49** | 14.1 ms |
| Fast-WCS v2 | 52.8% | 8.60 | 11.6 ms |
| Local 2018-style RLX1F reconstruction | 37.3% | 13.82 | 33.2 ms |
| SciPy FAQ + unary truncation adapter | 8.8% | 35.46 | 3.7 ms |
| Coordinate-only Hungarian | **79.0%** | 14.05 | 0.12 ms |
| POT partial OT + exact-`L` projection | 76.7% | 14.81 | 0.85 ms |
| POT FGW + exact-`L` projection | 61.0% | 21.85 | 1.92 ms |

The coordinate-only result is the key finding: because matched points share a coordinate frame and differ by only small noise, a plain exact-`L` assignment on Euclidean distances already recovers about 79% of pairs. POT receives essentially that same unary cost and gets similar accuracy. This means these data make spatial proximity an easier signal than graph structure. The feature-assisted benchmark therefore cannot establish that POT is a stronger graph-matching method than v3, and it is not a clean test of v3's graph-matching contribution.

Within this setup, v3 and v2 are close in accuracy. V3 has a slightly better mean common objective, but averages about 21% more runtime. V3 leads v2 on the size and outlier sweeps, while v2 is slightly ahead on noise and density. Those small differences should be treated as descriptive: 10 seeds per setting is limited, and this is not the base paper's generator.

## Larger scale checks from the “rigorous” script

I also ran its three intended sizes with 10 seeds each: `(M,N,L)=(20,30,15)`, `(40,60,30)`, and `(70,100,50)`. These use the same feature-assisted alpha=0.7 setup and adapters described above.

| Size | Fast-WCS v3 accuracy / time | Fast-WCS v2 accuracy / time | Coordinate-only accuracy | POT partial OT accuracy |
|---|---:|---:|---:|---:|
| 20 × 30, L=15 | 60.0% / 16.0 ms | 54.0% / 13.9 ms | 90.0% | 80.0% |
| 40 × 60, L=30 | 42.3% / 27.4 ms | 35.0% / 22.8 ms | 67.7% | 62.3% |
| 70 × 100, L=50 | 32.4% / 60.4 ms | 32.4% / 54.6 ms | 48.0% | 50.0% |

At 70 × 100, v3 and v2 tie on mean accuracy; v3 takes about 11% longer. Both GNCCP methods beat the local RLX1F reconstruction on accuracy, but coordinate-only matching and partial OT remain strong because the generator exposes the matched coordinate frame.

## How to read these comparisons

The common objective is recomputed for every exact-`L` output, but it is not each baseline's native loss. FAQ uses a full padded assignment followed by unary truncation; POT/FGW use transport, then projection. Their accuracy and common objective are useful diagnostic outputs, not fully like-for-like optimizer comparisons. The direct coordinate-only baseline demonstrates that this generator has a strong non-graph shortcut.

The base-paper synthetic comparison remains the separate alpha=1 benchmark in `paper_synthetic_eccv_protocol_results.json`. These new results are a supplementary feature-assisted stress test. They should not be merged into one headline score or presented as a reproduction of Yang et al. or GMBench.

## Artifacts

- Comprehensive sweeps: `comprehensive_adapted_results.json`
- Larger scale cases: `rigorous_scale_adapted_results.json`
- Adapted runner: `benchmark_comprehensive_adapted.py`
- Earlier 300-pair 3D tuning check: `v3_pointcloud_paired_results.json`
