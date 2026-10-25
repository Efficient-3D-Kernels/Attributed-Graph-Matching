# Synthetic graph-matching benchmark: protocol and current results

## What was run

The benchmark runner follows the four synthetic WCS and four PIW sweeps described in Yang et al. (2018): noise, problem size, outlier count, and edge density. Each setting uses 30 paired accuracy instances and 50 runtime instances. Every result is checked to be an exact-‑`L` partial permutation. In addition to ground-truth accuracy and wall time, the runner evaluates the common discrete WCS objective

\[
F(X)=\|((X\mathbf 1)(X\mathbf 1)^T)\circ A_G-XA_HX^T\|_F^2,\qquad \alpha=1.
\]

This follows the reproducibility direction of the ECCV 2022 study: methods receive consistent problem data and results include both runtime and objective value. The ECCV suite itself is not run here; it targets vision datasets and a containerized set of methods, so the results below are a synthetic WCS/PIW comparison inspired by its reporting practices, not a GMBench reproduction. The benchmark authors emphasize common input formats, reproducible solver versions, and objective/runtime reporting, and caution that small popular instances can be too easy to evaluate optimization quality ([ECCV 2022 study and companion benchmark](https://vislearn.github.io/gmbench/), [code](https://github.com/vislearn/gmbench)).

## Protocol fidelity and limits

The base paper specifies 2D uniform point sets, a random planted `L`-partial permutation, coordinate noise on planted pairs, Euclidean edge weights, four WCS and four PIW sweeps, 30 accuracy runs, and 50 timing runs. It does not fully state how the two sparse topologies are coupled, the exact edge-sampling sequence, PRNG, or all baseline solver settings. The reconstruction therefore shares a latent random topology over planted pairs, samples outlier-incident edges independently, and independently adds/removes `round(0.5 * sigma * edge_count)` edges per graph. Noise-added edges now use the Euclidean endpoint distance too; the earlier arbitrary unit weight is removed. Topology coupling remains an explicit assumption to verify against author code or supplementary material before calling this a faithful reproduction.

The original paper's SM/GA/PGM affinity is said to follow Egozi et al. (2013). SM, GA, PGM, EPF, RLX1/RLX2, and RLX1F/RLX2F are not all available here as original implementations. The `Local 2018-style RLX1F` row is a local reconstruction, not the original MATLAB solver. GRAMPA is a published 2020 method adapted to rectangular scores and exact-`L` projection. PyGMtools SM/RRWM/IPFP are generic open-source solvers, also adapted to exact-`L` output. The resulting rows are useful diagnostics, not substitutes for the original base-paper implementations.

## Main result: v3 versus in-repository methods

The following macro averages weight each setting equally within its suite. Accuracy is mean planted-match accuracy across settings; runtime is the median of setting-level median times. Objective is the mean of per-setting mean WCS objective; lower is better within a suite.

| Suite | Method | Accuracy | WCS objective | Median runtime |
|---|---|---:|---:|---:|
| WCS | Fast-WCS v3 | 12.4% | 36.1 | 6.6 ms |
| WCS | Fast-WCS v2 | 10.2% | 41.3 | 4.9 ms |
| WCS | Local 2018-style RLX1F | 17.7% | 48.7 | 28.3 ms |
| WCS | GRAMPA (2020) | 8.5% | 99.9 | 0.4 ms |
| PIW | Fast-WCS v3 | 62.6% | 96.9 | 7.1 ms |
| PIW | Fast-WCS v2 | 55.9% | 112.2 | 6.4 ms |
| PIW | Local 2018-style RLX1F | 42.6% | 140.6 | 44.7 ms |
| PIW | GRAMPA (2020) | 18.9% | 321.7 | 0.7 ms |

On this reconstructed generator, v3 consistently improves accuracy and WCS objective over v2. It is about 1.34x slower than v2 on WCS and 1.11x slower on PIW. The local RLX1F reconstruction has higher planted accuracy than v3 on WCS (by about 5.3 percentage points), despite worse mean WCS objective and around 4.3x higher runtime. On PIW, v3 is more accurate than the local reconstruction by about 20.1 points and is around 6.3x faster. GRAMPA is fastest but has lower accuracy under this rectangular exact-`L` adapter. These results do not support a claim that v3 is the most accurate method across both WCS and PIW.

## ECCV-style classic solver comparison on the WCS noise sweep

SM, RRWM, and IPFP were run over all 11 WCS noise levels with 30 accuracy and 50 timing runs per level. For these methods, PyGMtools consumes a Lawler affinity matrix; it does not define a separate task objective for each solver. The matrix was derived from the WCS residual: for two candidate matches `(i,j)` and `(k,l)`, the pair cost is `(A_G[i,k] - A_H[j,l])^2`. Summing those costs over a binary exact-`L` assignment is exactly the WCS objective above. Since these generic solvers do not natively enforce the known partial cardinality, their score matrix is projected to an exact-`L` assignment. This is an adaptation and can affect the solver's relaxed search; it is not the original base-paper affinity pipeline.

| Method | Mean accuracy over noise settings | Median runtime per instance |
|---|---:|---:|
| Fast-WCS v3 | 11.1% | 6.7 ms |
| PyGMtools SM | 2.4% | 17.2 ms |
| PyGMtools RRWM | 2.4% | 164.6 ms |
| PyGMtools IPFP | 2.6% | 26.1 ms |

The PyGMtools accuracy is close to the random-overlap level for this setting, so these adapter results are diagnostic rather than decisive. IPFP emitted numerical warnings in some cases but returned valid exact-`L` assignments in all recorded runs. For a paper-quality comparison, replace these adapters with original implementations and the affinity construction prescribed by the base paper; use an exact cardinality formulation for partial matching or report the projection as a distinct postprocessing step.

## Recommended interpretation

This pass establishes a reproducible local comparison and identifies the current tradeoff: v3 improves the fast classic v2 and is strong on PIW, but the local RLX1F-style reconstruction remains more accurate on WCS. It does not yet establish superiority over the full 2018 model suite. Before submission, verify the generator/topology assumption, source and run original baselines with their stated affinity and settings, include a native exact-`L` classical formulation, and test larger instances. The ECCV study's open-source/version-pinned approach and objective/runtime reporting are useful standards, but its vision datasets and methods must not be conflated with the WCS/PIW results here.

## Artifacts

- Full 4-sweep WCS/PIW results: `paper_synthetic_eccv_protocol_results.json` and `.csv`
- 11-level WCS noise comparison including SM/RRWM/IPFP: `paper_synthetic_eccv_noise_classical_results.json` and `.csv`
- Runner: `benchmark_paper_synthetic_wcs.py`
