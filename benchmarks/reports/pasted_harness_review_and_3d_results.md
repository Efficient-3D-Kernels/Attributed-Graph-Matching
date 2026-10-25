# Pasted harness review and paired 3D v3 results

## What the pasted script does

The pasted `benchmark_harness.py` is not yet a runnable benchmark for this repository. It defines metric and solver helper functions, but its `__main__` block only creates a toy active/inactive list and demonstrates ROC-AUC, PR-AUC, BEDROC, and enrichment-factor calculations. It never calls the solver registry or runs graph/molecule instances. It imports `gnccp_v3`, which is not the module in this workspace (`gnccp_advanced_v3.py`).

Useful ideas to retain:

- paired trials and confidence intervals;
- paired significance tests for methods run on identical instances;
- untrimmed Kabsch RMSD over every predicted correspondence;
- clearly stated success thresholds, if their units and scientific meaning are justified.

Changes needed before using it as a paper benchmark:

- Virtual-screening ROC/PR/BEDROC/EF metrics require labeled active/inactive compounds and per-compound ranking scores. They do not evaluate graph correspondence by themselves.
- SciPy FAQ pads unequal graphs to a square assignment, then truncates a full assignment. It does not enforce the WCS exact-`L` objective; with no unary cost matrix `C`, truncation is arbitrary. It returns objective `0.0`, which cannot be compared with the WCS objective.
- The partial-OT adapter uses Euclidean point distances, not the graph objective, and its mass-to-cardinality conversion/rounding needs a separately validated exact-`L` formulation.
- `alpha=0.7` is a point-cloud setting; the base synthetic WCS/PIW experiments use `alpha=1`. Results from these settings should not be mixed.
- The stated EF ceiling formula is exact only when the selected top-`k` can contain at least `k` actives; the function's more general expression accounts for the finite number of actives. Percentile bootstrap is implemented, but BCa is not; Wilcoxon p-values also need a multiple-comparison plan if many configurations are tested.
- `full_atom_kabsch_rmsd` evaluates all supplied pairs, but those are only the predicted matched subset `L`, not necessarily every atom in either full molecule. Its name should say “all predicted pairs.” A `999.0` sentinel for fewer than three pairs is better represented as an unavailable metric plus a validity/failure count.

## New paired holdout benchmark

I used the script's statistical ideas on the existing controlled synthetic 3D point-cloud setup. This is **not molecular data**: coordinates are in arbitrary synthetic units, with 16 source points, 22 target points, 12 planted pairs, Gaussian coordinate noise `0.03`, and distractors. Runtime includes the point-cloud matching call, including graph/feature construction. RMSD is untrimmed over all 12 predicted pairs after Kabsch alignment.

There are 300 paired instances from fresh seeds `60000–60299`. The two candidate configurations were selected after earlier exploratory tuning, so this is a fresh-seed check, but not a preregistered or external benchmark. Paired bootstrap intervals are percentile intervals. Wilcoxon p-values are unadjusted in the JSON.

| v3 configuration | Pair accuracy (95% bootstrap CI) | Median all-pair RMSD | RMSD < 1 synthetic unit | Median time |
|---|---:|---:|---:|---:|
| Default: kernel scale 1.0, 3 FW steps, 2 refinement passes | 27.4% (23.4–31.6) | 0.729 | 86.3% | 8.91 ms |
| Narrow kernel 0.85, 3 FW steps, 2 passes | 30.1% (26.0–34.3) | 0.728 | 81.0% | 9.40 ms |
| Narrow kernel 0.85, 3 FW steps, 4 passes | 30.2% (25.9–34.6) | **0.667** | 88.3% | 10.99 ms |
| Narrow kernel 0.85, 2 FW steps, 2 passes | 30.3% (26.3–34.6) | 0.726 | **89.3%** | **8.39 ms** |

Against default, the **2-step narrow-kernel setting** was 0.88 ms faster on average (paired bootstrap 95% CI: 0.36–1.41 ms faster; Wilcoxon `p=0.002`, unadjusted). Accuracy was 2.9 percentage points higher on average, but its paired 95% CI ranged from 2.1 points worse to 8.0 points better (`p=0.129`). RMSD was not significantly different. This looks like a promising speed configuration with no observed accuracy loss, but the accuracy advantage is not established.

The **four-pass polish** lowered mean paired RMSD by 0.063 synthetic units (95% CI 0.017–0.109 lower; unadjusted `p=0.0079`) and improved the median, but increased runtime by 1.87 ms on average (95% CI 1.25–2.49 ms slower). Its planted accuracy gain was not statistically clear. After Holm correction across the nine candidate/metric comparisons, the runtime improvement for the 2-step setting remains significant, while the RMSD improvement for four-pass polish does not.

## Answer to “can v3 be faster and more accurate?”

The benchmark supports a **small speed improvement candidate**: reduce Frank–Wolfe substeps from 3 to 2 and use RBF scale 0.85. On this 300-pair synthetic holdout it was about 6% faster by median time and did not show a measurable RMSD penalty. Its higher mean pair accuracy is uncertain. More refinement improved alignment RMSD but cost runtime. So there is a speed/accuracy trade-off to investigate; these results do not yet prove a universally faster and more accurate algorithm. The next validation should test the speed setting on larger and real 3D/molecular data, with chemistry-aware atom constraints and an external test set.

## Artifacts

- Paired benchmark runner: `benchmark_v3_pointcloud_paired.py`
- 300-pair output, per-instance samples, confidence intervals, and paired tests: `v3_pointcloud_paired_results.json`
- Original pasted text: `C:\Users\abdus\.codex\attachments\42590d4c-df24-49e1-83b0-f36a597d8cf0\Pasted text.txt`
