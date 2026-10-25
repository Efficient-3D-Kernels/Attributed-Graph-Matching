# Fast-WCS: Website Data, Scripts, and Live Demonstration Guide

This comprehensive guide serves as the definitive reference for operating, demonstrating, and extending the **Fast-WCS Interactive Web Platform**. It details:
1. **Where all benchmark data and datasets are stored** across every problem domain.
2. **What scripts to run** to reproduce benchmarks, regenerate graphs, and supply live metrics to the web UI.
3. **What each file is and why it is used**, mapping specific files to experimental regimes.
4. **How the website architecture operates** as both an exhaustive research documentation hub and a real-time, interactive demonstration platform.

---

## Table of Contents
- [1. Executive Architectural Overview](#1-executive-architectural-overview)
- [2. Master Dataset Inventory (Where Data is Stored)](#2-master-dataset-inventory-where-data-is-stored)
  - [Domain 1: Synthetic Relational Graphs (ER, Regular, Scale-Free)](#domain-1-synthetic-relational-graphs-er-regular-scale-free)
  - [Domain 2: Planar Image Keypoints (Oxford Affine)](#domain-2-planar-image-keypoints-oxford-affine)
  - [Domain 3: Scale-Free Social Networks (Stanford SNAP CollegeMsg)](#domain-3-scale-free-social-networks-stanford-snap-collegemsg)
  - [Domain 4: 3D Molecular Conformer Superposition (RDKit & Drug SDFs)](#domain-4-3d-molecular-conformer-superposition-rdkit--drug-sdfs)
  - [Cross-Domain: QMOF Nanocrystals & Drosophila Connectomes](#cross-domain-qmof-nanocrystals--drosophila-connectomes)
- [3. Master Script Execution Matrix (What Scripts to Run & Why)](#3-master-script-execution-matrix-what-scripts-to-run--why)
  - [Live Benchmark & Metric Generation Scripts](#live-benchmark--metric-generation-scripts)
  - [Standalone Domain-Specific Solvers](#standalone-domain-specific-solvers)
  - [Visual Figure & Graph Generation Scripts](#visual-figure--graph-generation-scripts)
- [4. File-by-File Matrix (Role, Inputs, Outputs, and Use Cases)](#4-file-by-file-matrix-role-inputs-outputs-and-use-cases)
- [5. How the Website Operates (Frontend & Backend Integration)](#5-how-the-website-operates-frontend--backend-integration)
  - [Interactive Web Pages](#interactive-web-pages)
  - [Flask Backend API Endpoints (`app.py`)](#flask-backend-api-endpoints-apppy)
  - [Data Flow: How Graphs & Metrics Render on Web Pages](#data-flow-how-graphs--metrics-render-on-web-pages)
- [6. Step-by-Step Guide: Showing New Domain Graphs on the Website](#6-step-by-step-guide-showing-new-domain-graphs-on-the-website)
  - [Updating Static Benchmark JSON Feeds](#updating-static-benchmark-json-feeds)
  - [Connecting a New Interactive Live Solver Tab](#connecting-a-new-interactive-live-solver-tab)
  - [Adding New Chart.js Visualizations](#adding-new-chartjs-visualizations)
- [7. Quick-Start Execution & Live Demo Playbook](#7-quick-start-execution--live-demo-playbook)

---

## 1. Executive Architectural Overview

The Fast-WCS Web Platform serves a dual purpose:
1. **Interactive Live Demonstration:** Enables real-time execution of our Level-3 BLAS Frank-Wolfe optimization engine, allowing users to interactively manipulate problem parameters ($M, N, L$, noise $\sigma$, edge density $\rho$), rotate 3D molecular conformers in WebGL via Three.js, view live objective convergence traces, and execute live multi-model benchmark shootouts against 6 baseline algorithms.
2. **Comprehensive Research Documentation:** Provides a complete, peer-review-grade documentation hub detailing the formal mathematical foundations (Theorems 1–4, Lemmas 1–2, Propositions 1–4), algorithmic derivations, and empirical benchmark results across 4 diverse domains.

```
+---------------------------------------------------------------------------------------+
|                                  USER BROWSER CLIENT                                  |
|   +--------------------+  +--------------------+  +--------------------------------+  |
|   |    demo.html       |  |    mapping.html    |  |  docs.html / methodology.html  |  |
|   | Interactive Solver |  | 3D WebGL Viewer    |  | Formal Theory, Proofs, &       |  |
|   | Parameter Controls |  | 5 Benchmark Charts |  | API Documentation              |  |
|   +---------+----------+  +---------+----------+  +--------------------------------+  |
+-------------|-----------------------|-------------------------------------------------+
              | HTTP POST             | HTTP GET / POST
              v                       v
+---------------------------------------------------------------------------------------+
|                                FLASK BACKEND (app.py)                                 |
|  * /api/run_live_gnccp          * /api/run_multimodel_benchmark                       |
|  * /api/run_sweep_benchmarks    * /api/run_scaling_benchmark                          |
|  * /api/run_match               * /api/benchmark_data                                |
+-------------------------------------+-------------------------------------------------+
                                      | Direct Calls / JSON Ingestion
        +-----------------------------+-----------------------------+
        |                                                           |
        v                                                           v
+--------------------------------+          +-----------------------------------------+
|    MODULAR DOMAIN SOLVERS      |          |       PRECOMPUTED BENCHMARK REPO        |
|  * fastwcs_synthetic.py        |          |  * synthetic_benchmark_results.json    |
|  * fastwcs_image_sift.py       |          |  * image_benchmark_results.json        |
|  * fastwcs_social_network.py   |          |  * benchmarks/live_runs/csv/*.csv       |
|  * fastwcs_molecular_3d.py     |          |  * benchmarks/figures/standalone/*.png  |
|  * fastwcs_solver.py           |          |  * socialnetwork/results/tables/*.csv   |
+--------------------------------+          +-----------------------------------------+
```

---

## 2. Master Dataset Inventory (Where Data is Stored)

### Domain 1: Synthetic Relational Graphs (ER, Regular, Scale-Free)
* **Storage Location:** Generated programmatically with deterministic random seeds to ensure 100% reproducible benchmarking.
  * Live Generator: [`benchmarks/scripts/live_common.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/scripts/live_common.py) and [`benchmarks/scripts/run_synthetic_replication_live.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/scripts/run_synthetic_replication_live.py).
  * Web JSON Feeds: [`website/synthetic_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/synthetic_benchmark_results.json).
  * Aggregated CSV Results: [`benchmarks/live_runs/csv/synthetic_results.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/synthetic_results.csv) and [`benchmarks/live_runs/csv/topologies_results.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/topologies_results.csv).
  * Per-Trial Raw Logs: [`benchmarks/live_runs/csv/synthetic_trials.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/synthetic_trials.csv).
* **Format:** Planted subgraph model with 2D Gaussian spatial coordinates, Euclidean pairwise affinities, and perturbed adjacency matrices $\mathbf{A}_G \in \mathbb{R}^{M \times M}$, $\mathbf{A}_H \in \mathbb{R}^{N \times N}$.
* **Data Scale:** 1,400 instances across 28 parameter settings (Sweeps A–D: noise $\sigma \in [0, 0.10]$, problem size $M \in [20, 60]$, outliers $M-L \in [0, 10]$, edge density $\rho \in [0.2, 1.0]$).

### Domain 2: Planar Image Keypoints (Oxford Affine)
* **Storage Location:**
  * Raw Image Sequences: [`benchmarks/data/oxford_affine/`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/data/oxford_affine/) (8 sequences: `bark`, `bikes`, `boat`, `graf`, `leuven`, `trees`, `ubc`, `wall`).
  * Downloader Script: [`benchmarks/scripts/fetch_oxford_affine.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/scripts/fetch_oxford_affine.py).
  * Web JSON Feed: [`website/image_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/image_benchmark_results.json).
  * Aggregated CSV Results: [`benchmarks/live_runs/csv/image_results.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/image_results.csv).
  * Per-Trial Raw Logs: [`benchmarks/live_runs/csv/image_trials.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/image_trials.csv).
* **Format:** High-resolution image pairs (.ppm / .png) with ground-truth homography matrices $H_{1\to i}$. SIFT keypoints are extracted using OpenCV, planar topologies are constructed via Delaunay triangulation, and pairwise Euclidean descriptor distances form the cost matrix $\mathbf{C} \in \mathbb{R}^{M \times N}$.
* **Data Scale:** 16 test image pairs (image 1 matched to images 2 and 3 across 8 sequences) + 8 held-out calibration pairs (image 1 to image 4).

### Domain 3: Scale-Free Social Networks (Stanford SNAP CollegeMsg)
* **Storage Location:**
  * Raw Dataset: [`socialnetwork/data/CollegeMsg.txt`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/data/CollegeMsg.txt) (compressed backup in [`socialnetwork/data/CollegeMsg.txt.gz`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/data/CollegeMsg.txt.gz)).
  * Preprocessed Graphs & Loaders: [`socialnetwork/src/data_loader.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/src/data_loader.py) and [`socialnetwork/src/graph_builder.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/src/graph_builder.py).
  * Benchmark Tables: [`socialnetwork/results/tables/`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/results/tables/) (`social_network_perturbation_results.csv`, `social_network_temporal_results.csv`, `social_network_global_kernels.csv`).
  * Benchmark Plots: [`socialnetwork/results/plots/`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/socialnetwork/results/plots/) (`social_network_accuracy_frontier.png`, `social_network_runtime_pareto.png`).
* **Format:** Directed, timestamped temporal messages: `[source_user, target_user, unix_timestamp]`. Contains 1,899 nodes and 59,835 message interactions.
* **Data Scale:** Subgraphs extracted for active users: $L=30, 40, 100, 400$ nodes across early and later temporal communication windows.

### Domain 4: 3D Molecular Conformer Superposition (RDKit & Drug SDFs)
* **Storage Location:**
  * Live Conformer Generator: [`benchmarks/scripts/run_3d_superposition_live.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/scripts/run_3d_superposition_live.py) (uses RDKit ETKDGv3 3D conformer generator with hydrogens).
  * Raw Target SDF Files: [`c:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\Data\`](file:///c:/Users/abdus/Downloads/FYP/Efficient-3D-Kernels-for-molecular-property-prediction/Data/) (includes `1798_actives_new.sdf`, `1843_actives_new.sdf`, `435034_actives_new.sdf`, `Aromatase_actives_new.sdf`, `LBD_actives_new.sdf`, `MMP_actives_new.sdf`, `p53_actives_new.sdf`).
  * Aggregated CSV Results: [`benchmarks/live_runs/csv/superposition_3d_results.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/superposition_3d_results.csv).
  * Per-Trial Raw Logs: [`benchmarks/live_runs/csv/superposition_3d_trials.csv`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/live_runs/csv/superposition_3d_trials.csv).
* **Format:** Cartesian 3D coordinates $P \in \mathbb{R}^{M \times 3}, Q \in \mathbb{R}^{N \times 3}$, atom element types, and covalent bond connectivity matrices.
* **Data Scale:** 20 diverse FDA drug-like scaffolds, 300 test trials split across 3 evaluation scenarios (S1 near-rigid, S2 deformation, S3 outlier-heavy).

### Cross-Domain: QMOF Nanocrystals & Drosophila Connectomes
* **Storage Location:**
  * QMOF Database: [`mof_project/data/`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/mof_project/data/) (`UiO-66.cif`, `MOF-5.cif`, `HKUST-1.cif`) and [`mof_project/qmof_graph_cache.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/mof_project/qmof_graph_cache.json).
  * Drosophila Connectomes: Mushroom body ROI connectomes from Hemibrain vs FlyWire/FAFB (4.4K vs 9.9K neurons).

---

## 3. Master Script Execution Matrix (What Scripts to Run & Why)

### Live Benchmark & Metric Generation Scripts

All benchmark scripts reside in [`benchmarks/scripts/`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/benchmarks/scripts/) and follow strict empirical protocols (fixed seeds, held-out parameter calibration, exact-$L$ Hungarian projection, paired Wilcoxon signed-rank significance tests).

| Command to Run | What it Tests / Case | Input Data Consumed | Outputs Generated | Typical Runtime |
| :--- | :--- | :--- | :--- | :---: |
| `python benchmarks/scripts/run_synthetic_replication_live.py` | **Synthetic WCS Protocol (Sweeps A–D)**: Evaluates noise robustness ($\sigma \in [0, 0.1]$), problem size ($M \in [20, 60]$), outliers ($M-L \in [0, 10]$), and density ($\rho \in [0.2, 1.0]$) against 13 baseline solvers. | Deterministic seed generator in `live_common.py` | `benchmarks/live_runs/csv/synthetic_results.csv`, `synthetic_trials.csv` | ~4.5 min |
| `python benchmarks/scripts/run_topologies_benchmark_live.py` | **Non-Euclidean Graph Topologies**: Tests performance on Erdös-Rényi, Regular, and Barabási-Albert Scale-Free topologies with flat barycenters ($C=0$). | Deterministic synthetic topology generator | `benchmarks/live_runs/csv/topologies_results.csv`, `topologies_trials.csv` | ~2.5 min |
| `python benchmarks/scripts/run_image_keypoints_live.py` | **Real Visual Keypoints**: Evaluates planar keypoint matching on Oxford Affine image sequences under viewpoint deformation and outliers. | `benchmarks/data/oxford_affine/` image pairs | `benchmarks/live_runs/csv/image_results.csv`, `image_trials.csv` | ~1.8 min |
| `python benchmarks/scripts/run_social_network_live.py` | **Scale-Free Dynamic Social Networks**: Matches temporal interaction subgraphs of Stanford CollegeMsg under 0% to 30% structural noise. | `socialnetwork/data/CollegeMsg.txt` | `benchmarks/csv_data/tab_social_network.csv`, `socialnetwork/results/tables/*.csv` | ~3.0 min |
| `python benchmarks/scripts/run_3d_superposition_live.py` | **3D Molecular Point Clouds**: Benchmarks 3D superposition under arbitrary $\mathrm{SO}(3)$ rotations against ICP, CPD, RANSAC, FAQ, and Gromov-Wasserstein. | RDKit conformer generator (20 drug scaffolds) | `benchmarks/live_runs/csv/superposition_3d_results.csv`, `superposition_3d_trials.csv` | ~2.2 min |
| `python benchmarks/scripts/run_large_scale_grid_live.py` | **Asymptotic Scalability Grid**: Sweeps $N \in [20, 150]$ and $L/N \in [0.4, 1.0]$ to verify $\Theta(N^3)$ computational scaling vs $O(N^{5.4})$ RLX1. | Matrix dimension grid | `benchmarks/live_runs/csv/large_scale_grid.csv`, `large_scale_grid_trials.csv` | ~3.5 min |
| `python benchmarks/scripts/run_multi_scale_ablation_live.py` | **Component Ablation Study**: Measures the marginal speedup and accuracy contribution of each algorithmic stage (S0 $\to$ S1 $\to$ S2 $\to$ S3 $\to$ S4 $\to$ S5). | Multi-scale synthetic regimes ($N \in \{30, 60, 100\}$) | `benchmarks/live_runs/csv/ablation_results.csv`, `ablation_linesearch_calls.csv` | ~2.0 min |
| `python benchmarks/scripts/audit_engine_v3.py` | **Mathematical Engine Audit**: Validates Theorem 1 analytical gradient against numerical finite differences and verifies machine-precision duality gap computation. | Random symmetric matrices | `benchmarks/live_runs/csv/gradient_microbenchmark.csv` | ~15 sec |

---

### Standalone Domain-Specific Solvers

For developers, reviewers, or live command-line demos, Fast-WCS provides standalone modular scripts at the project root:

| Standalone Script | Domain / Case | Execution Command | Key Feature Highlight |
| :--- | :--- | :--- | :--- |
| [`fastwcs_synthetic.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/fastwcs_synthetic.py) | Synthetic Relational Graphs | `python fastwcs_synthetic.py` | Multi-start homotopy continuation ($K=10$ restarts) to shatter flat-barycenter gradient deadlocks. |
| [`fastwcs_image_sift.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/fastwcs_image_sift.py) | Planar Image Keypoints | `python fastwcs_image_sift.py` | Feature-coupled GNCCP fusing Delaunay topology with SIFT descriptors + Level-3 BLAS 2-Opt. |
| [`fastwcs_social_network.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/fastwcs_social_network.py) | Dynamic Social Networks | `python fastwcs_social_network.py` | Weisfeiler-Lehman tree-history homotopy ($C_{\text{WL}}$ prior + tree-guided convex warm-start). |
| [`fastwcs_molecular_3d.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/fastwcs_molecular_3d.py) | 3D Molecular Point Clouds | `python fastwcs_molecular_3d.py` | Hierarchical 2D-to-3D pipeline with rotation-invariant distance geometry and reflection-safe Kabsch SVD. |
| [`fastwcs_solver.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/fastwcs_solver.py) | Unified Python Factory | `python -c "import fastwcs_solver; print(fastwcs_solver.AVAILABLE_DOMAINS)"` | Factory dispatcher exposing unified class interface `get_solver(domain, **kwargs)`. |

---

### Visual Figure & Graph Generation Scripts

To regenerate the publication-grade figures shown on the website and in the research paper:

| Script to Run | Figures Produced | Output Location |
| :--- | :--- | :--- |
| `python benchmarks/scripts/make_standalone_figures_live.py` | Generates all 13 standalone benchmark curves: noise, outliers, size, density, scaling runtime, speedup factors, RMSD cumulative distributions, Pareto frontiers, and Cardano line search latency. | `benchmarks/figures/standalone/*.png` and `*.pdf` |
| `python benchmarks/scripts/make_topologies_figures_live.py` | Generates topology comparison figures across ER, Regular, and Scale-Free graphs. | `benchmarks/figures/standalone/fig_topologies_*.png` |
| `python socialnetwork/plot_social_network_benchmark.py` | Generates CollegeMsg accuracy frontiers and runtime Pareto curves. | `socialnetwork/results/plots/*.png` and `*.pdf` |
| `python benchmarks/scripts/make_readme_results_live.py` | Updates markdown documentation tables directly from live benchmark CSVs. | Project `README.md` and `paper_ready_wcs_results.json` |

---

## 4. File-by-File Matrix (Role, Inputs, Outputs, and Use Cases)

| File Name | Primary Role | What Inputs it Consumes | What Outputs it Produces | When to Use This File |
| :--- | :--- | :--- | :--- | :--- |
| [`website/app.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/app.py) | Flask Web Server & API Dispatcher | HTTP POST payloads with solver parameters ($M, N, L, \sigma, \rho$, domain) | JSON responses with 3D coordinates, match pairs, convergence history, and benchmark arrays | Always running to host the interactive website and live demonstration. |
| [`website/mapping.html`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/mapping.html) | Interactive 3D WebGL & Benchmark Dashboard | API responses from `/api/run_live_gnccp` and precomputed JSON files | Interactive Three.js 3D atom view + 5 Chart.js dynamic benchmark graphs | The primary live demonstration screen for presentations and faculty reviews. |
| [`website/demo.html`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/demo.html) | Multi-Domain Matching Sandbox | API responses from `/api/run_match` | Node-to-node correspondence cards and real-time homotopy convergence curves | Used to demonstrate matching across MOF crystals, Social networks, and Brain connectomes. |
| [`website/docs.html`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/docs.html) | Formal Theory & API Reference | Static documentation content | Rendered KaTeX mathematical proofs, complexity tables, and solver API parameters | Used for examining algorithmic proofs, Lemma/Theorem derivations, and complexity bounds. |
| [`website/methodology.html`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/methodology.html) | Visual Algorithmic Deep-Dive | Static methodology content | Visual schematics of GNCCP relaxation, Cardano quartic root extraction, and 2-opt refinement | Used to explain *why* Fast-WCS achieves $\Theta(N^3)$ complexity without Lawler tensors. |
| [`website/progress.html`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/progress.html) | Empirical Progress & Milestones | Precomputed benchmark statistics | High-level summary cards, speedup multipliers, and verification checklists | Quick summary dashboard of project deliverables and empirical verification. |
| [`website/synthetic_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/synthetic_benchmark_results.json) | Static Precomputed Synthetic Benchmark Data | Generated from `synthetic_results.csv` | Arrays of noise, outliers, problem size, and edge density curves for 7 solvers | Loaded instantly by Chart.js when offline or when avoiding live recalculation. |
| [`website/image_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/image_benchmark_results.json) | Static Precomputed Image Benchmark Data | Generated from `image_results.csv` | Arrays of visual keypoint accuracy and runtime scaling for 7 solvers | Loaded by Chart.js for real-world visual keypoint evaluation. |

---

## 5. How the Website Operates (Frontend & Backend Integration)

### Interactive Web Pages

The website frontend is constructed with modern vanilla JavaScript, Three.js, Chart.js, and a responsive academic light theme:

1. **`index.html` (Homepage & Executive Overview):**
   * Presents the core research question: *How can we match high-dimensional relational structures in exact $\Theta(N^3)$ time without Lawler tensor explosion?*
   * Features architectural diagrams, interactive domain navigation cards, and direct links to live demos.
2. **`demo.html` (Live Multi-Domain Solver):**
   * Allows selecting between **MOF Nanocrystals**, **Dynamic Social Networks**, and **Drosophila Brain Connectomes**.
   * Lets users select candidate subgraphs, triggers `/api/run_match`, and graphs the objective function $F(X)$ descent over continuation parameter $\zeta \in [1, -1]$.
3. **`mapping.html` (Flagship 3D Molecule & Multi-Model Benchmark Suite):**
   * **3D Molecule Visualizer:** Renders reference molecule (green) and target molecule (cyan) in real-time WebGL. When matched, atoms are connected by amber correspondence vectors, and the target conformer is aligned via closed-form Kabsch SVD.
   * **Live Execution Console:** Displays microsecond-level timing timestamps for Rank-1 factorization, analytical gradient computation, Hungarian linear assignment, and duality gap certificate checking.
   * **Multi-Model Benchmark Shootout:** Executes Fast-WCS against RLX1, RLX1F, RLX2, PGM, GA, and SM simultaneously, populating a comparative performance table with speedup multipliers.
   * **Dynamic Parameter Sweeps:** 4 interactive Chart.js canvases plotting accuracy and runtime vs. Noise ($\sigma$), Outliers ($M-L$), Problem Size ($M$), and Graph Density ($\rho$).

---

### Flask Backend API Endpoints (`app.py`)

The Flask backend in [`website/app.py`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/app.py) exposes high-performance API endpoints running synchronous Level-3 BLAS linear algebra:

#### 1. Real-Time GNCCP Engine: `POST /api/run_live_gnccp`
* **Request Payload:**
  ```json
  {
    "M": 14,
    "N": 18,
    "L": 10,
    "seed": 42
  }
  ```
* **Processing:**
  1. Generates 3D molecular conformations with an $L$-atom common core and private decoy atoms.
  2. Applies random 3D spatial rotation $\mathbf{R} \in \mathrm{SO}(3)$ and translation $\mathbf{t}$.
  3. Builds 2D topological adjacencies $\mathbf{A}_G, \mathbf{A}_H$ and applies Frobenius normalization.
  4. Runs GNCCP Frank-Wolfe homotopy with Rank-1 factorization ($\mathbf{U} = \mathbf{r}\mathbf{r}^T$) and analytical gradient ($\nabla F$).
  5. Extracts exact bijective atom pairings via Hungarian assignment on $-\mathbf{X}$.
  6. Computes reflection-safe Kabsch 3D SVD alignment ($\det \mathbf{R} = +1$) and evaluates all-atom RMSD.
* **Response Payload:** Contains 3D coordinate arrays, aligned coordinates, matching index pairs, RMSD, wall-clock runtimes, duality gap logs, and convergence traces.

#### 2. Multi-Model Benchmark Shootout: `POST /api/run_multimodel_benchmark`
* **Request Payload:** `{ "M": 14, "N": 18, "L": 10, "sigma": 0.03, "density": 0.5, "seed": 42 }`
* **Processing:** Solves the exact same problem instance using all 7 graph matching algorithms:
  1. **Fast-WCS (Ours)** — $\Theta(N^3)$ exact closed-form gradient.
  2. **RLX1 (Yang et al. 2018)** — $O(N^{5.4})$ 4th-order polynomial.
  3. **RLX1F (Fast Truncation)** — $O(N^{3.5})$ heuristic drop.
  4. **RLX2 (Yang et al. 2018)** — $O(N^{5.4})$ 6th-order polynomial.
  5. **PGM (Projected Gradient Matching)** — $O(N^4)$ affinity projection.
  6. **GA (Graduated Assignment)** — $O(N^4)$ softmax annealing.
  7. **SM (Spectral Matching)** — $O(N^4)$ leading eigenvector approximation.
* **Response Payload:** Array of `{ name, complexity, time_ms, accuracy, f_val, rmsd, speedup }`.

#### 3. Live 4-Way Parameter Sweeps: `POST /api/run_sweep_benchmarks`
* Executes live parameter sweeps across:
  * **Noise Robustness:** $\sigma \in \{0.00, 0.02, 0.04, 0.06, 0.08, 0.10\}$
  * **Outlier Tolerance:** Outlier count $M-L \in \{0, 2, 4, 6, 8, 10\}$
  * **Problem Size Scaling:** $M = N \in \{20, 28, 36, 44, 52, 60\}$
  * **Graph Density:** Edge density $\rho \in \{0.20, 0.40, 0.60, 0.80, 1.00\}$
* Returns structured arrays formatted for direct insertion into Chart.js datasets.

---

### Data Flow: How Graphs & Metrics Render on Web Pages

```
[User clicks "Run Live Benchmark"]
                |
                v
[JavaScript fetch('/api/run_sweep_benchmarks')]
                |
                +--> Flask generates problem instances (seed=42)
                +--> Solves with Fast-WCS + 6 baseline algorithms
                +--> Computes accuracy %, objective F, and runtimes
                +--> Returns JSON payload to browser
                |
                v
[JavaScript receives JSON]
                |
                +--> benchNoiseChart.data.datasets = [...]
                +--> benchNoiseChart.update()
                +--> benchOutlierChart.update()
                +--> benchSizeChart.update()
                +--> benchDensityChart.update()
                |
                v
[Canvases animate with updated curves instantly]
```

---

## 6. Step-by-Step Guide: Showing New Domain Graphs on the Website

### Updating Static Benchmark JSON Feeds
If you have run new benchmark sweeps and want the website to display the updated numbers without re-running solvers live:

1. **Re-export benchmark results to JSON:**
   Run the live export script to update `paper_ready_wcs_results.json` and CSV files:
   ```bash
   python benchmarks/scripts/make_readme_results_live.py
   ```
2. **Update the website JSON files:**
   * For synthetic sweeps: Update [`website/synthetic_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/synthetic_benchmark_results.json) with keys `sigmas`, `acc_noise`, `outliers`, `acc_outliers`, `sizes`, `time_size`, `densities`, `acc_density`.
   * For image sweeps: Update [`website/image_benchmark_results.json`](file:///C:/Users/abdus/Desktop/Fast-WCS-Research-Deliverables/website/image_benchmark_results.json) with keys `sizes`, `acc_size`, `time_size`.
3. **Verify in Browser:**
   Refresh [http://localhost:8000/mapping.html](http://localhost:8000/mapping.html); the charts automatically parse and render the updated JSON values upon load.

---

### Connecting a New Interactive Live Solver Tab
To add a new live domain solver (e.g. Brain Connectomes or Social Networks) to the interactive UI:

1. **Implement the domain route in `website/app.py`:**
   ```python
   @app.route('/api/run_social_live', methods=['POST'])
   def run_social_live():
       data = request.json or {}
       noise = float(data.get('noise', 0.1))
       from fastwcs_social_network import FastWCS_Social
       solver = FastWCS_Social(wl_iterations=3, beta=0.5)
       # Load CollegeMsg graph and run matching
       ...
       return jsonify({
           "accuracy": acc,
           "runtime_ms": t_ms,
           "mapping": matches
       })
   ```

2. **Add an interactive button/canvas in `website/mapping.html` or `demo.html`:**
   ```html
   <button class="btn-action" onclick="runLiveSocialBenchmark()">
     Run Social Network Matching
   </button>
   <canvas id="canvasSocial"></canvas>
   ```

3. **Wire the JavaScript fetch call:**
   ```javascript
   async function runLiveSocialBenchmark() {
     const res = await fetch('/api/run_social_live', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ noise: 0.2 })
     });
     const data = await res.json();
     updateSocialChart(data);
   }
   ```

---

### Adding New Chart.js Visualizations
To add a new graph canvas in HTML:
```html
<div class="chart-card">
  <h3>Runtime Pareto Frontier</h3>
  <canvas id="newChartCanvas" width="400" height="260"></canvas>
</div>
```
Initialize the chart in JavaScript:
```javascript
const ctx = document.getElementById('newChartCanvas').getContext('2d');
const myChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    datasets: [{
      label: 'Fast-WCS (Ours)',
      data: [100.0, 100.0, 100.0, 100.0, 98.5, 96.2, 94.0],
      borderColor: '#8B1E2D',
      backgroundColor: 'rgba(139, 30, 45, 0.1)',
      borderWidth: 2.5,
      tension: 0.3
    }]
  },
  options: {
    responsive: true,
    scales: {
      y: { title: { display: true, text: 'Node Matching Accuracy (%)' }, min: 0, max: 100 },
      x: { title: { display: true, text: 'Structural Perturbation Noise' } }
    }
  }
});
```

---

## 7. Quick-Start Execution & Live Demo Playbook

### Step 1: Install Required Dependencies
Ensure your Python environment has the necessary scientific packages installed:
```bash
pip install flask numpy scipy networkx matplotlib
```
*(Optional for 3D Conformer benchmarks: `pip install rdkit-pypi`)*

### Step 2: Launch the Flask Web Server
Navigate to the website directory and launch the server:
```bash
cd C:\Users\abdus\Desktop\Fast-WCS-Research-Deliverables\website
python app.py
```
*The server will start at `http://127.0.0.1:8000` (or `http://localhost:8000`).*

### Step 3: Recommended Presentation Walkthrough
During project evaluations, demonstrations, or committee reviews, follow this sequence:

1. **Executive Opening (`index.html`):**
   * Introduce the research problem: High-dimensional relational matching under deformations.
   * Highlight our core result: Exact closed-form $\Theta(N^3)$ Frank-Wolfe optimization that outpaces Lawler tensor baselines by up to $30\times$.
2. **Interactive 3D Molecular Demonstration (`mapping.html`):**
   * Click **"Run Live GNCCP Optimization"**.
   * Show the live execution console: Point out Lemma 1 Rank-1 factorization, Theorem 1 analytical gradient, and the microsecond-level Hungarian assignment times.
   * Rotate and zoom the 3D molecular conformers with the mouse: Highlight the amber correspondence lines and the Kabsch SVD alignment.
   * Click **"Run Multi-Model Benchmark Suite"**: Show the live 7-model comparison table where Fast-WCS achieves the highest accuracy with the fastest execution time.
3. **Parameter Sweeps & Robustness (`mapping.html` - Bottom Section):**
   * Demonstrate the 4 interactive benchmark charts:
     * **Noise Curve:** Accuracy remains robust up to $\sigma = 0.08$.
     * **Outlier Tolerance:** High accuracy maintained as outliers increase from 0 to 10.
     * **Problem Size Scaling:** Runtime grows smoothly following the theoretical cubic slope.
     * **Edge Density:** Consistent performance across sparse ($\rho=0.2$) to dense ($\rho=1.0$) graphs.
4. **Cross-Domain Adaptations (`demo.html`):**
   * Demonstrate matching on **MOF Nanocrystals** (screening UiO-66 against candidate frameworks).
   * Demonstrate matching on **Dynamic Social Networks** (CollegeMsg user community alignment).
5. **Formal Proofs & Verification (`docs.html` & `methodology.html`):**
   * Direct faculty to the mathematical proofs of Theorems 1–4, showing that every line of code strictly mirrors the formal mathematical manuscript.
