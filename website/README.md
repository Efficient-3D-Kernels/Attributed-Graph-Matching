# Fast-WCS Interactive Web Platform & Live Demonstration Suite

Welcome to the **Fast-WCS Interactive Web Platform**, an interactive research platform and real-time demonstration engine for **Fast-WCS: Fast Weighted Common Subgraph Matching via Level-3 BLAS Frank-Wolfe Optimization and Closed-Form SVD Superposition**.

> **Detailed Guide Available:** For a complete, file-by-file breakdown of dataset storage locations, benchmark reproduction scripts, and UI customization guides, refer to [`WEBSITE_DATA_AND_SCRIPTS_GUIDE.md`](WEBSITE_DATA_AND_SCRIPTS_GUIDE.md).

---

## 1. Quick Start: Launching the Platform

### Prerequisites
Install the required dependencies:
```bash
pip install flask numpy scipy networkx matplotlib
```
*(Optional for 3D molecular conformers: `pip install rdkit-pypi`)*

### Launch the Server
From this directory, run:
```bash
python app.py
```
Open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)** (or [http://127.0.0.1:8000](http://127.0.0.1:8000))

---

## 2. Web Application Structure & Pages

The web platform is organized into 7 distinct interactive interfaces:

| Page | URL Path | Description & Capabilities |
| :--- | :--- | :--- |
| **Home** | [`index.html`](index.html) (`/`) | Project overview, core research contributions, architectural schematics, and interactive domain navigation. |
| **Live 3D & Benchmark** | [`mapping.html`](mapping.html) (`/mapping`) | **Flagship Live Demo:** Interactive WebGL 3D molecular conformer viewer (Three.js + OrbitControls), real-time Frank-Wolfe execution console, 7-algorithm benchmark shootout table, and 4 dynamic parameter sweep canvases (Chart.js). |
| **Cross-Domain Sandbox** | [`demo.html`](demo.html) (`/demo`) | Interactive matching sandbox for Metal-Organic Frameworks (MOFs), Stanford CollegeMsg social networks, and Drosophila brain connectomes with live objective descent curves. |
| **Documentation** | [`docs.html`](docs.html) (`/docs`) | Complete mathematical formulation, formal proofs for Theorems 1–4, Lemmas 1–2, and Propositions 1–4 rendered in KaTeX, plus API documentation. |
| **Methodology** | [`methodology.html`](methodology.html) (`/methodology`) | Step-by-step visual breakdown of the GNCCP convex-concave continuation, Cardano quartic root extraction, and Level-3 BLAS 2-opt refinement. |
| **Progress** | [`progress.html`](progress.html) (`/progress`) | Implementation status, milestone verification checklist, and high-level empirical summary metrics across all 4 domains. |
| **Team** | [`team.html`](team.html) (`/team`) | Project contributors, author affiliations, and academic acknowledgments. |

---

## 3. Where is Data Stored?

| Problem Domain | Storage Location & File Format | Key Files |
| :--- | :--- | :--- |
| **Synthetic Relational Graphs** | Fixed deterministic seeds generating planted subgraph instances with Gaussian coordinate perturbation. | Generator: `benchmarks/scripts/live_common.py`<br>Web Data: [`synthetic_benchmark_results.json`](synthetic_benchmark_results.json)<br>CSV: `benchmarks/live_runs/csv/synthetic_results.csv` |
| **Planar Image Keypoints** | High-resolution image pairs across 8 sequences of the Oxford Affine benchmark with ground-truth homographies and SIFT descriptors. | Raw Images: `benchmarks/data/oxford_affine/`<br>Web Data: [`image_benchmark_results.json`](image_benchmark_results.json)<br>CSV: `benchmarks/live_runs/csv/image_results.csv` |
| **Scale-Free Social Networks** | Directed, timestamped temporal interactions (1,899 nodes, 59,835 edges) from the Stanford SNAP CollegeMsg dataset. | Raw Data: `socialnetwork/data/CollegeMsg.txt`<br>Results Tables: `socialnetwork/results/tables/*.csv`<br>Plots: `socialnetwork/results/plots/*.png` |
| **3D Molecular Point Clouds** | Cartesian 3D coordinates generated via RDKit ETKDGv3 across 20 drug scaffolds, and raw drug SDF files. | Raw SDFs: [`Data/*.sdf`](../Data/)<br>CSV: `benchmarks/live_runs/csv/superposition_3d_results.csv` |

---

## 4. What Scripts to Run?

### Live Benchmark Reproduction
To run the full suite of live benchmarks that generate the numbers and curves for the website:

```bash
# 1. Run full synthetic sweeps A-D (Noise, Size, Outliers, Density)
python benchmarks/scripts/run_synthetic_replication_live.py

# 2. Run real image keypoints benchmark on Oxford Affine
python benchmarks/scripts/run_image_keypoints_live.py

# 3. Run Stanford CollegeMsg dynamic social network benchmark
python benchmarks/scripts/run_social_network_live.py

# 4. Run 3D molecular superposition benchmark under SO(3) rotations
python benchmarks/scripts/run_3d_superposition_live.py

# 5. Run asymptotic scalability grid sweep (N = 20 to 150)
python benchmarks/scripts/run_large_scale_grid_live.py

# 6. Run component ablation study (S0 to S5)
python benchmarks/scripts/run_multi_scale_ablation_live.py
```

### Standalone Figure Generation
To generate the publication-grade PNG and vector PDF figures:
```bash
python benchmarks/scripts/make_standalone_figures_live.py
python benchmarks/scripts/make_topologies_figures_live.py
python socialnetwork/plot_social_network_benchmark.py
```

### Updating Web JSON Feeds
To refresh the static JSON feeds consumed by Chart.js on the website:
```bash
python benchmarks/scripts/make_readme_results_live.py
```

---

## 5. Live Demonstration Guide for Reviewers & Faculty

When presenting the live demonstration, follow this recommended walkthrough:

1. **Open the 3D Molecular Dashboard:** Go to `http://localhost:8000/mapping.html`.
2. **Execute Live Optimization:** Click the maroon **"Run Live GNCCP Optimization"** button.
   * Point out the real-time execution log: Lemma 1 Rank-1 factorization, Theorem 1 analytical gradient, and the microsecond-level Hungarian assignment times.
   * Interact with the 3D canvas: Rotate, pan, and zoom the reference (green) and target (cyan) molecules, highlighting the amber correspondence vectors and the reflection-safe Kabsch SVD alignment.
3. **Execute the Multi-Model Shootout:** Click **"Run Multi-Model Benchmark Suite"**.
   * Compare Fast-WCS against all 6 baselines (RLX1, RLX1F, RLX2, PGM, GA, SM) in real time.
   * Show that Fast-WCS achieves the highest matching accuracy while running orders of magnitude faster.
4. **Demonstrate Robustness Sweeps:** Scroll to the bottom of `mapping.html` to showcase the 4 interactive Chart.js canvases plotting accuracy and runtime against noise, outliers, problem size, and edge density.
5. **Demonstrate Cross-Domain Adaptations:** Navigate to `demo.html` to showcase matching on MOF nanocrystals, dynamic social networks, and fly brain connectomes.
6. **Review Formal Mathematical Foundations:** Navigate to `docs.html` to review the formal mathematical proofs corresponding directly to the implementation.
