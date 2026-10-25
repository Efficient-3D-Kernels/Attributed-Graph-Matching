# An Efficient Framework for Structural Correspondence and Partial Matching in Attributed Graphs

**An optimization-based framework for robust partial graph matching across heterogeneous relational domains.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Paper: 20 Pages](https://img.shields.io/badge/Paper-20%20Pages%20Camera--Ready-red.svg)](paper_materials/research_paper.pdf)
[![Vercel Deployment](https://img.shields.io/badge/Deployment-Vercel%201--Click-black.svg)](https://vercel.com/)

---

## 📄 Research Paper Draft Deliverables

The final camera-ready 20-page manuscript, complete LaTeX source, and publication figures are organized in [`paper_materials/`](paper_materials/):

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Compiled Paper PDF** | [`paper_materials/research_paper.pdf`](paper_materials/research_paper.pdf) | **Final 20-page camera-ready research paper** (*Title: "An Efficient Framework for Structural Correspondence and Partial Matching in Attributed Graphs"*, 0 warnings, publication quality). |
| **LaTeX Source** | [`paper_materials/research_paper.tex`](paper_materials/research_paper.tex) | Complete LaTeX manuscript source with all mathematical proofs, lemmas, and algorithms. |
| **Live Benchmarking Tables** | [`paper_materials/live/`](paper_materials/live/) | Macro definitions (`live_values.tex`) and automated empirical tables (`tab_synthetic.tex`, `tab_social_network.tex`, `tab_image.tex`, `tab_ablation.tex`, `tab_3d.tex`). |
| **Vector Figures (PDF/PNG)** | [`paper_materials/figures/`](paper_materials/figures/) | High-resolution publication plots (`fig_master_benchmark_overview.pdf`, `fig_space_time_complexity_frontier.pdf`, `fig_pareto_balance.pdf`, `fig_topologies_size.pdf`). |
| **Reference Literature** | [`paper_materials/base_paper/`](paper_materials/base_paper/) | Foundational 2018 IEEE TNNLS reference paper by Yang et al. |

---

## 🛠️ Important Implementation Files & File Paths

Below is the complete index of all core mathematical solvers, domain-specific engines, benchmarking suites, and interactive web tools:

### 1. Core Fast-WCS / AURA-Net Optimization Engines
| File Name | File Path | Core Responsibility |
| :--- | :--- | :--- |
| `fastwcs_solver.py` | [`fastwcs_solver.py`](fastwcs_solver.py) | **Master Solver Factory:** Unified entrypoint `get_solver(domain)` exposing plug-and-play interfaces for synthetic, vision, social, and molecular domains. |
| `gnccp_advanced_v3.py` | [`gnccp_advanced_v3.py`](gnccp_advanced_v3.py) | **Production Fast-WCS Optimization Core:** Implements Lemma 1 rank-one co-matching factorization ($O(N^3)$ vs $O(N^4)$ tensor), Theorem 1 exact closed-form Level-3 BLAS gradient, Theorem 2 Cardano analytical quartic line search, exact augmented Jonker-Volgenant oracle, and vectorized 2-opt combinatorial refinement. |
| `gnccp_advanced_v4_wloa.py` | [`gnccp_advanced_v4_wloa.py`](gnccp_advanced_v4_wloa.py) | **WL-OA Augmented Core:** Advanced GNCCP coupled with Weisfeiler-Lehman Optimal Assignment subtree kernels for expressive structural node coloring. |
| `gnccp_baseline_2018.py` | [`gnccp_baseline_2018.py`](gnccp_baseline_2018.py) | **Classical Baseline Solver:** Yang et al. 2018 original heuristic grid-search implementation used for empirical speedup and accuracy baselines. |

### 2. Domain-Specific Matching Engines
| File Name | File Path | Core Responsibility |
| :--- | :--- | :--- |
| `fastwcs_synthetic.py` | [`fastwcs_synthetic.py`](fastwcs_synthetic.py) | **Multi-Topology Synthetic Engine:** Benchmark generator and matcher for Erdős–Rényi, Barabási–Albert, Watts–Strogatz, and Random Regular graphs with controlled noise and outlier churn. |
| `fastwcs_social_network.py` | [`fastwcs_social_network.py`](fastwcs_social_network.py) | **Dynamic Social Network Engine:** Temporal community matching on Stanford SNAP network datasets using Weisfeiler-Lehman Tree-History Homotopy. |
| `fastwcs_image_sift.py` | [`fastwcs_image_sift.py`](fastwcs_image_sift.py) | **Planar Visual Keypoint Engine:** Evaluates Oxford Affine benchmark image pairs combining Delaunay spatial triangulation with 128-D SIFT and 256-bit binary ORB local appearance descriptors. |
| `fastwcs_molecular_3d.py` | [`fastwcs_molecular_3d.py`](fastwcs_molecular_3d.py) | **3D Conformer & Pharmacophore Engine:** Matches molecular topologies and coordinates, computing untrimmed Root-Mean-Square Deviation (RMSD) via closed-form Kabsch SVD superposition. |

### 3. Interactive Web Application & Demo (`website/`)
| File Name | File Path | Core Responsibility |
| :--- | :--- | :--- |
| `app.py` | [`website/app.py`](website/app.py) | **Flask Server & API:** Serves web views and exposes endpoints (`/api/run_match`, `/api/run_multimodel_benchmark`, `/api/run_domain_visual_match`, `/api/benchmark_data`). |
| `mapping.html` | [`website/mapping.html`](website/mapping.html) | **Bijective Map & Benchmark Viewer:** Interactive bipartite matching visualizer with live Cytoscape graph rendering and multi-solver benchmarking. |
| `index.html` | [`website/index.html`](website/index.html) | **Project Portal & Overview:** Hero landing page with architecture summary and navigation. |
| `demo.html` | [`website/demo.html`](website/demo.html) | **Interactive Demo:** Live side-by-side graph correspondence playground. |
| `docs.html` | [`website/docs.html`](website/docs.html) | **Technical Documentation:** Mathematical formulations, proofs, and complexity bounds. |
| `methodology.html` | [`website/methodology.html`](website/methodology.html) | **Theoretical Foundations:** Step-by-step breakdown of GNCCP homotopy and Cardano line search. |
| `all_domain_benchmark_results.json` | [`website/all_domain_benchmark_results.json`](website/all_domain_benchmark_results.json) | Pre-computed benchmark results across all 4 experimental domains. |

### 4. Social Network Benchmarking Suite (`socialnetwork/`)
| File Name | File Path | Core Responsibility |
| :--- | :--- | :--- |
| `benchmark_social_network.py` | [`socialnetwork/benchmark_social_network.py`](socialnetwork/benchmark_social_network.py) | End-to-end benchmark harness evaluating Stanford CollegeMsg dataset across varying noise regimes. |
| `visualize_social_network.py` | [`socialnetwork/visualize_social_network.py`](socialnetwork/visualize_social_network.py) | Visual analytics for community structure discovery and correspondence alignment. |
| `wl_oa_kernel.py` | [`socialnetwork/src/kernels/wl_oa_kernel.py`](socialnetwork/src/kernels/wl_oa_kernel.py) | Fast vectorized Weisfeiler-Lehman Optimal Assignment kernel implementation. |

---

## 🚀 Quickstart & Execution Guide

### 1. Launch Interactive Web Demonstration Locally
```bash
cd website
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to explore:
- **Bijective Map & Benchmark Viewer:** [http://127.0.0.1:8000/mapping.html](http://127.0.0.1:8000/mapping.html)
- **Live Interactive Demo:** [http://127.0.0.1:8000/demo.html](http://127.0.0.1:8000/demo.html)
- **Mathematical Docs:** [http://127.0.0.1:8000/docs.html](http://127.0.0.1:8000/docs.html)

### 2. Run Direct Python Solver Benchmark
```python
from fastwcs_solver import get_solver
from fastwcs_synthetic import make_synthetic_benchmark_instance

# Generate a benchmark instance (M=20, N=25, L=15)
A_G, A_H, C, P_gt = make_synthetic_benchmark_instance(M=20, N=25, L=15, noise_ratio=0.10)

# Instantiate Fast-WCS solver
solver = get_solver('synthetic')
result = solver.match(A_G, A_H, C, L=15, alpha=0.5)

print(f"Objective Energy : {result.objective_val:.4f}")
print(f"Runtime          : {result.runtime_ms:.2f} ms")
print(f"Discovered Pairs : {len(result.matches)}")
```

### 3. Run Stanford Social Network Benchmark
```bash
cd socialnetwork
python benchmark_social_network.py
```

---

## 🌐 1-Click Vercel Deployment

The web application is fully pre-configured for instant zero-configuration deployment to **Vercel Serverless Functions**:

- **Serverless Entrypoint:** [`api/index.py`](api/index.py)
- **Routing Rules:** [`vercel.json`](vercel.json)
- **Lightweight Dependencies:** [`requirements.txt`](requirements.txt) (`Flask`, `numpy`, `scipy`, `networkx`)

To deploy to Vercel in 1 click:
```bash
vercel --prod
```
Or import the repository directly via the [Vercel Dashboard](https://vercel.com/new).

---

## 📊 Summary of Theoretical & Algorithmic Results

| Metric / Innovation | Classical GNCCP (2018) | Fast-WCS / AURA-Net (Ours) | Advantage |
| :--- | :--- | :--- | :--- |
| **Tensor Space Complexity** | $\mathcal{O}(N^4)$ Kronecker tensor | $\mathcal{O}(N^2)$ Rank-one factorization | **Eliminates memory bottleneck** ($<4$ MB footprint) |
| **Gradient Computation** | $\mathcal{O}(N^4)$ tensor contraction | $\Theta(N^3)$ via 3 Level-3 BLAS GEMMs | **Machine precision exactness** ($\epsilon_{\text{mach}} \approx 10^{-16}$) |
| **Step-Size Line Search** | Heuristic 100-point grid search | Exact Cardano quartic root solver | **$4.8\times$ faster**, provably optimal $\alpha^*$ |
| **Partial Cardinality** | Heuristic truncation | Augmented Jonker-Volgenant oracle | **Strict exact-$L$ bijection** (0 constraint violations) |
| **Discrete Polishing** | Naive rounding | Vectorized BLAS-3 2-Opt refinement | **Recovers global discrete optimum** |
| **Benchmark Speedup** | Baseline ($1.0\times$) | **$12\times$ to $48\times$ wall-clock speedup** | Sub-second execution across all topologies |

---

## 👥 Research Team & Supervision

**Department of Computer Science & Engineering**  
**Amrita Vishwa Vidyapeetham, Coimbatore, India**

- **Faculty Supervisor:** **Dr. T. Ramraj** (`t_ramraj@cb.amrita.edu`) — TIFAC-CORE / CSE
- **Shaik Abdus Sattar** — `CB.SC.U4CSE23245` (`cb.sc.u4cse23245@cb.students.amrita.edu`)
- **A.C Sanhitha Reddy** — `CB.SC.U4CSE23206` (`cb.sc.u4cse23206@cb.students.amrita.edu`)
- **Meera J Vaishnav** — `CB.SC.U4CSE23633` (`cb.sc.u4cse23633@cb.students.amrita.edu`)
- **Geethika G Nair** — `CB.SC.U4CSE23616` (`cb.sc.u4cse23616@cb.students.amrita.edu`)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
