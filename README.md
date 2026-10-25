# An Efficient Framework for Structural Correspondence and Partial Matching in Attributed Graphs

**An optimization-based framework for robust partial graph matching across heterogeneous relational domains.**

---

## Overview

Graph matching is the fundamental mathematical problem of identifying corresponding nodes and shared structural patterns between two graphs, even when their sizes, connectivity patterns, edge weights, or node attributes differ.

This research presents an enhanced **Graduated Non-Convexity and Concavity Procedure (GNCCP)** framework for Weighted Common Subgraph (WCS) matching (**Fast-WCS**). The approach formulates graph correspondence as a constrained optimization problem over partial permutation matrices and progressively transforms a convex relaxation into a non-convex objective to recover discrete, certified node correspondences.

The framework focuses on improving computational efficiency, structural consistency, and robustness while supporting partial matching across complex topological domains: synthetic random regular and scale-free networks, real-world planar visual keypoints, dynamic social networks, and attributed crystalline and chemical topologies.

---

## Research Objectives

- Develop an efficient optimization framework for weighted common subgraph matching with strict $\mathcal{O}(N^2)$ memory footprint.
- Recover certified partial one-to-one correspondences between nodes in attributed graphs ($\sum_{i,j} X_{ij} = L$).
- Eliminate the $\mathcal{O}(N^4)$ memory barrier of Kronecker affinity tensors and quadratic penalty approximations.
- Improve computational efficiency through Lemma 1 rank-one factorization and Theorem 1 exact closed-form Level-3 BLAS gradient computation.
- Guarantee exact step sizes via Cardano's analytical cubic line search and exact Hungarian linear minimization oracles.
- Evaluate matching performance across social, chemical, image, and multi-topology benchmark domains.
- Investigate robustness under structural perturbations, noise sweeps, and outlier churn.

---

## Core Methodology & Architectural Innovations

The proposed framework combines graph-theoretic formulations with mathematical optimization to identify structurally corresponding subgraphs:

### 1. Weighted Common Subgraph Formulation
Given two attributed graphs $G$ and $H$, the objective minimizes combined structural mismatch and unary attribute discrepancy:
$$F(\mathbf{X}) = \alpha\, \|\mathbf{U} \circ \hat{\mathbf{A}}_G - \mathbf{X}\hat{\mathbf{A}}_H\mathbf{X}^T\|_F^2 + (1 - \alpha)\operatorname{tr}(\mathbf{C}^T\mathbf{X})$$
over the convex hull of partial permutation matrices $\mathcal{D}_L = \operatorname{conv}(\mathcal{P}_L)$.

### 2. Six Core Algorithmic Innovations
1. **Lemma 1 (Rank-One Factorization):** Evaluates the co-matching indicator $\mathbf{U} = \mathbf{r}\mathbf{r}^T$ where $\mathbf{r} = \mathbf{X}\mathbf{1}_N$ in strictly $\mathcal{O}(MN + M^2)$ work, eliminating cubic intermediate tensor contractions.
2. **Theorem 1 (Exact Analytical Gradient):** Computes $\nabla_{\mathbf{X}} f(\mathbf{X}) = 4(\mathbf{R} \circ \hat{\mathbf{A}}_G)\mathbf{r}\mathbf{1}_N^T - 4\mathbf{R}\mathbf{X}\hat{\mathbf{A}}_H$ in strictly $\Theta(N^3)$ operations via three dense Level-3 BLAS GEMMs with zero finite-difference error.
3. **Theorem 2 (Cardano Exact Quartic Line Search):** Solves the 1D step-size subproblem $\lambda^* = \arg\min_{\lambda \in [0, 1]} J_\zeta(\mathbf{X} - \lambda \mathbf{D})$ in closed-form $\mathcal{O}(1)$ time using Cardano's cubic root formula.
4. **Augmented Burkard Hungarian Oracle:** Constructs an exact $(M+N-L) \times (M+N-L)$ augmented cost matrix guaranteeing $\sum_{i,j} Y^*_{ij} = L$ without heuristic truncation.
5. **Proposition 3 (Vectorized BLAS-3 Combinatorial 2-Opt Refinement):** Rapidly polishes discrete permutations in $\mathcal{O}(M)$ time per accepted swap, eliminating local discretization loss.
6. **Multi-Domain Adaptations:** Multi-start exploratory homotopy for symmetric regular graphs, planar Delaunay spatial invariants for visual keypoints, and Weisfeiler--Lehman Tree-History Homotopy for dynamic social networks.

---

## Application Domains & Experimental Validation

### 1. Dynamic Social Networks (Stanford CollegeMsg)
- Evaluated on 1,899 college students and 59,835 timestamped interactions over 193 days.
- Fast-WCS with Weisfeiler--Lehman Tree-History Homotopy maintains **100.0% accuracy** at 30% structural perturbation noise ($N=100$), executing in 126 ms.

### 2. Multi-Topology Sweeps (Erdős-Rényi, Random Regular, Barabási-Albert)
- Breaks automorphic symmetry traps on Random Regular graphs (85.5% accuracy vs. 4.5% for FAQ).
- Leverages scale-free hub hierarchies on Barabási-Albert networks (98.8% accuracy).

### 3. Real-World Vision (Oxford Affine Dataset)
- 16 image pairs with SIFT unary descriptors and Delaunay geometric triangulations.
- Fast-WCS delivers 67.3% matching accuracy in 18.3 ms, running $30\times\text{--}74\times$ faster than quadratic baselines.

### 4. Large-Scale Scalability Grid ($N=150$)
- Scales strictly in tens of milliseconds (80.8 ms at $N=150$), providing $20\times\text{--}72.5\times$ speedup over exact GNCCP with LP oracles, with strictly $\mathcal{O}(N^2)$ memory footprint ($<1.8$ MB).

---

## Directory Organization

```text
.
├── paper_materials/                   # [PUBLICATIONS & FINAL LATEX]
│   ├── research_paper.pdf             # Camera-ready 20-page manuscript
│   ├── research_paper.tex             # LaTeX source code
│   ├── figures/standalone/            # Publication-quality vector figures (PDF & PNG)
│   ├── live/                          # Live LaTeX tables and values
│   └── base_paper/                    # Yang et al. IEEE TNNLS 2018 reference files
│
├── website/                           # [INTERACTIVE WEB DOCUMENTATION & DEMO]
│   ├── app.py                         # Flask application with live benchmark runner
│   ├── index.html, mapping.html, ...  # Interactive frontend with live Cytoscape graphs
│   ├── README.md                      # Website documentation & quickstart
│   ├── WEBSITE_DATA_AND_SCRIPTS_GUIDE.md # Complete mapping of data and scripts
│   └── all_domain_benchmark_results.json # Cached multi-domain benchmark results
│
├── socialnetwork/                     # [DYNAMIC SOCIAL NETWORK SUITE]
│   ├── benchmark_social_network.py    # Stanford CollegeMsg benchmark harness
│   ├── benchmark_wl.py, benchmark_wl_oa.py # Weisfeiler-Lehman tree kernel benchmarks
│   ├── plot_social_network_benchmark.py # Multi-panel publication plotting scripts
│   └── results/                       # Generated matchings, tables, and Pareto plots
│
├── core_pipeline/                     # [CORE FAST-WCS SOLVER & AUDITS]
│   ├── fastwcs_solver.py              # Master vectorized Fast-WCS solver
│   ├── run_live_real_verification.py  # Live dataset verification suite
│   ├── verify_all_tables_and_proofs.py# Complexity, alpha sweeps, and proof audits
│   └── IEEE_TNNLS_Live_Benchmark_Proof.ipynb # Executed Jupyter proof notebook
│
├── benchmarks/                        # [BENCHMARK SCRIPTS & HARNESSES]
│   ├── scripts/                       # Synthetic, image, scaling, and topology runners
│   └── reports/                       # Formal markdown experiment reports
│
├── 2D_Kernel_Final_Pipeline/          # [ALGORITHMIC PIPELINES & ABLATIONS]
│   ├── gnccp_advanced_v3.py           # Production Fast-WCS solver implementation
│   ├── gnccp_advanced_v4_wloa.py      # WL-OA augmented social network solver
│   └── Final_Screening_Pipeline.ipynb # Screening pipeline notebook
│
├── README.md                          # Master project documentation
└── .gitignore                         # Clean git hygiene (excludes heavy Data/)
```

---

## Quickstart & Reproduction

### 1. Launch Interactive Web Documentation & Live Demo
```bash
cd website
python app.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to view:
- **Bijective Map & Benchmark Viewer:** Interactive graph matching visualization with live solver execution.
- **Master Documentation:** Theoretical derivations, theorems, and implementation guides.
- **Cross-Domain Benchmarks:** Synthetic, Multi-Topology, Oxford Vision, and Social Network evaluations.

### 2. Run Master Benchmark Audits
```bash
# Verify Table I-VI derivations and theoretical proofs:
cd core_pipeline
python verify_all_tables_and_proofs.py

# Run dynamic social network benchmarks:
cd ../socialnetwork
python benchmark_social_network.py
```

### 3. View Camera-Ready Research Paper
- **Compiled PDF:** `paper_materials/research_paper.pdf` (20 pages, camera-ready).
- **LaTeX Source:** `paper_materials/research_paper.tex`.

---

## Research Team

**Guide:** Dr. T. Ramraj  
Department of Computer Science and Engineering  
Amrita Vishwa Vidyapeetham, Coimbatore, India  
`t_ramraj@cb.amrita.edu`

**Project Team:**
- **Dr. T Ramraj** — Research Guide & Principal Investigator
- **Allampati Chenchu Sanhitha Reddy** — `cb.sc.u4cse23206@cb.students.amrita.edu`
- **Shaik Abdus Sattar** — `cb.sc.u4cse23245@cb.students.amrita.edu`
- **Geethika G Nair** — `cb.sc.u4cse23616@cb.students.amrita.edu`
- **Meera J Vaishnav** — `cb.sc.u4cse23633@cb.students.amrita.edu`

---

## References & Foundational Literature

1. **Yang et al.** (2018) — *An Algorithm for Finding the Most Similar Given-Sized Subgraphs in Two Weighted Graphs*, IEEE Trans. Neural Netw. Learn. Syst.
2. **Liu et al.** (2015) — *A Graduated Non-Convexity and Concavity Procedure for Graph Matching*, IEEE TPAMI.
3. **Zaslavskiy et al.** (2009) — *A Path Following Algorithm for the Graph Matching Problem*, IEEE TPAMI.
4. **Shervashidze et al.** (2011) — *Weisfeiler-Lehman Graph Kernels*, JMLR.
5. **Panzarasa et al.** (2009) — *Patterns and Dynamics of Users' Behavior in Online Communities (CollegeMsg)*, JASIST.
6. **Burkard et al.** (2009) — *Assignment Problems*, SIAM.
