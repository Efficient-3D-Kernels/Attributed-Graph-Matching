# An Efficient Framework for Structural Correspondence and Partial Matching in Attributed Graphs

**An optimization-based framework for robust partial graph matching across heterogeneous relational domains.**

## Overview

Graph matching is the problem of identifying corresponding nodes and shared structural patterns between two graphs, even when their sizes, connectivity patterns, edge weights, or node attributes differ.

This research explores an enhanced **Graduated Non-Convexity and Concavity Procedure (GNCCP)** framework for Weighted Common Subgraph (WCS) matching. The approach formulates graph correspondence as a constrained optimization problem over partial permutation matrices and progressively transforms a convex relaxation into a non-convex objective to recover discrete node correspondences.

The framework focuses on improving computational efficiency, structural consistency, and robustness while supporting partial matching across different types of attributed graphs.

## Research Objectives

- Develop an efficient optimization framework for weighted common subgraph matching.
- Recover partial one-to-one correspondences between nodes in attributed graphs.
- Improve computational efficiency through rank-one factorization and analytical gradient computation.
- Incorporate graph normalization, structural node features, and convergence-aware optimization.
- Evaluate matching performance across social, chemical, and biological graph domains.
- Investigate robustness under graph perturbations and structural changes.

## Core Methodology

The proposed framework combines graph-theoretic formulations with mathematical optimization to identify structurally corresponding subgraphs.

### 1. Weighted Common Subgraph Formulation

Given two attributed graphs \(G\) and \(H\), the objective is to identify a set of corresponding nodes that minimizes structural mismatch while accounting for node attributes and edge weights.

The correspondence is represented by a partial permutation matrix \(X\), where each selected node in one graph maps to at most one node in the other graph.

### 2. Enhanced GNCCP Optimization

The GNCCP framework gradually transforms the optimization objective from a convex formulation toward a non-convex formulation to obtain discrete graph correspondences.

The implementation incorporates:

- **Rank-one factorization:** Represents the co-matching matrix using \(U = rr^T\), where \(r = X\mathbf{1}\), reducing the cost of this intermediate computation.
- **Analytical gradient computation:** Uses a closed-form gradient for structural optimization.
- **Frank–Wolfe optimization:** Iteratively improves the relaxed correspondence matrix.
- **Hungarian algorithm:** Solves the linear assignment subproblem to obtain discrete correspondences.
- **Convergence monitoring:** Uses the Frank–Wolfe duality gap to assess convergence and support early stopping.
- **Additional optimization components:** Graph normalization, multi-scale structural features, Armijo line search, and Sinkhorn-style projection.

### 3. Cross-Domain Evaluation

The same underlying graph-matching formulation is investigated across three distinct application domains, with domain-specific graph representations and evaluation procedures.

## Application Domains

### Social Network Analysis

Social networks are represented as weighted graphs in which users are nodes and their interactions are weighted edges.

The framework is evaluated on the Stanford SNAP CollegeMsg temporal social-network dataset to study:

- Structural correspondence under controlled graph perturbations.
- Matching robustness at different perturbation levels.
- Partial matching between temporally separated interaction graphs.
- Optimization convergence and structural similarity.

### Metal–Organic Framework (MOF) Crystals

MOF structures are represented as attributed graphs in which atoms are nodes and chemical bonds are edges.

Using structures from the QMOF database, the research investigates:

- Structural motif screening in metal–organic framework crystals.
- Matching based on atom attributes and bond topology.
- Candidate validation using three-dimensional structural alignment.
- Geometric verification using the Kabsch algorithm and root mean square deviation (RMSD).

The investigated motif includes the Zr–O–C–O–Zr bridging-carboxylate linkage associated with UiO-family structures.

### Drosophila Brain Connectomes

Brain connectomes are represented as weighted graphs in which neurons are nodes and synaptic connections are weighted edges.

The framework is applied to connectome data from neuPrint to investigate:

- Structural correspondence between neuronal networks.
- Connectivity-based node representations.
- Matching across independently reconstructed fly brain datasets.
- Computational efficiency on large biological graphs.

The experiments focus on the Mushroom Body region and use binned synaptic connectivity information to construct graph representations.

## Experimental Evaluation

The research evaluates the framework using domain-specific experiments and performance measures.

| Evaluation aspect | Purpose |
|---|---|
| Matching accuracy | Measures the correctness of recovered correspondences where ground truth is available. |
| Perturbation robustness | Examines performance as graph structures are modified. |
| Structural similarity | Evaluates correspondence between graph topologies. |
| ROC-AUC and Youden's J-statistic | Assess screening performance and threshold selection in the MOF experiments. |
| RMSD | Measures geometric discrepancy during 3D structural verification. |
| Computational latency | Evaluates runtime and optimization efficiency. |

### Preliminary Results

Selected results reported in the project presentation include:

- **Social networks:** Matching accuracy remained above 98% at perturbation levels up to 20%, reaching 92.33% at 30% perturbation in the reported 400-user experiment.
- **MOF crystals:** Rank-one factorization achieved a reported 42.2× speedup for the isolated co-matching computation at \(N = 500\) atoms.
- **Drosophila brain connectomes:** The reported Mushroom Body benchmark reduced execution time from 83 seconds to 14 seconds, approximately 5.9× faster than the baseline GNCCP implementation.

These are preliminary, experiment-specific results rather than guarantees of performance across every graph domain or dataset.

## Technology Stack

- **Programming language:** Python
- **Mathematical optimization:** GNCCP, Frank–Wolfe optimization
- **Graph matching:** Weighted Common Subgraph (WCS) formulation, partial assignment, Hungarian algorithm
- **Numerical computation:** Matrix-based optimization and analytical gradient computation
- **Structural verification:** Kabsch algorithm, singular value decomposition (SVD), RMSD
- **Classification and screening:** Support Vector Machine (SVM) kernel fusion, as part of the proposed extended pipeline
- **Data sources:** Stanford SNAP CollegeMsg, QMOF, neuPrint connectome datasets

## Research Contributions

The project investigates a unified approach to structural correspondence across heterogeneous attributed graphs.

Its key contributions include:

1. An optimization-based formulation for partial weighted graph matching.
2. Rank-one factorization to reduce the computational cost of co-matching matrix construction.
3. Analytical gradient computation for structural mismatch optimization.
4. Convergence-aware Frank–Wolfe optimization with assignment-based updates.
5. Cross-domain evaluation across social, chemical, and biological networks.
6. Integration of topological matching with proposed geometric verification and hybrid-kernel classification stages.

## Project Status

The project is being developed and evaluated in stages.

- [x] Weighted Common Subgraph formulation and enhanced GNCCP-based optimization.
- [x] Preliminary evaluation on social network data.
- [x] Preliminary evaluation on MOF crystal structures.
- [x] Preliminary evaluation on Drosophila brain connectomes.
- [ ] Further validation on controlled synthetic graph pairs with known ground-truth correspondences.
- [ ] Extended evaluation of 3D structural verification and hybrid SVM kernel fusion.

*The final two stages represent planned or ongoing work and should be updated as their implementations and evaluations are completed.*

## Repository Structure

The repository is intended to organize the implementation, experimental configurations, and evaluation results for the proposed framework.

```text
.
├── src/                    # Core graph-matching and optimization modules
├── data/                   # Dataset preparation and graph inputs
├── experiments/            # Domain-specific experiments
│   ├── social_networks/
│   ├── mof_crystals/
│   └── drosophila_connectome/
├── results/                # Experimental outputs and evaluation metrics
├── notebooks/              # Exploratory analysis and visualizations
├── requirements.txt        # Python dependencies
└── README.md
```

*This is a suggested repository organization; adapt it to match the actual files and directories in the repository.*

## Research Team

**Guide:** Dr. T. Ramraj  
Department of Computer Science and Engineering  
Amrita Vishwa Vidyapeetham, Coimbatore, India

**Project team**
- Allampati Chenchu Sanhitha Reddy
- Shaik Abdus Sattar
- Geethika G Nair
- Meera J Vaishnav

## References

- Liu, X., Fan, C., Liu, Y., and Li, H. B. — Research on GNCCP-based weighted common subgraph matching.
- Yang et al. — Weighted Common Subgraph matching and GNCCP optimization.
- Zaslavskiy et al. — Path Following for graph matching.
- Stanford SNAP — CollegeMsg temporal social-network dataset.
- QMOF — Quantum MOF database.
- neuPrint — Connectome data and analysis platform.

For the complete theoretical derivations, experimental configurations, and supporting literature, refer to the associated research paper and project documentation.

---

*This repository investigates efficient structural correspondence and partial matching in attributed graphs, with an emphasis on optimization, computational efficiency, and cross-domain applicability.*
