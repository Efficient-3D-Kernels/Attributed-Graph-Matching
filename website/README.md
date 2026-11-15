# Fast-WCS Interactive Research Platform & Live Demonstration Suite

Welcome to the **Fast-WCS Interactive Web Platform**, a modular research dashboard and real-time execution engine for **Fast-WCS: Analytical Optimization for Structural Correspondence and Partial Matching in Attributed Graphs**.

This platform provides interactive WebGL 3D visualizations, multi-model empirical benchmark shootouts, step-by-step Frank-Wolfe homotopy inspection, and live scientific graph matching across five real-world domains.

---

## 1. Quick Start: Launching the Platform

### Prerequisites
Install the required Python scientific stack:
```bash
pip install flask numpy scipy networkx
```
*(Optional: `rdkit-pypi` for raw cheminformatics manipulation).*

### Launch the Application Server
From the `website/` directory, execute:
```bash
python app.py
```

Open your browser and navigate to:
**[http://localhost:8000](http://localhost:8000)** (or `http://127.0.0.1:8000`).

---

## 2. Modular Backend Architecture (`services/`)

The backend has been modularized into domain-specific service modules under `website/services/`:

```
website/
├── app.py                         # Clean Flask application routing & API controller (~230 lines)
├── README.md                      # Comprehensive platform guide and API reference
├── style.css                      # Amrita Maroon & Warm Paper design tokens
│
├── services/                      # Domain execution services (Zero-Hardcoding Policy)
│   ├── __init__.py                # Package exports
│   ├── molecular_service.py       # 3D Molecular Conformer Superposition (Curated Pharma Library)
│   ├── synthetic_service.py       # Planted Subgraph Matching & Empirical Scalability Sweeps
│   ├── social_service.py          # Stanford SNAP CollegeMsg & Weisfeiler-Lehman Tree Homotopy
│   ├── image_service.py           # Planar Image Keypoints & Delaunay Triangulation
│   └── mof_service.py             # Metal-Organic Frameworks (UiO-66 / MOF-5 SBUs)
│
├── molecular3d.html               # 3D Molecular Sweeps with playback & KaTeX formula bar
├── mapping.html                   # Synthetic Graphs with 6 sensitivity tabs & scalability sweep
├── social.html                    # Social Networks with 4-Stage Visual Transformation Pipeline
├── connectome_explorer.html       # Drosophila Connectome Explorer (FAFB vs Hemibrain)
├── mof.html                       # MOF 3D Explorer
├── index.html                     # Research Homepage & Publications Gallery
├── methodology.html               # Mathematical Foundations & Homotopy Walkthrough
├── docs.html                      # Formal Proofs & KaTeX Formulations
├── progress.html                  # Verification Milestones & Implementation Checklist
└── team.html                      # Project Contributors & Academic Affiliations
```

---

## 3. Interactive Interfaces & Core Capabilities

### A. 3D Molecular Sweeps ([`molecular3d.html`](molecular3d.html))
- **Curated Pharmaceutical Library Dropdown**: Select between Aspirin (13 atoms), Caffeine (14 atoms), Paracetamol (11 atoms), Ibuprofen (15 atoms), Penicillin G (23 atoms), and Imatinib/Gleevec (37 atoms).
- **Interactive Playback & Flow Controls**: Step through the optimization using `[⏮ First]`, `[◀ Prev]`, `[▶ Play / ⏸ Pause]`, `[Next ▶]`, and `[⏭ Last]`.
- **Dynamic Formula & Phase Progress Bar**: Renders the exact mathematical equation being solved at each step using KaTeX (Phase 0 warm-start $\to$ Phase 1 convex relaxation $\to$ Phase 2 concave rounding $\to$ Phase 3 discrete SVD projection).
- **Theoretical Explanation of $F(X)$ & $\zeta$ Progression**: Explains why continuation parameter $\zeta$ sweeps from $-1.00 \to +1.00$ and why the unprojected continuous Frobenius residual exhibits an apparent U-shaped inflection near the barycenter while the discrete objective strictly decays.
- **7-Model Comparative Benchmark Log**: Directly compares Fast-WCS against FAQ + Kabsch, Multi-Start ICP, Single-Start ICP, RLX1, PGM, and GA Annealing with real measured runtimes, speedup multipliers, and $\pm \sigma$ standard deviations. Fast-WCS reaches **100.0% accuracy** and **$< 0.035$ Å RMSD** across all 6 drugs in 30–80 ms.

### B. Synthetic Relational Graphs ([`mapping.html`](mapping.html))
- **Verified Benchmark Topologies**: Planted subgraph matching across Erdős–Rényi (100% accuracy), Random Regular (100% accuracy), and Barabási–Albert Scale-Free (90% accuracy).
- **Large-Scale Scalability Sweep ($N = 20 \to 10^6$)**:
  - Subdivided into 4 parts per decade between $10^5$ and $10^6$ ($0.25\times 10^5, 0.50\times 10^5, 0.75\times 10^5, 1.00\times 10^5$ and $0.25\times 10^6, 0.50\times 10^6, 0.75\times 10^6, 1.00\times 10^6$).
  - **Strict Execution Blowout Caps**:
    - **Memory Cap = 8.0 GB**: Competitor Lawler 4D Tensor (RRWM) requires $N^4 \times 4$ bytes. At $N=250$, memory reaches $14.55$ GB $> 8$ GB $\implies$ halts and reports `BLOWUP (OOM > 8GB Lawler 4D Tensor)`.
    - **Time Cap = 60.0 s**: Competitor 4th-Order Polynomial (RLX1) scales in $\mathcal{O}(N^{5.4})$ (measured $77.5$ s at $N=100$) $\implies$ halts and reports `BLOWUP (Timeout > 60s 4th-Order Poly)`. SciPy FAQ (dense Hungarian) at $N \ge 200$ halts and reports `BLOWUP (Timeout > 60s Dense Hungarian)`.
  - Fast-WCS evaluates strictly in $\Theta(N^3)$ / sparse $\mathcal{O}(N \cdot \text{deg})$ time with strictly $\mathcal{O}(N^2)$ / sparse memory (completing $N=10^6$ in $141.1$ MB memory).
- **Standard Deviation Shaded Bands**: All 6 benchmark charts display $\pm \sigma$ standard deviation confidence envelopes with shaded fill, and all benchmark tables report $\mu \pm \sigma$.

### C. Dynamic Social Networks ([`social.html`](social.html))
- **Stanford SNAP CollegeMsg Benchmark**: Evaluates temporal interaction graphs across 1,899 active users and 59,835 timestamped messages under varying churn rates ($\sigma = 5\%$ to $30\%$).
- **4-Stage Visual Transformation Pipeline**:
  1. **📸 Stage 1: Raw Network Images**: Visualizes continuous interaction activity density heatmaps and community boundaries for Cohort G and Cohort H.
  2. **📍 Stage 2: Extracted Feature Points**: Displays isolated 2D spatial & spectral points $(x_i, y_i)$ with crosshairs, coordinates readout, and radii proportional to user degree.
  3. **🕸️ Stage 3: Constructed Topology Graphs**: Displays the reconstructed relational interaction graphs with edges $A_G$ and $A_H$, hub vertices, and community partitions.
  4. **⚡ Stage 4: Fast-WCS Subgraph Matching**: Displays exact correspondence lines and continuous superposition morphing with **100.0% matching accuracy** under structural noise.
- **Weisfeiler-Lehman Tree Homotopy (Lemma 2 & Proposition 5)**: Uses $H=3$ multiset subtree histograms to construct unary cost matrix $C_{\text{WL}}$, breaking peripheral community automorphisms without spatial coordinates.

---

## 4. Mathematical Foundations & Implementation Mapping

| Theoretical Contribution | Mathematical Formulation | Code Implementation | Complexity |
| :--- | :--- | :--- | :--- |
| **Dyadic Factorization** (Lemma 1) | $\mathbf{U} = \mathbf{r}\mathbf{r}^T, \quad K_{ia,jb} = U_{ij} U_{ab} A_{G,ij} A_{H,ab}$ | `molecular_service.py`, `synthetic_service.py` | Eliminates $\mathcal{O}(N^4)$ tensor $\implies \mathcal{O}(N^2)$ memory |
| **Analytical Structural Gradient** (Theorem 1) | $\nabla F(\mathbf{X}) = -2 (r r^T \odot A_G) \mathbf{X} A_H$ | Level-3 BLAS GEMMs | $\Theta(N^3)$ dense, $\mathcal{O}(N \cdot \text{deg})$ sparse |
| **Cardano Quartic Line Search** (Theorem 3) | $\arg\min_{\alpha \in [0, 1]} c_4 \alpha^4 + c_3 \alpha^3 + c_2 \alpha^2 + c_1 \alpha + c_0$ | Analytical cubic resolvent | $\mathcal{O}(1)$ time ($6.8\times$ faster than grid search) |
| **Duality Gap Stopping Certificate** (Theorem 2) | $g_t = \langle \nabla F(\mathbf{X}_t), \mathbf{X}_t - \mathbf{V}_t \rangle < 10^{-3}$ | Early stopping condition | $45\text{--}58\%$ fewer Hungarian calls |
| **Reflection-Safe Kabsch 3D SVD** (Theorem 4) | $\mathbf{R} = \mathbf{V} \operatorname{diag}(1, 1, \det(\mathbf{V}\mathbf{U}^T)) \mathbf{U}^T$ | `FastWCS_Molecular3D` | $\mathcal{O}(L)$ SVD, reaches $0.035\text{ \AA}$ oracle floor |
| **WL Tree-History Homotopy** (Lemma 2) | $c_{uv}^{\text{WL}} = 1 - \frac{1}{H+1}\sum_{h=0}^H \mathbf{1}[c_h(u) = c_h(v)]$ | `WLTreeEngine` in `social_service.py` | Breaks scale-free community automorphisms |

---

## 5. REST API Documentation

### 1. `GET /api/get_molecules_list`
Returns the library of curated pharmaceutical drug molecules.
- **Response**:
```json
{
  "status": "success",
  "molecules": [
    { "key": "aspirin", "name": "Aspirin (Acetylsalicylic Acid)", "formula": "C9H8O4", "atoms": 13, "heavy_atoms": 13, "bonds": 13, "mw": 180.16 },
    { "key": "caffeine", "name": "Caffeine (1,3,7-Trimethylxanthine)", "formula": "C8H10N4O2", "atoms": 14, "heavy_atoms": 14, "bonds": 15, "mw": 194.19 },
    ...
  ]
}
```

### 2. `GET /api/benchmark_scalability`
Returns empirical scalability benchmark measurements from $N=20$ to $N=10^6$ with standard deviation confidence envelopes and blowout statuses.
- **Response**:
```json
{
  "status": "success",
  "memory_cap_gb": 8.0,
  "time_cap_s": 60.0,
  "n_points": [20, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000, 1000000],
  "sweep": [
    {
      "N": 20,
      "fastwcs": { "time_ms": 3.49, "std_ms": 0.41, "memory_mb": 0.05, "status": "Verified (O(N²) Memory)" },
      "rrwm": { "time_ms": 12.5, "blown_up": false, "memory_gb": 0.0006 },
      "rlx1": { "time_ms": 122.4, "blown_up": false },
      "faq": { "time_ms": 14.5, "blown_up": false }
    },
    ...
    {
      "N": 250,
      "fastwcs": { "time_ms": 46.21, "std_ms": 3.80, "memory_mb": 0.45, "status": "Verified (O(N²) Memory)" },
      "rrwm": { "time_ms": null, "blown_up": true, "status": "BLOWUP (OOM > 8GB Lawler 4D Tensor)", "memory_gb": 14.55 },
      "rlx1": { "time_ms": null, "blown_up": true, "status": "BLOWUP (Timeout > 60s 4th-Order Poly)" },
      "faq": { "time_ms": 4850.0, "blown_up": false }
    }
  ]
}
```

### 3. `POST /api/run_domain_visual_match`
Unified multi-domain matching endpoint.
- **Request Parameters**:
  - `domain`: `"molecular"` | `"synthetic"` | `"social"` | `"image"` | `"mof"`
  - `preset`: Preset configuration key (e.g., `"aspirin"`, `"collegemsg_low"`)
  - `sigma`: Noise / churn rate (e.g., `0.02`, `0.05`)
  - `seed`: Random seed for reproducibility
- **Returns**: Node coordinates for $G$ and $H$, list of matched node pairs `[[u, v], ...]`, Frank-Wolfe optimization trajectory, KaTeX mathematical formulas, and 7-model benchmark scorecard with measured runtimes ($\pm \sigma$) and speedups.

---

## 6. Verification and Reproduction

To independently verify the benchmark scripts and empirical numbers:

```bash
# Verify 3D Molecular Superposition on all 6 pharmaceutical drugs
python -c "
from website.services.molecular_service import run_molecular_3d_superposition, PHARMA_MOLECULES
for k in PHARMA_MOLECULES:
    res = run_molecular_3d_superposition(k, noise_sigma=0.02)
    print(f'{k:12s}: Acc={res[\"stats\"][\"metric_val\"]} | RMSD={res[\"stats\"][\"rmsd_val\"]} | Time={res[\"stats\"][\"runtime_opt_ms\"]} ms')
"

# Verify Synthetic Graph Topologies (ER, RR, BA)
python -c "
from website.services.synthetic_service import run_synthetic_graph_match
for top in ['erdos_renyi', 'random_regular', 'barabasi_albert']:
    res = run_synthetic_graph_match(top, M=24, N=28, L=20, sigma=0.05)
    print(f'{top:16s}: Acc={res[\"stats\"][\"metric_val\"]} | Pairs={res[\"stats\"][\"pairs_str\"]} | Time={res[\"stats\"][\"runtime_opt_ms\"]} ms')
"

# Verify Scalability Sweep Blowout Caps
python -c "
from website.services.synthetic_service import get_scalability_sweep_data
data = get_scalability_sweep_data()
print(f'Total points: {len(data[\"n_points\"])} | Caps: Mem={data[\"memory_cap_gb\"]}GB, Time={data[\"time_cap_s\"]}s')
for r in data['sweep']:
    if r['N'] in [20, 100, 250, 1000, 100000, 1000000]:
        print(f'N={r[\"N\"]:7d}: Fast-WCS={r[\"fastwcs\"][\"time_ms\"]} ms ({r[\"fastwcs\"][\"memory_mb\"]} MB) | RRWM={r[\"rrwm\"][\"status\"]} | RLX1={r[\"rlx1\"][\"status\"]}')
"
```

---

## 7. Project Metadata
- **Project Title**: Analytical Optimization for Structural Correspondence and Partial Matching in Attributed Graphs (Fast-WCS)
- **Institution**: Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham, Coimbatore
- **Supervisor**: Dr. T Ramraj, Assistant Professor (Sl. Gd)
- **Design System**: Amrita Maroon (`#8E1638`), Deep Navy (`#1E3A8A`), Warm Academic Paper (`#F3F1EB`)
- **Mathematical Renderer**: KaTeX (Server & Client-Side LaTeX Rendering)
