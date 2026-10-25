# Core Pipeline: Fast-WCS & Chemoinformatics Superposition

This folder contains the verified, clean production codebase:

### 1. Universal Domain-Agnostic Graph Matching Engine
- **`gnccp_generalized.py`** (and alias **`fast_wcs_general.py`**):
  - A domain-agnostic mathematical solver for finding common subgraphs across arbitrary scientific fields:
    - **Computer Vision & Robotics:** Spatial point clouds, Delaunay meshes, SIFT/ORB keypoints with closed-form SVD Kabsch 2D/3D registration.
    - **Social & Complex Networks:** NetworkX graph matching (ego-networks, communities, citation graphs).
    - **Neuroscience & Connectomics:** Synaptic wiring graphs and neuron arborization (Hemibrain vs. FAFB).
    - **Materials Science & MOFs:** Crystal lattices and secondary building units (SBUs).
    - **Pure Mathematics:** Arbitrary weighted adjacency matrices and unary cost matrices.
  - To test all domains live:
    ```bash
    python gnccp_generalized.py
    ```

### 2. Specialized Chemoinformatics & Pharmacophore Engine
- **`wcs_gnccp.py`**:
  - The molecular matching engine tailored for SDF libraries with 10D atom feature vectors, element-mismatch penalties, and 3D conformer alignment.

### 3. Verification & Benchmark Suites
- **`run_live_real_verification.py`**:
  - Master live script computing ROC-AUC, PR-AUC, Early Enrichment, and 3D Kabsch superposition on real SDF libraries (`Data/`).
- **`verify_all_tables_and_proofs.py`**:
  - Full audit verifying Table I (complexity), Table II (macromolecular speedups up to 5,717x), and alpha parameter sweeps.
- **`IEEE_TNNLS_Live_Benchmark_Proof.ipynb`**:
  - Fully executed master Jupyter notebook with live outputs, log-log regressions, and figures.
- **`build_and_run_notebook.py`**:
  - Automated compiler and runner for the benchmark notebook.
