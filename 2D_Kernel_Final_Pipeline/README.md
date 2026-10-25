# WCS-GNCCP: Hierarchical 2D-3D Molecular Graph Kernel

This repository contains the official implementation of the WCS-GNCCP algorithm, a hybrid molecular graph kernel that combines 2D topological matching with 3D geometric alignment for drug discovery and virtual screening.

## Core Files & Execution Guide

### 1. The Core Algorithm
* **`wcs_gnccp.py`**: The main mathematical engine containing the Frank-Wolfe optimization, Hungarian solver, and Kabsch 3D alignment. (Do not run directly; imported by other scripts).

### 2. Speed & Competitor Benchmarking
* **`benchmark_competitors.py`**: Tests the CPU execution speed (in milliseconds) of our kernel against classic graph kernels like Weisfeiler-Lehman and Shortest Path.
* **`benchmark_pytorch.py`**: Tests the GPU/Tensor execution speed of the PyTorch-accelerated version of the kernel.

### 3. Accuracy & Performance Benchmarking
* **`benchmark_accuracy.py`**: Tests the end-to-end classification accuracy on a small subset of molecules using an SVM with Eigenvalue Ridge Correction, demonstrating how the 3D alignment filters out 2D decoys.
* **`final_results_generator.py`**: The heavy-duty evaluation script. Runs rigorous 5-Fold Cross Validation across multiple Tox21 datasets (Aromatase, p53, MMP) and prints a publication-ready accuracy/AUC table.

### 4. Cloud Deployment
* **`generate_kaggle_gpu.py`**: Generates the final PyTorch notebook structure optimized for massive parallelization on Kaggle Cloud GPUs.

## How to Run
For a full evaluation, run the scripts in the following order:
1. `python benchmark_competitors.py` (To verify speed)
2. `python benchmark_accuracy.py` (To verify the 3D decoy-filtering hypothesis)
3. `python final_results_generator.py` (To generate the final results table)
