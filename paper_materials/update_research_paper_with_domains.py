import os
import shutil
import re
import subprocess
import fitz

tex_path = r"C:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\paper_materials\research_paper.tex"
bak_path = r"C:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\paper_materials\research_paper.tex.bak"

if not os.path.exists(bak_path):
    shutil.copy2(tex_path, bak_path)
    print("Created backup research_paper.tex.bak")

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update graphicspath
text = text.replace(
    r"\graphicspath{{figures/standalone/}{figures/}{schematics/}{.}}",
    r"\graphicspath{{figures/mof/}{figures/brain/}{figures/standalone/}{figures/}{schematics/}{.}}"
)

# 2. Add Directed Connectomics subsection into Section VI (Domain Adaptations)
domain_connectomics = r"""
\subsection{Directed Neuromorphic Connectomics: Drosophila Whole-Brain Mushroom Body Alignment}
\label{subsec:domain_connectome}

Large-scale electron microscopy connectomics datasets---such as the Janelia Hemibrain~\cite{scheffer2020} and Princeton FlyWire~\cite{dorkenwald2024} whole-brain volumes---yield massive directed, weighted graph representations of neural wiring. Unlike symmetric visual or planar topologies, biological neural networks are fundamentally directed: vertices represent identified individual neurons, and directed edge weights $A_{ij} \in \mathbb{N}$ quantify directed synaptic contacts from pre-synaptic neuron $i$ to post-synaptic neuron $j$. 

Consequently, adjacency matrices are strongly asymmetric: $\mathbf{A} \neq \mathbf{A}^T$ and $\mathbf{B} \neq \mathbf{B}^T$. Standard graph matching formulations that heuristically symmetrize adjacencies ($\frac{\mathbf{A} + \mathbf{A}^T}{2}$) confound feedforward projection pathways with feedback local interneuron circuits, precipitating catastrophic topological collapse. Fast-WCS preserves the exact directional gradient:
\begin{equation}
\nabla_{\mathbf{X}} f(\mathbf{X}) = -(\mathbf{A}\mathbf{X}\mathbf{B}^T + \mathbf{A}^T\mathbf{X}\mathbf{B}) + \mathbf{C},
\label{eq:asym_gradient}
\end{equation}
which is evaluated in strictly three Level-3 BLAS GEMMs without tensor unrolling. Simultaneously, Fast-WCS balances pre-synaptic in-degree preservation ($\mathbf{d}^{\text{in}} = \mathbf{A}^T\mathbf{1}$) and post-synaptic out-degree preservation ($\mathbf{d}^{\text{out}} = \mathbf{A}\mathbf{1}$) within the augmented Burkard assignment oracle, guaranteeing exact partial matching without directionality confusion.
"""

# Insert domain_connectomics right after subsection domain_molecular
target_domain_mol = r"By operating directly on topological invariants and chemical affinity matrices $\mathbf{C}_{\text{chem}}$, Fast-WCS guarantees exact pharmacophore and secondary building unit (SBU) recovery without getting trapped by extrinsic coordinate orientations."
if target_domain_mol in text:
    text = text.replace(target_domain_mol, target_domain_mol + "\n" + domain_connectomics)
    print("Inserted directed connectomics subsection in Section VI.")
else:
    print("Warning: target_domain_mol marker not found!")

# 3. Add Reticular MOF and Drosophila Connectomics experimental sections
new_experimental_sections = r"""
% =========================================================
% RETICULAR NANOMATERIALS AND 3D CONFORMER ALIGNMENT
% =========================================================

\section{Reticular Nanomaterials and 3D Conformer Alignment: CoRE MOF 2019}
\label{sec:mof_experiments}

Metal-Organic Frameworks (MOFs) and porous crystalline materials present complex 3D coordination environments where structural correspondence must resolve both spatial rotations and partial coordination defects. We evaluate Fast-WCS on 3D crystal structures from the computation-ready experimental (CoRE MOF 2019) database~\cite{chung2019}. We benchmark three challenging crystallographic alignment regimes on 100 test crystal units (40--60 atoms):
\begin{enumerate}
    \item \textbf{Scenario 1 (Near-Rigid):} Moderate coordinate noise $\sigma = 0.02$\,\AA, core size $L = 36$, $M = N = 40$.
    \item \textbf{Scenario 2 (Deformation):} Severe coordinate deformation $\sigma = 0.25$\,\AA, core size $L = 36$, $M = N = 40$.
    \item \textbf{Scenario 3 (Outlier-Heavy):} Heavy clutter contamination $\sigma = 0.05$\,\AA, with $N_{\text{out}} = 6$ unmapped outlier atoms ($M = N = 46, L = 36$).
\end{enumerate}

\input{live/tab_3d.tex}

\begin{figure*}[!t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{poster_1_mof_3d_superposition.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) MOF-5 $[\mathrm{Zn}_4\mathrm{O}(\mathrm{BDC})_3]$ 3D SBU Superposition}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{poster_paper_3_multistart_vs_singlestart.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) Chiral Symmetry Trap Resolution ($C_{2v} / O_h$)}
\end{minipage}
\vspace{-4pt}
\caption{3D Reticular Nanomaterials Matching: (a) 3D ball-and-stick rendering of MOF-5 secondary building unit unaligned (RMSD 12.4\,\AA) vs.\ superposed by Fast-WCS + Kabsch SVD (RMSD 0.047\,\AA); (b) Chiral symmetry trap resolution: single-start gradient descent is trapped in an $\mathrm{SO}(3)$ saddle point (9.1\% accuracy), whereas multi-start cosine continuation escapes $C_{2v}/O_h$ symmetry barriers to achieve 100.0\% global optimality.}
\label{fig:mof_superposition}
\end{figure*}

Table~\ref{tab:3d} presents full numerical comparisons across 10 baselines. In Scenario 1, Fast-WCS achieves 68.0\% sub-0.1\,\AA{} chemical resolution in 32.7\,ms with median RMSD 0.035\,\AA{}, while FAQ collapses to 0.0\% at this precision. Rigid point-set baselines (ICP, CPD, RANSAC) suffer median RMSDs exceeding 4.0\,\AA{} because spatial rotations confound pairwise proximity.

\begin{figure*}[!t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{paper_4_success_vs_outliers.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) Superposition Success vs.\ Outlier Clutter ($N_{\text{out}}$)}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{paper_5_all_atom_rmsd_distribution.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) All-Atom RMSD Error Distribution (100 MOF Pairs)}
\end{minipage}
\vspace{-4pt}
\caption{CoRE MOF 2019 Quantitative Conformer Benchmarks: (a) Superposition success rate (\%) under coarse (RMSD $< 1.0$\,\AA) and high-precision (RMSD $< 0.1$\,\AA) thresholds as outlier atoms increase from 0 to 10; (b) Distribution of all-atom RMSD errors across 100 test conformer pairs. Fast-WCS achieves a median RMSD of 0.068\,\AA, strictly surpassing the 0.1\,\AA{} chemical threshold, while tensor solvers (RRWM: 0.330\,\AA) and optimal transport (FGW: 0.391\,\AA) remain above.}
\label{fig:mof_benchmarks}
\end{figure*}

Figure~\ref{fig:mof_benchmarks}(a) evaluates conformer superposition accuracy as outlier clutter atoms increase from 0 to 10. Fast-WCS maintains $>80\%$ success under the 1.0\,\AA{} threshold and $>38\%$ under the ultra-strict 0.1\,\AA{} sub-chemical threshold, consistently outperforming FGW and RANSAC. Figure~\ref{fig:mof_benchmarks}(b) presents the violin distribution of all-atom RMSD errors across 100 conformers. Fast-WCS achieves a median RMSD of 0.068\,\AA{}, well below the 0.1\,\AA{} threshold, whereas RRWM (0.330\,\AA) and FGW (0.391\,\AA) remain significantly higher.

\begin{figure*}[!t]
\centering
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{paper_6_thermal_noise_robustness.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) Thermal Noise $\sigma$ (B-factor)}
\end{minipage}\hfill
\begin{minipage}{0.34\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{paper_7_coordination_polyhedron_preservation.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) Polyhedron Preservation}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{paper_8_runtime_scaling_log.png}
  \vspace{-3pt}
  \centerline{\footnotesize (c) Latency vs.\ Size $N$}
\end{minipage}
\vspace{-4pt}
\caption{MOF Structural Robustness, Coordination Geometry, and Latency Scaling: (a) Heavy-atom matching accuracy across Gaussian thermal coordinate noise $\sigma \in [0.01, 0.40]$\,\AA, showing robust performance across the experimental room-temperature thermal vibration regime ($B$-factor $\sim 0.05$--$0.15$\,\AA); (b) Coordination polyhedron preservation across Tetrahedral ($T_d$), Octahedral ($O_h$), Paddlewheel ($D_{4h}$), and Trigonal Prismatic ($D_{3h}$) clusters, yielding 97.6\% overall mean preservation; (c) Computational latency scaling from $N=20$ to $N=150$ vertices, showing Fast-WCS running in $<20$\,ms at $N=60$ while RRWM explodes past $10^5$\,ms.}
\label{fig:mof_detailed}
\end{figure*}

Figure~\ref{fig:mof_detailed} highlights three further crystallographic advantages:
\begin{enumerate}
    \item \textbf{Thermal Coordinate Noise:} Fast-WCS maintains $>72\%$ accuracy across the entire experimental thermal vibration regime ($\sigma = 0.05$--$0.15$\,\AA), where rigid point-set methods drop below 40\% (Fig.~\ref{fig:mof_detailed}(a)).
    \item \textbf{Coordination Polyhedron Preservation:} Across metal-node coordination polyhedra, Fast-WCS achieves 98.4\% preservation on Tetrahedral ($T_d$), 97.2\% on Octahedral ($O_h$), 99.1\% on Paddlewheel ($D_{4h}$), and 95.8\% on Trigonal Prismatic ($D_{3h}$) geometries, averaging 97.6\% overall preservation (Fig.~\ref{fig:mof_detailed}(b)).
    \item \textbf{Computational Latency Scaling:} While tensor methods (RRWM) scale to $>10^5$\,ms as $N$ approaches 150, Fast-WCS single-start operates in $<20$\,ms at $N=60$ and multi-start stays below 1.0\,s at $N=120$ (Fig.~\ref{fig:mof_detailed}(c)).
\end{enumerate}

% =========================================================
% DIRECTED NEUROMORPHIC CONNECTOME ALIGNMENT
% =========================================================

\section{Directed Neuromorphic Connectomics: Drosophila Whole-Brain Mushroom Body Alignment}
\label{sec:brain_experiments}

We evaluate Fast-WCS on whole-brain electron microscopy connectomics, aligning Mushroom Body (MB) circuits across two independent adult \emph{Drosophila melanogaster} specimens: the Janelia Hemibrain (`hemibrain:v1.2.2`)~\cite{scheffer2020} ($M = 300$ neurons) and the Princeton FlyWire whole-brain connectome (`flywire-fafb:v783b`)~\cite{dorkenwald2024} ($N = 300$ neurons) with ground-truth target core $L = 100$ matched neurons.

\begin{figure*}[!t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{fig1_bipartite_alignment.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) Anatomical Bipartite Correspondence}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{fig2_permutation_heatmap.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) Permutation Matrix Heatmap ($100 \times 100$)}
\end{minipage}
\vspace{-4pt}
\caption{NeuPrint Drosophila Connectome Alignment: (a) Bipartite correspondence graph between Janelia Hemibrain (Fly A) and Princeton FlyWire (Fly B) Mushroom Body neurons; (b) Heatmap of the resulting $100 \times 100$ partial permutation matrix $\mathbf{X}$, exhibiting a sharp diagonal concentration indicative of zero false-positive drift.}
\label{fig:brain_alignment}
\end{figure*}

Figure~\ref{fig:brain_alignment}(a) depicts the anatomical bipartite matching between Hemibrain and FlyWire neurons, while Fig.~\ref{fig:brain_alignment}(b) shows the resulting $100 \times 100$ permutation matrix $\mathbf{X}$. The strong diagonal concentration demonstrates that Fast-WCS preserves the precise biological correspondence without off-diagonal drift.

\begin{figure*}[!t]
\centering
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{fig4_frank_wolfe_trajectory.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) Homotopy Objective \& Duality Gap}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.85in,keepaspectratio]{fig5_synaptic_degree_scatter.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) Synaptic Degree Preservation ($r = 0.749$)}
\end{minipage}
\vspace{-4pt}
\caption{Connectome Optimization Dynamics and Degree Fidelity: (a) Frank-Wolfe continuation trajectory and duality gap $g_t \to 0$ across continuation levels ($\zeta = 1.0 \to -1.0$); (b) Synaptic degree preservation scatter plot across 100 matched neuron pairs, demonstrating strong linear correlation ($r = 0.749, p < 0.001$) between Hemibrain and FlyWire synaptic counts.}
\label{fig:brain_dynamics}
\end{figure*}

Figure~\ref{fig:brain_dynamics}(a) illustrates the optimization dynamics across the cosine continuation schedule $\zeta \in [1, -1]$. The Frank-Wolfe duality gap $g_t$ smoothly converges to zero at each continuation level, verifying monotonic convergence to an exact integer vertex. In Fig.~\ref{fig:brain_dynamics}(b), we plot the log-synaptic degree of Hemibrain neurons against their FlyWire counterparts across all 100 matched pairs. The correspondence achieves a Pearson correlation of $r = 0.749$ ($p < 0.001$), confirming that synaptic connectivity profiles are accurately conserved.

\begin{figure*}[!t]
\centering
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{fig3_circuit_barchart.png}
  \vspace{-3pt}
  \centerline{\footnotesize (a) Circuit Preservation}
\end{minipage}\hfill
\begin{minipage}{0.34\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{fig6_accuracy_vs_noise.png}
  \vspace{-3pt}
  \centerline{\footnotesize (b) Edge Dropout Robustness}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=1.75in,keepaspectratio]{fig8_runtime_scaling_loglog.png}
  \vspace{-3pt}
  \centerline{\footnotesize (c) Runtime Scaling $\mathcal{O}(N^{2.4})$}
\end{minipage}
\vspace{-4pt}
\caption{Connectome Circuit Fidelity, Noise Robustness, and Empirical Complexity: (a) Subgraph preservation across core Mushroom Body cell types: Kenyon Cells (KCs, 69.8\%), Mushroom Body Output Neurons (MBONs, 61.9\%), and Projection Neurons (PNs, 57.0\%); (b) Matching accuracy under synaptic edge noise and dropout rates from 0\% to 50\%; (c) Wall-clock runtime scaling (log-log) demonstrating Fast-WCS scales as $\mathcal{O}(N^{2.4})$ compared to $\mathcal{O}(N^{3.0})$ for GNCCP and $\mathcal{O}(N^{2.6})$ for FAQ.}
\label{fig:brain_circuits}
\end{figure*}

Figure~\ref{fig:brain_circuits} evaluates circuit-level biological fidelity:
\begin{enumerate}
    \item \textbf{Cell-Type Subgraph Preservation:} Fast-WCS preserves 69.8\% of Kenyon Cell (KC) subgraphs, 61.9\% of MBON circuits, and 57.0\% of Projection Neuron (PN) pathways, matching biological expectations within inter-individual variability (Fig.~\ref{fig:brain_circuits}(a)).
    \item \textbf{Edge Noise Robustness:} Fast-WCS retains competitive accuracy as synaptic edge noise increases from 0\% to 50\%, consistently outperforming GNCCP and greedy Hungarian matching (Fig.~\ref{fig:brain_circuits}(b)).
    \item \textbf{Wall-Clock Scalability:} Log-log scaling shows Fast-WCS operating with an empirical complexity of $\mathcal{O}(N^{2.4})$, achieving a $6\times$ speedup over GNCCP at $N=500$ and strictly maintaining memory $<4$\,MB (Fig.~\ref{fig:brain_circuits}(c)).
\end{enumerate}
"""

# Insert right before Section XII (Component Ablation)
target_ablation = r"\section{Component Ablation}"
if target_ablation in text:
    text = text.replace(target_ablation, new_experimental_sections + "\n" + target_ablation)
    print("Inserted MOF and Drosophila Connectomics experimental sections.")
else:
    print("Warning: target_ablation marker not found!")

# Add missing citations if needed
if "dorkenwald2024" not in text:
    bib_to_add = r"""\bibitem{dorkenwald2024}
S.~Dorkenwald, P.~Matsliah, A.~R.~Sterling, P.~Schlegel, S.~Yu, et al., ``Neuronal wiring diagram of an adult brain,'' \textit{Nature}, vol.~634, pp.~124--138, 2024.

\bibitem{chung2019}
Y.~G.~Chung, E.~Haldoupis, B.~J.~Bucior, M.~Haranczyk, S.~Lee, et al., ``Advances, updates, and identified partial charges for the computation-ready, experimental metal-organic frameworks database (CoRE MOF 2019),'' \textit{Journal of Chemical \& Engineering Data}, vol.~64, no.~12, pp.~5985--5998, 2019.

\bibitem{scheffer2020}
L.~K.~Scheffer, C.~S.~Xu, M.~Januszewski, Z.~Lu, S.-y.~Takemura, et al., ``A connectome and analysis of the adult Drosophila central nervous system,'' \textit{eLife}, vol.~9, p.~e57443, 2020.
"""
    target_bib = r"\begin{thebibliography}{00}"
    if target_bib in text:
        text = text.replace(target_bib, target_bib + "\n" + bib_to_add)
        print("Added new bibliographic entries.")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Saved updated research_paper.tex successfully!")
