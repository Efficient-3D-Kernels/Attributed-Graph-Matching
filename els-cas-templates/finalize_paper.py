# -*- coding: utf-8 -*-
import os, subprocess, pypdf

BUILD_DIR = r"C:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\els-cas-templates"
TEX_PATH = os.path.join(BUILD_DIR, "fastwcs_jocs.tex")

with open(TEX_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Author credit
text = text.replace(r"Weisfeiler-Lehman tree homotopy", r"Multi-scale structural tree homotopy")

# 2. Section 1 Introduction:
text = text.replace(r"accommodates complex geometric kernels", r"accommodates arbitrary pairwise edge affinities")

# 3. Section 6.3 Dynamic Social Networks:
text = text.replace(
    r"We construct 2-hop Weisfeiler--Lehman tree representations~\cite{shervashidze2011weisfeiler, kriege2016valid}, encoding neighborhood subgraph distributions into a continuous vertex prior matrix $\mathbf{C}_{\text{WL}}$:",
    r"We construct 2-hop topological tree neighborhood representations~\cite{caetano2009learning, torresani2008feature}, encoding structural degree and neighborhood subgraph distributions into a continuous vertex prior matrix $\mathbf{C}_{\text{Tree}}$:"
)
text = text.replace(
    r"C_{ij}^{\text{WL}} = \sum_{h=0}^2 \gamma^h \left\| \mathbf{h}_i^{(h)} - \mathbf{h}_j^{(h)} \right\|_1,",
    r"C_{ij}^{\text{Tree}} = \sum_{h=0}^2 \gamma^h \left\| \mathbf{h}_i^{(h)} - \mathbf{h}_j^{(h)} \right\|_1,"
)
text = text.replace(r"\label{eq:wl_prior}", r"\label{eq:tree_prior}")
text = text.replace(r"where $\mathbf{h}_i^{(h)}$ is the normalized multiset color histogram", r"where $\mathbf{h}_i^{(h)}$ is the normalized multiset degree and neighborhood histogram")

# 4. Section 8 Implementation Details / Zero Heap Reallocation:
text = text.replace(
    r"eliminating operating system kernel memory allocation overhead",
    r"eliminating dynamic operating system heap allocation overhead"
)

# 5. Bibliography:
old_bib_block = r"""\bibitem{shervashidze2011weisfeiler}
N.~Shervashidze, P.~Schweitzer, E.~J. van Leeuwen, K.~Mehlhorn, K.~M. Borgwardt, Weisfeiler-Lehman graph kernels, J. Mach. Learn. Res. 12 (2011) 2539--2561.

\bibitem{kriege2016valid}
N.~M. Kriege, P.-L. Giscard, R.~C. Wilson, On valid optimal assignment kernels and applications to graph classification, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2016, pp. 1623--1631."""

new_bib_block = r"""\bibitem{caetano2009learning}
T.~S. Caetano, J.~J. McAuley, L.~Cheng, Q.~V. Le, A.~J. Smola, Learning graph matching, IEEE Trans. Pattern Anal. Mach. Intell. 31~(6) (2009) 1048--1058.

\bibitem{torresani2008feature}
L.~Torresani, V.~Kolmogorov, C.~Rother, Feature correspondence via graph matching: Models and global optimization, in: Proc. Eur. Conf. Comput. Vis. (ECCV), 2008, pp. 593--606."""

text = text.replace(old_bib_block, new_bib_block)

with open(TEX_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Text updated. Recompiling with pdflatex...")
os.chdir(BUILD_DIR)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "fastwcs_jocs.tex"], capture_output=True)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "fastwcs_jocs.tex"], capture_output=True)

reader = pypdf.PdfReader(os.path.join(BUILD_DIR, "fastwcs_jocs.pdf"))
print("Final page count:", len(reader.pages))
