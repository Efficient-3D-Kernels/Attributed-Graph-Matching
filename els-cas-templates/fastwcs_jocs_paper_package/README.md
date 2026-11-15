# Fast-WCS Elsevier JoCS Manuscript Package

**Manuscript Title:** Fast-WCS: An Analytical Homotopy Framework for Scalable Attributed Graph Matching Across Physical, Visual, and Topological Domains  
**Submission Target:** Journal of Computational Science (Elsevier)  
**Document Class:** Elsevier CAS Double Column (cas-dc.cls)  
**Page Count:** Exactly 18 Pages  

---

## 1. Directory Structure

- astwcs_jocs.tex: Complete LaTeX source file (18 pages).
- astwcs_jocs.pdf: Compiled production PDF.
- astwcs_refs.bib: Complete BibTeX reference database.
- cas-dc.cls: Elsevier CAS Double-Column LaTeX document class.
- cas-common.sty: Elsevier CAS common style package.
- cas-model2-names.bst: Elsevier author-year/numbered BibTeX style.
- stfloats.sty: Two-column bottom float handler.
- igures/: Complete directory of publication figures (standalone vector PDFs, PNGs, MOF, connectomics).

---

## 2. Compilation Instructions

### Local Command Line (pdfLaTeX):
To compile locally from terminal or PowerShell:
`ash
pdflatex fastwcs_jocs.tex
pdflatex fastwcs_jocs.tex
`
*(Two passes ensure all cross-references, section labels, and citations resolve with exact Page X of 18 footers).*

### Overleaf / arXiv / Elsevier Editorial Manager:
1. Upload this zip file directly or extract its contents into an Overleaf project.
2. Set the main document to astwcs_jocs.tex.
3. Set the compiler to pdfLaTeX with TeX Live 2024 or later.
