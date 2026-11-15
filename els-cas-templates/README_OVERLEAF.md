# Fast-WCS Journal of Computational Science (JoCS) Submission Package

This self-contained archive contains the complete manuscript, figures, style classes, and compiled PDF for:
**"Fast-WCS: An Analytical Homotopy Framework for Scalable Attributed Graph Matching Across Physical, Visual, and Topological Domains"**

## Files Included:
- `fastwcs_jocs.tex`: Complete LaTeX manuscript (Elsevier CAS double-column format).
- `fastwcs_jocs.pdf`: Compiled 18-page manuscript.
- `cas-dc.cls`: Elsevier CAS double-column LaTeX document class.
- `cas-common.sty`: Elsevier CAS common style package.
- `elsarticle-num.bst`: Standard numbered bibliography style.
- `cas-refs.bib`: BibTeX bibliography database.
- `figures/`: Vector graphics (PDF) and bitmap renderings (PNG) referenced in the paper.
- `figs/`: Additional supplementary figure components referenced in the paper.

## How to Compile Locally or on Overleaf:
1. **Local (pdflatex)**:
   ```bash
   pdflatex fastwcs_jocs.tex
   pdflatex fastwcs_jocs.tex
   ```
2. **Overleaf**:
   Upload this ZIP directly to Overleaf (New Project -> Upload Project). It compiles out of the box with `pdfLaTeX` set as the compiler.
