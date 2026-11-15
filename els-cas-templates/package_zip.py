# -*- coding: utf-8 -*-
import os, shutil, zipfile, fitz, re

BUILD_DIR = r"C:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\els-cas-templates"
ARTIFACT_DIR = r"C:\Users\abdus\.gemini\antigravity\brain\dacd62f2-a740-4c03-a5f6-542b13372520"

# 1. Copy PDF to artifact dir
pdf_src = os.path.join(BUILD_DIR, "fastwcs_jocs.pdf")
pdf_dst = os.path.join(ARTIFACT_DIR, "fastwcs_jocs.pdf")
shutil.copy2(pdf_src, pdf_dst)
print("Copied PDF to artifact dir:", os.path.getsize(pdf_dst), "bytes")

# 2. Render all 18 pages to PNG
doc = fitz.open(pdf_src)
print(f"Total pages in PDF: {len(doc)}")
out_dirs = [
    os.path.join(BUILD_DIR, "rendered_final_18_pages"),
    os.path.join(ARTIFACT_DIR, "rendered_final_18_pages")
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    fname = f"page_{i+1:02d}.png"
    for d in out_dirs:
        pix.save(os.path.join(d, fname))
print("Rendered all 18 pages to PNGs in both locations.")

# 3. Create complete self-contained ZIP archive
zip_path_local = os.path.join(BUILD_DIR, "fastwcs_jocs_complete_paper.zip")
zip_path_artifact = os.path.join(ARTIFACT_DIR, "fastwcs_jocs_complete_paper.zip")

readme_content = """# Fast-WCS Journal of Computational Science (JoCS) Submission Package

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
"""

readme_path = os.path.join(BUILD_DIR, "README_OVERLEAF.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

included_files = [
    "fastwcs_jocs.tex",
    "fastwcs_jocs.pdf",
    "cas-dc.cls",
    "cas-common.sty",
    "elsarticle-num.bst",
    "cas-refs.bib",
    "README_OVERLEAF.md"
]

with open(os.path.join(BUILD_DIR, "fastwcs_jocs.tex"), "r", encoding="utf-8") as f:
    tex_text = f.read()

figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text))

for target_zip in [zip_path_local, zip_path_artifact]:
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in included_files:
            fp = os.path.join(BUILD_DIR, f)
            if os.path.exists(fp):
                zf.write(fp, arcname=f)
        for fig in figs:
            for sub in ["", "figures", "figs", "figures/mof", "figures/brain", "figures/standalone"]:
                p = os.path.join(BUILD_DIR, sub, fig)
                if os.path.exists(p):
                    rel = os.path.relpath(p, BUILD_DIR)
                    zf.write(p, arcname=rel)
                    break

print(f"Created ZIP: {zip_path_local} ({os.path.getsize(zip_path_local):,} bytes)")
print(f"Created ZIP: {zip_path_artifact} ({os.path.getsize(zip_path_artifact):,} bytes)")
