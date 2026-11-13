# Fast-WCS Design System & Frontend Style Specification

> **Final Year Project (FYP)**  
> **Project Title:** Fast-WCS: Fast Weighted Common Subgraph Matching via Level-3 BLAS Frank-Wolfe Optimization and Closed-Form SVD Superposition  
> **Department:** Department of Computer Science & Engineering, Amrita Vishwa Vidyapeetham, Coimbatore  
> **Faculty Guide:** Dr. T Ramraj, Assistant Professor (Sl. Gd)  
> **Design Architecture:** Academic Editorial Parchment Theme & Dark Terminal Workbench Design System

---

## Table of Contents
1. [Design Philosophy & Aesthetic Direction](#1-design-philosophy--aesthetic-direction)
2. [Exact Color Palette & Design Tokens](#2-exact-color-palette--design-tokens)
   - [2.1 Core Academic Light Palette (CSS Variables)](#21-core-academic-light-palette-css-variables)
   - [2.2 Terminal Console & Developer Workbench (Dark Mode)](#22-terminal-console--developer-workbench-dark-mode)
   - [2.3 Status, Verification & Semantic Badges](#23-status-verification--semantic-badges)
   - [2.4 WebGL 3D Molecular Viewer & CPK Palette (Three.js)](#24-webgl-3d-molecular-viewer--cpk-palette-threejs)
   - [2.5 Chart.js Multi-Algorithm Benchmark Palette](#25-chartjs-multi-algorithm-benchmark-palette)
   - [2.6 Digital Poster Presentation Palette](#26-digital-poster-presentation-palette)
3. [Typography & Font Hierarchies](#3-typography--font-hierarchies)
   - [3.1 Font Families & Google Fonts Imports](#31-font-families--google-fonts-imports)
   - [3.2 Typographic Hierarchy & Scale](#32-typographic-hierarchy--scale)
4. [Layout Architecture & Grid Systems](#4-layout-architecture--grid-systems)
   - [4.1 Global Layout Shells](#41-global-layout-shells)
   - [4.2 Grid Structures & Alignments](#42-grid-structures--alignments)
   - [4.3 Responsive Breakpoints & Media Queries](#43-responsive-breakpoints--media-queries)
5. [Complete CSS Component Library](#5-complete-css-component-library)
   - [5.1 Navigation Bar & Header](#51-navigation-bar--header)
   - [5.2 Hero & Domain Highlight Band](#52-hero--domain-highlight-band)
   - [5.3 Interactive Buttons & Controls](#53-interactive-buttons--controls)
   - [5.4 Live Terminal Window & Execution Stream](#54-live-terminal-window--execution-stream)
   - [5.5 Visualizer Viewports & Floating Overlays](#55-visualizer-viewports--floating-overlays)
   - [5.6 Interactive Range Sliders & Presets](#56-interactive-range-sliders--presets)
   - [5.7 Benchmark Tables & Stat Cards](#57-benchmark-tables--stat-cards)
   - [5.8 Mathematical & Theoretical Proof Boxes](#58-mathematical--theoretical-proof-boxes)
   - [5.9 Keyframe Animations & Micro-Interactions](#59-keyframe-animations--micro-interactions)
6. [External Libraries, CDNs & Rendering Engines](#6-external-libraries-cdns--rendering-engines)
7. [Directory Structure of Web Assets](#7-directory-structure-of-web-assets)

---

## 1. Design Philosophy & Aesthetic Direction

The user interface for the **Fast-WCS** Final Year Research Platform was designed with a dual-nature aesthetic:

1. **Scholarly Academic Editorial (Light Surface):**
   - Inspired by the timeless elegance of university presses (Cambridge/Oxford), mathematical journals (*Annals of Mathematics*, *Nature Machine Intelligence*), and Edward Tufte's information design principles.
   - Employs warm, organic paper backgrounds (`#F3F1EB`, `#FBFAF7`) rather than sterile, harsh `#FFFFFF` to minimize eye fatigue during extended reading of complex mathematical proofs.
   - Uses Amrita Vishwa Vidyapeetham's signature institutional burgundy maroon (`#8E1638`) as the authoritative primary brand color, supported by warm charcoal ink (`#1B1917`) and slate accents (`#3D4B5C`).

2. **High-Performance Research Workbench (Dark Console Surface):**
   - The interactive terminal consoles and code execution engines use a high-contrast dark GitHub/Monokai-inspired palette (`#0d1117`, `#161b22`, `#30363d`).
   - Gives faculty, reviewers, and evaluators the sensation of observing a live high-performance computing (HPC) Level-3 BLAS engine running directly within the browser with microsecond latency metrics and duality gap telemetry.

---

## 2. Exact Color Palette & Design Tokens

### 2.1 Core Academic Light Palette (CSS Variables)

All primary colors are declared as CSS custom properties in the `:root` scope of [`style.css`](style.css):

| CSS Variable | Hex Code | RGB | Sample | Semantic Role / Description |
| :--- | :--- | :--- | :---: | :--- |
| `--paper` | `#F3F1EB` | `rgb(243, 241, 235)` | ![#F3F1EB](https://via.placeholder.com/18/F3F1EB/F3F1EB.png) | **Canvas / Body Background:** Warm parchment tone used for the entire document background. |
| `--paper-raised` | `#FBFAF7` | `rgb(251, 250, 247)` | ![#FBFAF7](https://via.placeholder.com/18/FBFAF7/FBFAF7.png) | **Elevated Surfaces:** Sticky navigation bar, cards, modals, table bodies, and interactive control wrappers. |
| `--ink` | `#1B1917` | `rgb(27, 25, 23)` | ![#1B1917](https://via.placeholder.com/18/1B1917/1B1917.png) | **Primary Text & Headings:** Deep warm charcoal black for high-readability body copy, `<h1>`–`<h3>`, and prominent labels. |
| `--ink-soft` | `#4B4742` | `rgb(75, 71, 66)` | ![#4B4742](https://via.placeholder.com/18/4B4742/4B4742.png) | **Secondary Text / Subtitles:** Neutral charcoal for descriptions, figure captions, table headers, and metadata. |
| `--maroon` | `#8E1638` | `rgb(142, 22, 56)` | ![#8E1638](https://via.placeholder.com/18/8E1638/8E1638.png) | **Institutional Primary Accent:** Amrita university maroon; primary CTA buttons, active links, metric numerals, borders. |
| `--maroon-dark` | `#6E0F2B` | `rgb(110, 15, 43)` | ![#6E0F2B](https://via.placeholder.com/18/6E0F2B/6E0F2B.png) | **Interactive Hover / Pressed:** Button active state, deep drop-shadows, high-contrast hover borders. |
| `--maroon-wash` | `#F3E3E7` | `rgb(243, 227, 231)` | ![#F3E3E7](https://via.placeholder.com/18/F3E3E7/F3F3E7.png) | **Accent Wash / Tint:** Subtle pinkish maroon tint used for tab hover states, chip badges, and guide callouts. |
| `--slate` | `#3D4B5C` | `rgb(61, 75, 92)` | ![#3D4B5C](https://via.placeholder.com/18/3D4B5C/3D4B5C.png) | **Cool Editorial Accent:** Used for mathematical formula chips, secondary neutral highlights, and metadata tags. |
| `--slate-wash` | `#E7EBEE` | `rgb(231, 235, 238)` | ![#E7EBEE](https://via.placeholder.com/18/E7EBEE/E7EBEE.png) | **Slate Tint:** Background for inline formulas, theorem metadata, and secondary status badges. |
| `--line` | `#DEDAD1` | `rgb(222, 218, 209)` | ![#DEDAD1](https://via.placeholder.com/18/DEDAD1/DEDAD1.png) | **Borders & Dividers:** Subtle, organic hairline divider rule between sections, table cells, and card outlines. |

```css
:root {
  --paper:        #F3F1EB;
  --paper-raised: #FBFAF7;
  --ink:          #1B1917;
  --ink-soft:     #4B4742;
  --maroon:       #8E1638;
  --maroon-dark:  #6E0F2B;
  --maroon-wash:  #F3E3E7;
  --slate:        #3D4B5C;
  --slate-wash:   #E7EBEE;
  --line:         #DEDAD1;
  --serif:        'Source Serif 4', Georgia, serif;
  --sans:         'IBM Plex Sans', -apple-system, sans-serif;
}
```

---

### 2.2 Terminal Console & Developer Workbench (Dark Mode)

The interactive Python Frank-Wolfe stdout terminal in [`demo.html`](demo.html) and [`mapping.html`](mapping.html) utilizes an engineered high-contrast dark theme:

| Element / Class | Hex / RGBA Code | Description |
| :--- | :--- | :--- |
| **Terminal Frame** (`.terminal-window`, `.terminal-body`) | `#0d1117` | Deep obsidian dark background matching GitHub terminal standards. |
| **Terminal Chrome** (`.terminal-header`, `.terminal-footer`) | `#161b22` | Header and footer toolbars with 1px top/bottom stroke. |
| **Terminal Border** (`border`) | `#30363d` | Dark grey border separating chrome from editor body. |
| **Default Terminal Text** (`.terminal-body`) | `#c9d1d9` | Light slate grey monospace text with high legibility. |
| **Window Control: Close** (`.dot-red`) | `#ff5f56` | macOS-style red circular control dot (12×12px). |
| **Window Control: Minimize** (`.dot-yellow`) | `#ffbd2e` | macOS-style yellow circular control dot (12×12px). |
| **Window Control: Maximize** (`.dot-green`) | `#27c93f` | macOS-style green circular control dot (12×12px). |
| **Command Tag / Blue Accent** (`.tag-cmd`, `.term-text.color-cmd`) | `#1f6feb` / `#58a6ff` | Command execution identifier and active bash commands. |
| **Initialization Tag** (`.tag-init`) | bg: `#388bfd33` (`rgba(56,139,253,0.2)`), border: `#388bfd66`, text: `#58a6ff` | System startup and matrix allocation tags. |
| **Lemma Tag** (`.tag-lemma`) | bg: `#a371f733` (`rgba(163,113,247,0.2)`), border: `#a371f766`, text: `#d2a8ff` | Rank-1 outer-product factorization messages (Lemma 1). |
| **Theorem Tag** (`.tag-thm`) | bg: `#d2992233` (`rgba(210,153,34,0.2)`), border: `#d2992266`, text: `#f0883e` | Exact analytical gradient and quartic line search (Theorems 1 & 3). |
| **Step Tag** (`.tag-step`) | bg: `#21262d`, text: `#8b949e` | Iteration step counters and Frank-Wolfe step indicators. |
| **Success Tag** (`.tag-success`, `.color-success`) | bg: `#23863633` (`rgba(35,134,54,0.2)`), border: `#23863666`, text: `#7ee787` | Convergence notifications, Hungarian assignment completions. |
| **Highlight Tag** (`.tag-highlight`, `.color-highlight`) | bg: `#f0883e33` (`rgba(240,136,62,0.2)`), border: `#f0883e66`, text: `#ffa657` | Microsecond timing telemetry and duality certificates. |
| **Error Tag** (`.tag-error`, `.color-error`) | bg: `#da363333` (`rgba(218,54,51,0.2)`), border: `#da363366`, text: `#ff7b72` | Fallback exception notices and syntax error messages. |
| **Terminal Cursor** (`.terminal-cursor`) | `#58a6ff` | 8×15px pulsing electric blue blinking caret. |
| **Status Badge: Optimizing** (`.terminal-status-badge.optimizing`) | `#d29922` | Animated pulsing amber badge during active solver iterations. |
| **Status Badge: Ready** (`.terminal-status-badge`) | `#238636` | Solid emerald green pill badge when solver is ready/converged. |

---

### 2.3 Status, Verification & Semantic Badges

Used in benchmark comparison tables, status checklists, and metric ribbons across all pages:

| Badge Type | Background | Text Color | Border Color | Example Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Verified / Functional** | `#dcfce7` (`#E4EFE6`) | `#166534` (`#2C6B3A`) | `#bbf7d0` | 100% verified implementation checklists in `progress.html` |
| **Speedup Highlight** | `#dcfce7` | `#166534` | `#bbf7d0` | `22.8× Speedup` badges in benchmark tables |
| **Matched Node Pill** | `#dcfce7` | `#166534` | `#bbf7d0` | Matched correspondence coordinate pills (`.matched-glow`) |
| **In Progress / Warning** | `#F5EAD6` | `#8A5A16` | Transparent | Ongoing validation tasks |
| **Gold / SOTA Certificate** | `#fef3c7` | `#92400e` | `#fde68a` | Gold award badge for highest accuracy solver |
| **Integrated / Modular** | `#E7EBEE` | `#3D4B5C` | Transparent | Architecture component verification tag |
| **Algorithm Parameter Pill** | `#F3E3E7` | `#8E1638` | `#DEDAD1` | Numerical parameter tags in solver controls |
| **Ours Benchmark Row** | `#fdf2f4` | `#1B1917` (bold) | `#8E1638` (2px top/bottom) | Highlights Fast-WCS row in 7-algorithm shootout |
| **Ours Distinction Tag** | `#8E1638` | `#FFFFFF` | None | Small uppercase `[OURS]` indicator pill |

---

### 2.4 WebGL 3D Molecular Viewer & CPK Palette (Three.js)

In [`mapping.html`](mapping.html), the interactive 3D WebGL molecular viewer renders chemical structures, spatial rotation vectors, and Kabsch SVD alignments using the following color specifications:

#### Viewport Styling
- **Canvas Container Background:** `linear-gradient(135deg, #FAF8F5 0%, #EFEBE3 100%)`
- **Container Border:** `1px solid #DEDAD1` (border-radius: `6px`)
- **Hover/Interaction Cursor:** `grab` (switches to `grabbing` on mouse down)

#### Elemental CPK Color Scheme (3D Mesh Materials)
| Chemical Element | Symbol | Three.js Hex | Web Hex Code | Visual Tone |
| :--- | :---: | :---: | :---: | :--- |
| **Carbon** | `C` | `0x475569` | `#475569` | Deep Slate Grey |
| **Oxygen** | `O` | `0xd97706` | `#d97706` | Warm Saturated Amber |
| **Nitrogen** | `N` | `0x0284c7` | `#0284c7` | Deep Electric Sky Blue |
| **Sulfur** | `S` | `0xa16207` | `#a16207` | Golden Ochre Brown |
| **Zirconium** | `Zr` | `0x10b981` | `#10b981` | Vivid Emerald Green |
| **Zinc** | `Zn` | `0x14b8a6` | `#14b8a6` | Bright Cyan Teal |
| **Copper** | `Cu` | `0xf59e0b` | `#f59e0b` | Metallic Bronze Amber |

#### Molecular Mesh & Alignment Vectors
- **Molecule 1 (Reference) Matched Atoms:** `0x16a34a` (`#16a34a` — Dark Forest Green)
- **Molecule 2 (Target) Matched Atoms:** `0x22c55e` (`#22c55e` — Bright Spring Lime Green)
- **Molecule 1 Bonds (Cylinders, radius 0.08):** `0x94a3b8` (`#94a3b8` — Muted Slate)
- **Molecule 2 Bonds (Cylinders, radius 0.08):** `0xcbd5e1` (`#cbd5e1` — Pale Slate Grey)
- **Correspondence Vectors (Dashed Lines):** `0xd97706` (`#d97706` — Warm Amber, dashSize: 0.35, gapSize: 0.2, linewidth: 2.2)
- **Three.js Lighting:**
  - Ambient Light: `0xffffff`, intensity `0.9`
  - Key Directional Light: `0xffffff`, intensity `0.85`
  - Backlight / Fill: `0xedd8c0` (warm parchment fill), intensity `0.5`

---

### 2.5 Chart.js Multi-Algorithm Benchmark Palette

The dynamic telemetry curves and parameter sweeps (Accuracy vs. Noise $\sigma$, Outliers $N_{\mathrm{out}}$, Problem Size $M$, and Density $\rho$) in [`mapping.html`](mapping.html) and [`demo.html`](demo.html) follow this strict algorithm color coding:

| Algorithm / Model | Line / Border Color | Area Fill Color | Stroke Style |
| :--- | :--- | :--- | :--- |
| **Fast-WCS (Ours)** | `#8E1638` (Amrita Maroon) | `rgba(142, 22, 56, 0.08)` (Soft Maroon) | Solid line (3.2px width, pointRadius: 4.5) |
| **RLX1 (Yang et al. 2018)** | `#C2410C` (Rust Terracotta) | `transparent` | Dashed line `[4, 3]` (1.8px width) |
| **RLX1F (Fast Truncation)** | `#D97706` (Amber Orange) | `transparent` | Dashed line `[4, 3]` (1.8px width) |
| **RLX2 (Yang et al. 2018)** | `#DC2626` (Bright Red) | `transparent` | Dashed line `[4, 3]` (1.8px width) |
| **PGM (Projected Gradient)** | `#2563EB` (Royal Blue) | `transparent` | Solid line (1.8px width) |
| **GA (Graduated Assignment)** | `#0D9488` (Teal Green) | `transparent` | Solid line (1.8px width) |
| **SM (Spectral Matching)** | `#7C3AED` (Deep Violet) | `transparent` | Solid line (1.8px width) |
| **Standalone WL-OA** | `#238636` (Forest Green) | `transparent` | Solid line (2.0px width) |
| **Classic GNCCP (Yang 2018)**| `#1f6feb` (GitHub Blue) | `transparent` | Solid line (1.8px width) |
| **FAQ (Vogelstein 2015)** | `#8b949e` (Slate Grey) | `transparent` | Dashed line `[5, 5]` (1.5px width) |
| **RRWM (Lawler Tensor)** | `#238636` (Forest Green) | `transparent` | Solid line (1.5px width) |
| **Speedup Multiplier** | `#0284c7` (Sky Blue) | `transparent` | Solid line (2.5px width) |

---

### 2.6 Digital Poster Presentation Palette

In [`../../poster_presentation.html`](../poster_presentation.html), the digital poster interface utilizes an alternate presentation layout optimized for 4K projector displays and print exports:

```css
:root {
  --bg-page:         #edf2f7;  /* High-contrast neutral presentation background */
  --banner-teal:     #143545;  /* Deep midnight teal banner background */
  --pill-navy:       #1b3d54;  /* Deep navy navigation pill background */
  --card-bg:         #ffffff;  /* Pure white poster panel cards */
  --card-border:     #cbd5e0;  /* Subtle grey card borders */
  --text-primary:    #1a202c;  /* Midnight black high-legibility body */
  --text-secondary:  #4a5568;  /* Secondary slate text */
  --accent-gold:     #f6ad55;  /* Warm gold highlights & supervisor tags (#fbd38d) */
  --accent-green:    #38a169;  /* Emerald print button & verified badges */
  --accent-blue:     #3182ce;  /* Active button selector blue */
}
```

- **Amrita Emblem & Header Burgundy:** `#a31d24` (Official University Burgundy Red)
- **Top Control Bar:** `#1a202c` with shadow `0 8px 24px rgba(0,0,0,0.3)`

---

## 3. Typography & Font Hierarchies

### 3.1 Font Families & Google Fonts Imports

The typography marries a literary serif with an engineering sans-serif and a developer monospace:

```css
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
```

1. **`'Source Serif 4', Georgia, serif` (Variable: `--serif`):**
   - Designed by Frank Grießhammer (Adobe Type).
   - Applied to: Document headers (`h1`, `h2`), hero titles, abstract paragraphs, theorem statements, and project descriptions.
   - Characterized by sturdy serifs, warm proportions, and optical sizing (`opsz 8..60`) optimized for high-density academic reading.

2. **`'IBM Plex Sans', -apple-system, sans-serif` (Variable: `--sans`):**
   - Designed by Mike Abbink (IBM).
   - Applied to: Sticky header navigation, buttons, eyebrow subtitles, data tables, parameter labels, metric counters, chart tooltips, and card descriptions.
   - Characterized by neutral, geometric clarity with an engineered, industrial aesthetic.

3. **`'JetBrains Mono', 'Fira Code', 'Consolas', monospace`:**
   - Applied to: Live terminal stdout, code snippets, mathematical notation badges, Hungarian matrix indices, and runtime metrics.
   - Features programming ligatures, distinct zero characters, and high legibility at 11–13px.

---

### 3.2 Typographic Hierarchy & Scale

| Style / Element | Font Family | Size | Weight | Line Height | Letter Spacing | Color |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Hero Title (`.hero h1`)** | `'Source Serif 4'` | `2.9rem` (46px) | 600 | 1.15 | Normal | `var(--ink)` (`#1B1917`) |
| **Page Title (`h1`)** | `'Source Serif 4'` | `2.5rem` (40px) | 600 | 1.15 | Normal | `var(--ink)` (`#1B1917`) |
| **Section Header (`h2`)** | `'Source Serif 4'` | `1.6rem` (25.6px) | 600 | 1.25 | Normal | `var(--ink)` (`#1B1917`) |
| **Subsection Header (`h3`)**| `'IBM Plex Sans'` | `1.0rem` (16px) | 600 | 1.3 | Normal | `var(--ink)` (`#1B1917`) |
| **Eyebrow Header (`.eyebrow`)**| `'IBM Plex Sans'` | `13.5px` | 500 | Normal | `0.02em` | `var(--maroon)` (`#8E1638`) |
| **Abstract (`.abstract`)** | `'Source Serif 4'` | `1.08rem` (17.3px)| 400 | 1.6 | Normal | `var(--ink-soft)` (`#4B4742`)|
| **Body Copy (`p`, `li`)** | `'Source Serif 4'` | `17px` | 400 | 1.6 | Normal | `var(--ink)` (`#1B1917`) |
| **Subtitle (`.subtitle`)** | `'Source Serif 4'` | `1.15rem` (18.4px)| 400 (italic)| 1.4 | Normal | `var(--ink-soft)` (`#4B4742`)|
| **Big Metric Counter (`.stat`)**| `'IBM Plex Sans'` | `1.7rem` (27.2px)| 600–700 | 1.2 | Normal | `var(--maroon)` (`#8E1638`) |
| **Metric Label (`.label`)** | `'IBM Plex Sans'` | `14px` | 400 | 1.3 | Normal | `var(--ink-soft)` (`#4B4742`)|
| **Navigation Links** | `'IBM Plex Sans'` | `14.5px` | 400–500 | Normal | Normal | `var(--ink-soft)` → `var(--ink)` |
| **Terminal Output** | `'JetBrains Mono'` | `13px` | 400 | 1.55 | Normal | `#c9d1d9` |
| **Terminal Tag Badges** | `'JetBrains Mono'` | `11.5px` | 600 | 1.0 | `0.02em` | Specialized tag colors |
| **Table Headings (`th`)** | `'IBM Plex Sans'` | `13px` | 600 | Normal | `0.03em` (uppercase)| `var(--ink)` |
| **Table Cells (`td`)** | `'IBM Plex Sans'` | `14.5px` | 400 | Normal | Tabular numbers | `var(--ink)` |
| **Mathematical Formulas** | KaTeX / Monospace | `14–15.5px` | 400 | 1.4 | Normal | `var(--slate)` / `#1B1917` |

---

## 4. Layout Architecture & Grid Systems

### 4.1 Global Layout Shells

The layout is constrained to standard desktop reading widths to preserve typographic line-length readability (65–75 characters per line):

```css
/* Layout Shells */
.wrap {
  max-width: 1080px;
  margin: 0 auto;
  padding: 0 32px;
}

.prose {
  max-width: 680px;
  line-height: 1.6;
}

main {
  padding-bottom: 100px;
}
```

---

### 4.2 Grid Structures & Alignments

#### 1. Domain Performance Metric Band (`.domain-band`)
A 3-column metric ribbon displaying benchmark records:
```css
.domain-band {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  margin: 48px 0 64px;
}
.domain-band div {
  padding: 28px 28px 28px 0;
  border-right: 1px solid var(--line);
}
.domain-band div:last-child {
  border-right: none;
  padding-right: 0;
}
```

#### 2. Section Navigation Cards (`.section-links`)
A 2-column card grid with an inverted 1px border gap:
```css
.section-links {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
}
.section-links a {
  background: var(--paper-raised);
  padding: 26px 28px;
  text-decoration: none;
}
.section-links a:hover {
  background: var(--maroon-wash);
}
```

#### 3. Top Workbench Split (`.top-grid` & `.workbench-grid`)
Splits the 3D WebGL viewport / Terminal stdout and the dynamic telemetry curves:
```css
.top-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.workbench-grid {
  display: grid;
  grid-template-columns: 1.25fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}
```

#### 4. Team Roster Grid (`.team-grid`)
```css
.team-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  margin-bottom: 40px;
}
```

#### 5. Methodological Pipeline Stage (`.stage`)
A 2-column timeline layout displaying step numbers alongside theoretical formulas:
```css
.stage {
  display: grid;
  grid-template-columns: 56px 1fr;
  gap: 24px;
  padding: 32px 0;
  border-top: 1px solid var(--line);
}
```

---

### 4.3 Responsive Breakpoints & Media Queries

The site is fully responsive across three primary viewport tiers:

```css
/* Tablet & Small Desktop Breakpoint (<= 900px) */
@media (max-width: 900px) {
  .top-grid { grid-template-columns: 1fr; }
  .workbench-grid { grid-template-columns: 1fr; }
  .stats-ribbon { grid-template-columns: repeat(2, 1fr); }
  .math-cards { grid-template-columns: 1fr; }
}

/* Mobile Devices (<= 720px) */
@media (max-width: 720px) {
  h1 { font-size: 2rem; }
  .hero h1 { font-size: 2.1rem; }
  .domain-band { grid-template-columns: 1fr; }
  .domain-band div {
    border-right: none;
    border-bottom: 1px solid var(--line);
    padding: 20px 0;
  }
  .domain-band div:last-child { border-bottom: none; }
  .section-links { grid-template-columns: 1fr; }
  .team-grid { grid-template-columns: 1fr; }
  .video-grid { grid-template-columns: 1fr; }
  .stage { grid-template-columns: 1fr; }
  .nav-links { gap: 16px; font-size: 13.5px; }
  .result-stats { gap: 24px; }
}

/* Accessibility: Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 5. Complete CSS Component Library

### 5.1 Navigation Bar & Header

The top navigation header remains sticky on scroll with a subtle border and 1080px inner alignment:

```css
.site-header {
  border-bottom: 1px solid var(--line);
  background: var(--paper-raised);
  position: sticky;
  top: 0;
  z-index: 10;
}
.nav-inner {
  max-width: 1080px;
  margin: 0 auto;
  padding: 18px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}
.brand {
  font-family: var(--sans);
  font-weight: 600;
  font-size: 15px;
  letter-spacing: 0.01em;
  text-decoration: none;
  color: var(--ink);
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.brand small {
  font-weight: 400;
  color: var(--ink-soft);
  font-size: 13px;
}
.nav-links {
  display: flex;
  gap: 28px;
  font-family: var(--sans);
  font-size: 14.5px;
  list-style: none;
  margin: 0;
  padding: 0;
}
.nav-links a {
  text-decoration: none;
  color: var(--ink-soft);
  padding-bottom: 4px;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}
.nav-links a:hover,
.nav-links a[aria-current="page"] {
  color: var(--ink);
  border-bottom-color: var(--maroon);
}
.gh-btn {
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  color: var(--ink);
  border: 1px solid var(--ink);
  padding: 8px 16px;
  border-radius: 3px;
  white-space: nowrap;
  transition: all 0.15s ease;
}
.gh-btn:hover {
  background: var(--ink);
  color: var(--paper-raised);
}
```

---

### 5.2 Hero & Domain Highlight Band

```css
.hero {
  padding: 56px 0 0;
}
.hero .eyebrow {
  margin-bottom: 18px;
}
.hero h1 {
  font-size: 2.9rem;
  max-width: 820px;
}
.hero .abstract {
  max-width: 680px;
  color: var(--ink-soft);
  font-size: 1.08rem;
}
.hero-meta {
  font-family: var(--sans);
  font-size: 14px;
  color: var(--ink-soft);
  margin-top: 8px;
}
```

---

### 5.3 Interactive Buttons & Controls

#### Primary Action Button (`.btn-action`, `.run-main-btn`)
Used to trigger live Frank-Wolfe optimization and benchmark suites:
```css
.btn-action, .run-main-btn {
  background: var(--maroon);
  color: #FFFFFF;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
  display: flex;
  align-items: center;
  gap: 8px;
}
.btn-action:hover, .run-main-btn:hover {
  background: var(--maroon-dark);
}
.btn-action:active, .run-main-btn:active {
  transform: translateY(1px);
}
.run-main-btn:disabled {
  background: var(--ink-soft);
  cursor: not-allowed;
  opacity: 0.6;
}
```

#### Secondary Control Button (`.btn-secondary`, `.run-sub-btn`)
Used for parameter resets and data refreshes:
```css
.btn-secondary, .run-sub-btn {
  background: var(--paper-raised);
  color: var(--ink);
  border: 1px solid var(--line);
  padding: 9px 16px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-secondary:hover, .run-sub-btn:hover {
  background: var(--maroon-wash);
  color: var(--maroon);
  border-color: var(--maroon);
}
```

#### Micro Reset Button (`.btn-micro`)
```css
.btn-micro {
  background: var(--paper-raised);
  border: 1px solid var(--line);
  padding: 5px 12px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 600;
  color: var(--maroon);
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}
.btn-micro:hover {
  background: var(--maroon);
  color: #FFFFFF;
  border-color: var(--maroon);
}
```

#### Domain Selector Tabs (`.domain-tab-btn`)
```css
.domain-tab-btn {
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 500;
  padding: 8px 16px;
  border-radius: 20px;
  border: 1px solid var(--line);
  background: var(--paper-raised);
  color: var(--ink-soft);
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s;
}
.domain-tab-btn:hover {
  color: var(--maroon);
  border-color: var(--maroon);
  background: var(--maroon-wash);
}
.domain-tab-btn.active {
  background: var(--maroon);
  color: #FFFFFF;
  border-color: var(--maroon);
  font-weight: 600;
}
```

---

### 5.4 Live Terminal Window & Execution Stream

```css
.terminal-window {
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 8px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
  overflow: hidden;
  margin-top: 16px;
  margin-bottom: 24px;
}
.terminal-header {
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.terminal-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}
.terminal-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
}
.dot-red    { background: #ff5f56; }
.dot-yellow { background: #ffbd2e; }
.dot-green  { background: #27c93f; }

.terminal-title {
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 12px;
  color: #8b949e;
  letter-spacing: 0.02em;
}

.terminal-body {
  background: #0d1117;
  padding: 18px 20px;
  max-height: 440px;
  overflow-y: auto;
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 13px;
  line-height: 1.55;
  color: #c9d1d9;
}
.terminal-line {
  margin-bottom: 4px;
  word-break: break-all;
  display: flex;
  align-items: baseline;
  gap: 8px;
}
.term-tag {
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11.5px;
  flex-shrink: 0;
}
```

---

### 5.5 Visualizer Viewports & Floating Overlays

```css
#canvas3d-container {
  width: 100%;
  height: 480px;
  background: linear-gradient(135deg, #FAF8F5 0%, #EFEBE3 100%);
  border: 1px solid var(--line);
  border-radius: 6px;
  position: relative;
  overflow: hidden;
  cursor: grab;
}
#canvas3d-container:active {
  cursor: grabbing;
}

.overlay-badge {
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(251, 250, 247, 0.92);
  backdrop-filter: blur(4px);
  color: var(--ink);
  padding: 6px 12px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 12.5px;
  font-weight: 500;
  border: 1px solid var(--line);
  pointer-events: none;
  z-index: 5;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.overlay-hint {
  position: absolute;
  bottom: 12px;
  right: 14px;
  background: rgba(251, 250, 247, 0.92);
  color: var(--ink-soft);
  padding: 4px 10px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 11.5px;
  border: 1px solid var(--line);
  pointer-events: none;
  z-index: 5;
}

.overlay-status {
  position: absolute;
  top: 14px;
  right: 14px;
  background: var(--maroon-wash);
  color: var(--maroon);
  padding: 6px 12px;
  border-radius: 4px;
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 600;
  border: 1px solid var(--line);
  pointer-events: none;
  z-index: 5;
}
```

---

### 5.6 Interactive Range Sliders & Presets

```css
.alignment-slider-bar {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px 18px;
  margin-top: 14px;
}
.custom-range-slider {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: var(--line);
  outline: none;
  cursor: pointer;
  accent-color: var(--maroon);
}
.preset-chip {
  font-family: var(--sans);
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 12px;
  border: 1px solid var(--line);
  background: var(--paper);
  color: var(--ink-soft);
  cursor: pointer;
  transition: 0.15s;
}
.preset-chip:hover {
  background: var(--maroon-wash);
  color: var(--maroon);
  border-color: var(--maroon);
}
```

---

### 5.7 Benchmark Tables & Stat Cards

```css
.benchmark-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--sans);
  font-size: 14.5px;
  margin-top: 24px;
  background: var(--paper-raised);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
}
.benchmark-table th {
  background: var(--paper);
  padding: 12px 18px;
  text-align: left;
  border-bottom: 2px solid var(--line);
  font-weight: 600;
  color: var(--ink);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.benchmark-table td {
  padding: 12px 18px;
  border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}
.row-highlight-ours {
  background: #fdf2f4 !important;
}
.row-highlight-ours td {
  border-top: 2px solid var(--maroon) !important;
  border-bottom: 2px solid var(--maroon) !important;
  font-weight: 600;
}

.stat-box {
  background: var(--paper-raised);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px;
  text-align: center;
}
.stat-box .val {
  font-family: var(--sans);
  font-weight: 700;
  font-size: 1.7rem;
  color: var(--maroon);
  line-height: 1.2;
}
.stat-box .desc {
  font-family: var(--sans);
  font-size: 12.5px;
  color: var(--ink-soft);
  margin-top: 4px;
}
```

---

### 5.8 Mathematical & Theoretical Proof Boxes

Used in [`docs.html`](docs.html) and [`methodology.html`](methodology.html) for formal theorems:

```css
.theorem-box {
  background: var(--paper);
  border-left: 4px solid var(--maroon);
  padding: 16px 20px;
  margin: 18px 0;
  border-radius: 0 6px 6px 0;
  font-size: 15.5px;
}
.theorem-box strong {
  color: var(--maroon);
  font-family: var(--sans);
}
.math-block {
  overflow-x: auto;
  padding: 10px 0;
  margin: 12px 0;
}
code.inline-code {
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  background: var(--paper);
  border: 1px solid var(--line);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
  color: var(--maroon);
}
pre.code-block {
  background: #0d1117;
  color: #c9d1d9;
  padding: 16px 20px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  line-height: 1.5;
  overflow-x: auto;
}
```

---

### 5.9 Keyframe Animations & Micro-Interactions

```css
/* Live Terminal Cursor Blink */
@keyframes blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0; }
}

/* Optimizing Status Pulse */
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}
```

---

## 6. External Libraries, CDNs & Rendering Engines

The website architecture intentionally avoids bloated JavaScript frameworks (React, Angular, or Vue), keeping initial page loads instantaneous and lightweight using vanilla JavaScript paired with three specialized scientific rendering engines:

| Library / Engine | Version | CDN Provider | Purpose & Usage |
| :--- | :--- | :--- | :--- |
| **Three.js** | `r128` | Cloudflare / JSDelivr | Real-time WebGL rendering of 3D molecular structures, atomic sphere meshes, cylinder bonds, and spatial coordinates. |
| **OrbitControls** | `0.128.0` | JSDelivr | Full camera manipulation (3-axis spherical rotation, mouse pan, and pinch zoom). |
| **Chart.js** | `v3.x / v4.x` | JSDelivr | Real-time Frank-Wolfe duality gap descent curves, parameter sweep canvases, and execution runtime bar charts. |
| **KaTeX** | `0.16.8` | JSDelivr | Ultra-fast client-side mathematical typesetting for Theorems 1–4, Lemmas 1–2, and quadratic form equations. |
| **KaTeX Auto-Render**| `0.16.8` | JSDelivr | Parses `$$...$$` (display) and `$...$` (inline) LaTeX expressions across the DOM upon load. |

---

## 7. Directory Structure of Web Assets

```
website/
├── DESIGN_SYSTEM.md                           <- This Style & Design Specification
├── README.md                                  <- Web server setup & benchmark guide
├── style.css                                  <- Master CSS stylesheet (Core design tokens)
├── app.py                                     <- Python Flask backend server (port 8000)
├── index.html                                 <- Portal landing page & architectural summary
├── mapping.html                               <- 3D WebGL Molecular Viewer & 7-model benchmark
├── demo.html                                  <- Live Frank-Wolfe terminal & multi-domain sandbox
├── methodology.html                           <- Algorithmic pipeline & continuation steps
├── docs.html                                  <- Formal mathematical proofs (Theorems 1-4)
├── progress.html                              <- Verification checklists & domain metrics
└── team.html                                  <- Student researchers, guide, & affiliations
```

---

*Authored for the Final Year Project Evaluation Committee, Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham.*
