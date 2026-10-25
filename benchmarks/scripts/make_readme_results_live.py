"""Writes the live-results section of README.md from paper/live/live_values.tex.

The section between <!-- LIVE-RESULTS:BEGIN --> and <!-- LIVE-RESULTS:END --> is regenerated; every
number is taken from the live values produced by make_paper_tables_live.py.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- LIVE-RESULTS:BEGIN -->", "<!-- LIVE-RESULTS:END -->"


def md(v: str) -> str:
    v = v.replace("{,}", ",")
    v = re.sub(r"\\times 10\^\{(-?\d+)\}", r"e\1", v)
    return v.replace("$", "")


TEMPLATE = """{BEGIN}
## 5. Live Benchmark Results

All numbers in this section are generated from `benchmarks/live_runs/` (five scripts `benchmarks/scripts/run_*_live.py`,
executed from scratch on one Intel Core i7-1355U laptop CPU with single-threaded workers). Full tables:
`paper/live/tab_*.tex`; raw and aggregated data: `benchmarks/live_runs/csv/`; figures:
`benchmarks/figures/standalone/`. `python benchmarks/scripts/verify_paper_numbers.py` recomputes the paper's numbers
from the per-trial records.

| Benchmark | Fast-WCS v3 (defaults) | Strongest alternatives | Outcome |
|---|---|---|---|
| Synthetic, protocol of Yang et al. 2018 ({syn.ninstances} instances, 13 methods) | {syn.all.fwd.accavg}% mean accuracy, {syn.all.fwd.ms} ms median | RRWM {syn.all.rrwm.accavg}% and IPFP {syn.all.ipfp.accavg}% (Gaussian edge affinity, {syn.spd.rrwm}x / {syn.spd.ipfp}x slower); RLX1F {syn.all.rlxonef.accavg}% ({syn.spd.rlxonef}x slower) | Fastest GNCCP solver; clearly less accurate |
| Real images, Oxford Affine ({img.npairs} pairs, {img.ntrials} trials per setting) | {img.all.fwd.accavg}%, {img.all.fwd.ms} ms | Partial FGW {img.all.pgw.accavg}% ({img.all.pgw.ms} ms); RLX2 {img.all.rlxtwo.accavg}% ({img.spd.rlxtwo}x slower); RRWM {img.all.rrwm.accavg}% ({img.spd.rrwm}x slower) | Beats RLX1F/RLX1, ties RRWM; partial FGW is better and faster |
| (N, L) grid, N <= 150, {grid.ncells} cells | {grid.spd.rlxonef.min}-{grid.spd.rlxonef.max}x faster than RLX1F, {grid.spd.rlxone.min}-{grid.spd.rlxone.max}x faster than RLX1 | RLX1F more accurate at L = 0.8N ({grid.avg.0p8.rlx1f.acc}% vs {grid.avg.0p8.fast.acc}%) | Speed win in every cell; accuracy loss for partial L |
| 3D all-atom superposition (300 trials) | {td.all.fw.sone}% < 1 A, {td.all.fw.stenth}% < 0.1 A, {td.all.fw.ms} ms | IPFP+Kabsch {td.all.ipfp.sone}% / {td.all.ipfp.stenth}% ({td.all.ipfp.ms} ms); ICP 24 starts {td.all.icpms.sone}% / {td.all.icpms.stenth}% ({td.all.icpms.ms} ms) | Not the best; multi-start ICP dominates in latency and < 1 A rate |
| Component ablation (N = 30 / 60 / 100) | Full stack {abl.30.S5.spd}-{abl.100.S5.spd}x faster than Yang RLX1F; library defaults {abl.30.REF.spd}-{abl.100.REF.spd}x | Exact gradient + exact oracle lower the solved share (N = 60: {abl.60.S0.solved}% -> {abl.60.S4.solved}%) | Exact components buy speed, not a better homotopy path |
| Line search per call | Cardano quartic {abl.30.ls.cardano.us} / {abl.60.ls.cardano.us} / {abl.100.ls.cardano.us} us | 11-point grid {abl.30.ls.grid.us} / {abl.60.ls.grid.us} / {abl.100.ls.grid.us} us | {abl.30.ls.ratio}x / {abl.60.ls.ratio}x / {abl.100.ls.ratio}x cheaper |
| 2-opt refinement | Objective of discretized solution lowered in {abl.30.S5.twoopt.rate}% / {abl.60.S5.twoopt.rate}% / {abl.100.S5.twoopt.rate}% of runs, median {abl.30.S5.twoopt.drop}% / {abl.60.S5.twoopt.drop}% / {abl.100.S5.twoopt.drop}% | | Local improvement only |
| Gradient (N = 100) | Theorem 1: {grad.100.thm} ms | RLX1 gradient {grad.100.rlx} ms; finite differences {grad.100.fd} ms | {grad.100.spdfd}x faster than FD, max rel. error {grad.maxerr} |

Measured runtime exponents (L = 0.8N, N in [40, 150]): Fast-WCS {grid.slope.fast}, RLX1F {grid.slope.rlx1f},
RLX1 {grid.slope.rlx1}. These are fitted exponents in this size range, not asymptotic complexities.
{END}"""


def main():
    vals = dict(re.findall(r"\\LVdef\{([^}]+)\}\{(.*)\}\s*$",
                           (ROOT / "paper" / "live" / "live_values.tex").read_text(encoding="utf-8"), flags=re.M))
    text = re.sub(r"\{([a-zA-Z][\w.]*)\}", lambda m: md(vals[m.group(1)]) if m.group(1) in vals else m.group(0), TEMPLATE)
    text = text.replace("{BEGIN}", BEGIN).replace("{END}", END)
    missing = re.findall(r"\{([a-z][\w.]*\.[\w.]+)\}", text)
    if missing:
        raise KeyError(missing)
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    a, b = s.index(BEGIN), s.index(END) + len(END)
    readme.write_text(s[:a] + text + s[b:], encoding="utf-8")
    print("README live-results section written")


if __name__ == "__main__":
    main()
