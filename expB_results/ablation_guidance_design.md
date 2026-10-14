# Ablation Design: Binding-guided K5 vs Utilization-guided K5 vs Dual-guided K5

**Status: DESIGN ONLY — not executed.** Awaiting approval after Task 1–2 report.

## Motivation

Experiment B Stage 0 established that continuous dual variables (ν) improve
candidate quality over pure β-centroid directions. Open question: does the
*continuous* dual signal add value beyond simpler proxies for the same
underlying phenomenon (backhaul congestion)?

## Factors under test

All variants share identical:
- Candidate count K=5: `[stay, dir(ρ=0.25), dir(ρ=0.50), dir(ρ=0.25)+air, dir(ρ=0.50)+air]`
- System parameters, Block A solver, Lyapunov J, power model, motion constraints
- Exogenous inputs (same seeds / frozen states), cids protocol
- ρ grid, airship candidate logic, c_ν = 0.1 for the dual blend

Only the **UAV direction generator** changes:

| Variant | UAV unit direction e_u |
|---------|------------------------|
| Dual-guided (baseline) | `(1−w)·e_u^β + w·e_u^BH`, `w = ν̂/(c_ν+ν̂)`, fallback on cancellation |
| Binding-guided | `e_u^BH` if UAV's backhaul constraint is active/binding at base solve; else `e_u^β`. Binding test: dual ν_u > ε_ν **or** slack_u < ε_slack |
| Utilization-guided | `e_u^BH` if η_u = C_u^demand / C_u^capacity > η_thresh; else `e_u^β`. Threshold η_thresh tuned on a calibration split (default 0.8) |

All three re-solve full Block A at candidate positions before scoring.

## Evaluation protocol

### Setting A — Frozen 152 snapshots (primary, zero sim cost)
Same as Stage 0 / Task 2:
- Reproduce base solve; derive ν, slack, utilization per UAV
- Generate candidates under each variant
- Score with unified J = ψ_E·(E·P) − β·D
- Metrics: dJ vs β-K3 baseline; dual/binding/util selected rate;
  fraction of slots where variant direction ≠ β direction

### Setting B — Closed-loop seed 0/2 × 501 slots (confirmatory)
Only if Setting A shows a meaningful gap:
- Same Hybrid-slow / dualk5 infrastructure, K=5, same seeds
- Report wMOS, web queue, power, backhaul utilization
- Cost: ~2 min per variant per seed

## Hypotheses

- H1 (continuous duals matter): Dual-guided > Utilization-guided > β-K3 on dJ
- H0a: Utilization-guided ≈ Dual-guided → binary congestion flag suffices
- H0b: Binding-guided ≈ Dual-guided → 0/1 binding indicator suffices
- H0c: All ≈ β-K3 → dual machinery is unnecessary for K=5 candidate guidance

## Decision rule

If |mean dJ(Dual) − mean dJ(Util)| < 0.05 **and** paired win-rate < 55%,
report utilization as the simpler sufficient signal and drop continuous duals
from the headline method (keep as optional refinement). Otherwise retain duals.

## Files (when executed)

- `expB_ablation_guidance.py` — Setting A (+ optional Setting B)
- `expB_results/ablation_guidance/` — slots CSV, summary, figure
