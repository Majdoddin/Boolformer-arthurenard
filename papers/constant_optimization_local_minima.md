# Constant Optimization in Symbolic Regression: Local Minima and Remedies

## The problem

MCTS-4-SR seeds all constants at 1.0 (`bridge.cpp:30`) and uses Levenberg-Marquardt
(Eigen's MINPACK port) to optimize them. LM is a local optimizer — it converges
to the nearest local minimum, not the global one.

**Empirical demonstration** (run on our codebase): fitting `R * sin(R * x)` to
`3 * sin(freq * x)` with seed (1, 1) and 500 LM iterations:

| True (amplitude, freq) | LM found | MSE | Status |
|---|---|---|---|
| (3, 2) | (3.000, 2.000) | 5.8e-15 | OK |
| (3, 5) | (0.476, 2.510) | 4.3e+00 | Stuck |
| (3, 10) | (0.164, 1.469) | 4.4e+00 | Stuck |
| (3, 20) | (0.036, 1.466) | 4.4e+00 | Stuck |
| (3, 50) | (-3.000, 1.000) | 2.9e-10 | Aliased |
| (3, 100) | (3.000, 2.000) | 1.3e-09 | Aliased |

LM only finds the correct constants when the true frequency is close to the
seed (freq=2 from seed=1). For larger gaps, LM converges to a wrong basin
in ~7 function evaluations and declares "done" via ftol. More iterations
don't help — it's a basin problem, not a budget problem.

## Which constants are vulnerable

The key distinction is **linear vs nonlinear** appearance of the constant:

**Linear constants** (amplitude, offset): `R * f(x) + R`. The loss landscape
is convex in these constants — LM always finds the global optimum. No local
minima possible.

**Nonlinear constants** (frequency, exponent, denominator): constants that
appear inside nonlinear functions create multi-modal landscapes:

| Pattern | Why multi-modal |
|---|---|
| `sin(R * x)`, `cos(R * x)` | Periodic — many frequencies fit similarly on finite data |
| `exp(R * x)` | Exponential sensitivity — small R changes cause large output changes |
| `1 / (R + x)` | Landscape changes shape sharply near R = 0 |
| `x ^ R` | Nonlinear in exponent |

Kommenda et al. (2022) found that rank-deficient and ill-conditioned Jacobians
occur **frequently** in GP for symbolic regression, not just in edge cases.

## LM convergence behavior (empirical, from our tests)

Eigen's LM almost never exhausts the function evaluation budget (`maxfev`).
Even with `lm_iterations=1` (maxfev=3) on a 2-constant formula, LM exits
via `ftol` or `xtol` after 3 function evaluations. With the real budget of
50 (maxfev=52), a 6-constant nonlinear formula converged in 13 evaluations
during warm-start.

**Implication**: the iteration budget is not a meaningful bottleneck. LM
either finds a good local minimum quickly (3-13 evals) or gets stuck in a
bad one equally quickly. Additional iterations (warm-start, Tier 2) don't
help escape bad basins — LM is monotone-convergent within a basin.

## Remedies from the literature

### Multi-start LM

Run LM from k random starting points, keep the best result.

```
P(success with k starts) = 1 - (1-p)^k
```

where p = probability of a single start landing in the correct basin. Cost
scales linearly (k × LM cost), benefit logarithmically (diminishing returns).
Works when p is moderate (say >0.1); fails when p ≈ 0 (many basins, high
dimension).

### Global optimizers

| Method | Typical function evals (2-6 constants) | Strengths |
|---|---|---|
| Pure LM | 3-13 | Fast, exact for convex |
| CMA-ES | 600-2700 | Handles multi-modal, non-separable |
| CMA-ES(short) + LM | 150-500 | Global basin finding + fast local polish |
| Differential evolution | 500-2000 | Simple, robust |

CMA-ES alone is ~50-200× more expensive than LM per formula. A short
CMA-ES (20-50 generations) to find the basin followed by LM to polish
is ~30-50× more expensive.

Benchmarking SR constant optimization (Kommenda et al. 2024): LM had the
highest success rate overall, but PSO/BFGS performed better on wider ranges
of problems. No single method dominates all cases.

### Geodesic acceleration (Transtrum & Sethna 2012)

Second-order correction to the LM step that helps navigate narrow canyons
faster. Does NOT escape local minima — converges faster within the same
basin. One extra function evaluation per iteration. Not useful for the
basin problem; useful for ill-conditioned problems with elongated valleys.

### Uphill step acceptance (Umrigar & Nightingale, via Transtrum 2012)

Sometimes accept steps that increase the cost (like simulated annealing).
Can cross shallow ridges between basins. Marginal benefit — doesn't solve
deep multi-modality. Trades convergence speed for robustness to initial guess.

## The gating problem

Running a global optimizer on every formula is prohibitive (30-50× cost ×
millions of formulas). The natural idea is to gate it on top-N qualification:
only spend the expensive optimization on formulas that are "good enough."

**This doesn't work for global optimization.** If LM gets stuck in a bad
basin, the formula gets a low reward and never qualifies for top-N. The
formulas that need the global optimizer most are exactly the ones that fail
the gate. The gate is safe for warm-start LM (which can't escape basins
anyway) but defeats the purpose of a global optimizer (which can).

## Recommendation: structure-based gating

Gate on **structural features** of the formula, not on the reward:

```
if formula has R inside sin/cos/exp/log/pow arguments:
    run CMA-ES(short) + LM    (~150-500 function evals)
else:
    run LM only                (~3-13 function evals)
```

**Rationale.** Constants that appear linearly (`R*f(x) + R`) have convex
landscapes — LM is optimal, no global search needed. Constants inside
nonlinear functions (`sin(R*x)`, `exp(R*x)`) have multi-modal landscapes —
LM is likely to get stuck, global search is needed.

**Detection is cheap.** Walk the tree once: if any R leaf is a descendant
of a nonlinear unary op (sin, cos, exp, log, sqrt) without an intervening
binary op that makes it linear (+ or -), flag for global optimization. This
reuses the same tree-walk infrastructure as the risky-path check in
`simplify.hpp`.

**Cost estimate for MCTS-4-SR.** With the Nguyen benchmark operators
`{+, -, *, /, sin, cos, exp, log}` and typical tree depth ≤ 6:
- ~30-50% of formulas contain R inside nonlinear ops (rough estimate)
- These get CMA-ES(short)+LM: ~300 evals each
- Remaining 50-70% get pure LM: ~10 evals each
- Weighted average: ~100-160 evals per formula (vs current ~10)
- 10-16× total cost increase

This is significant. To keep it manageable:
- Apply only to top-N qualifiers + formulas that **nearly** qualify
  (e.g., reward > 0.5 × best_reward). This narrows the population.
- Use a very short CMA-ES (10-20 generations, ~60-120 evals) as a
  basin-finder, not a full global search.
- Cache by structure hash: same structure → same CMA-ES result.
  Only pay once per unique structure.

With caching and reward gating, the overhead drops to ~2-5% of total
evaluations — comparable to the Tier 2 pipeline cost.

## Open questions

1. **How often does the basin problem actually hurt benchmark results?**
   Nguyen formulas mostly have O(1) constants close to seed=1.0. The
   frequency-stuck problem may rarely arise. Need empirical measurement.

2. **Does MCTS diversity compensate?** MCTS evaluates many structurally
   different formulas. Even if one formula's LM gets stuck, a structurally
   different formula with the same semantics may land in a better basin.
   The search-level diversity might already solve the problem that
   per-formula global optimization addresses.

3. **CMA-ES implementation.** Eigen doesn't include CMA-ES. Would need
   an external library (libcmaes, or a header-only implementation) or a
   simple custom implementation (~200 lines for basic CMA-ES).

## References

- Transtrum & Sethna (2012). [Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization](https://arxiv.org/abs/1201.5885). Geodesic acceleration, uphill steps, delayed gratification.
- Kommenda et al. (2024). [Benchmarking symbolic regression constant optimization schemes](https://arxiv.org/html/2412.02126v1). LM vs BFGS vs PSO vs DE on SR problems.
- Kommenda et al. (2022). [Local optimization often is ill-conditioned in genetic programming for symbolic regression](https://arxiv.org/html/2209.00942). Rank-deficient Jacobians in SR.
- Chau (2024). [Multistart nonlinear least-squares fitting with gslnls](https://jchau.org/2024/07/31/multistart-nonlinear-least-squares-with-gslnls/). GSL multi-start implementation.
- Transtrum et al. (2011). [Geometry of nonlinear least squares](https://link.springer.com/article/10.1007/s11081-020-09571-2). Cluster Gauss-Newton for global NLS.
