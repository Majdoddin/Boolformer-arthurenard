# Huang 2025 — Nguyen Benchmark Runs

Log of per-case Huang MCTS runs (with `R` in default ops, 1 run each, seed 23654).
Hyperparameters: max_evals=2M, max_depth=6, K=500, c=4.0, γ=0.5.
Success criterion: `reward >= 1 - 1e-6` (matches paper convention).

## Nguyen-1: `x³ + x² + x`

- **Range:** [-1, 1], 40 samples
- **Result:** ✅ success — reward 1.0, 668 evals, 47ms, complexity 9
- **Found:** `x0·(x0·(x0 + 0.9999999738) + 0.9999999735)`
- **nsimplify(1e-3):** `x0·(x0·(x0+1)+1)` = `x0³ + x0² + x0` — exact

## Nguyen-2: `x⁴ + x³ + x² + x`

- **Range:** [-1, 1], 40 samples
- **Result:** ✅ success — reward 1.0, 20k evals, 1.9s, complexity 13
- **Found:** `(1 + x0·(1.19e-7 + x0)) · 1 · (1 + x0) · x0` (with tiny cross-term)
- **nsimplify(1e-3):** `x0·(x0+1)·(x0²+1)` = `x0⁴ + x0³ + x0² + x0` — exact

## Nguyen-3: `x⁵ + x⁴ + x³ + x² + x`

### Attempt 1 — paper's original range [-1, 1]
- **Result:** ✅ success numerically — reward 1.0, 198k evals, 24s, complexity 24
- **But structurally WRONG:** `x0²·(x0 + 904/917)·(x0² + 857/970) + exp(0.274·x·(x+0.47))·sin(x0)`
- The narrow range lets `exp·sin` approximate the missing `x0` term. Formula would blow up outside [-1, 1].

### Attempt 2 — widened range [-10, 10]
- **Result:** ✅ success and structurally correct — reward 1.0, 12k evals, 1.5s, complexity 14
- **Found:** `((x0 + 1)·x0·(x0² + 1) + 1)·x0`
- **nsimplify(1e-3):** `x0·(x0·(x0+1)·(x0²+1) + 1)` = `x0⁵ + x0⁴ + x0³ + x0² + x0` — exact
- **Observation:** wider range forced true symbolic recovery. Also **16× fewer evals** than narrow range.

## Nguyen-4: `x⁶ + x⁵ + x⁴ + x³ + x² + x`

- **Range:** [-1, 1], 40 samples
- **Result:** ✅ success — reward 1.0, 195k evals, 23s, complexity 18
- **Found:** `(1 + x0³)·x0·(x0² + 1 + x0)` after tiny-term cleanup
- **nsimplify(1e-3):** `x0·(x0³+1)·(x0²+x0+1)` = `x0⁶ + x0⁵ + x0⁴ + x0³ + x0² + x0` — exact
- **Observation:** clean Horner-like factorization found on narrow range (unlike Nguyen-3 which overfit the range).

## Nguyen-5: `sin(x²)·cos(x) - 1`

- **Range:** [-1, 1], 40 samples
- **Result:** ✅ success — reward 1.0, 75k evals, 7.4s, complexity 11
- **Found:** `-0.99999999 + sin(x0²)·cos(1.00000001·x0)`
- **nsimplify(1e-3):** `sin(x0²)·cos(x0) - 1` — exact

## Nguyen-6: `sin(x) + sin(x + x²)`

- **Range:** [-1, 1], 40 samples
- **Result:** ✅ success — reward 1.0, 2.5k evals, 0.19s, complexity 13
- **Found:** `sin(x0·(1 + x0)) + sin(x0 + tiny)`
- **nsimplify(1e-3):** `sin(x0) + sin(x0·(x0+1))` — exact
- **Observation:** fastest non-trivial solve so far (<200ms).

## Nguyen-7: `log(x²+1) + log(x+1)`

- **Range:** [0, 2], 40 samples
- **Result:** ✅ success — reward 1.0, 79k evals, 9.4s, complexity 12
- **Found:** `log((x0²+1)·(x0+1))`
- **nsimplify(1e-3):** `log((x0+1)·(x0²+1))` — equals target via `log(ab) = log(a)+log(b)`. Exact.

## Nguyen-8: `sqrt(x)`

- **Range:** [0, 4], 40 samples
- **Result:** ✅ success — reward 1.0, 341 evals, 24ms, complexity 16
- **Found:** `exp(0.5·(log(x0) + tiny·(x0 - 1/x0)))`
- **nsimplify(1e-3):** `sqrt(x0)` — exact
- **Observation:** fastest solve (24ms). `sqrt` not in op set, so MCTS found `exp(0.5·log(x))` — a non-trivial reformulation using 4 ops.

## Nguyen-9: `sin(x) + sin(y²)`

- **Range:** [0, 1], 40 samples, 2 vars
- **Result:** ✅ success — reward 1.0, 16k evals, 1.4s, complexity 30
- **Found:** `sin(x1²) + tiny·(x1/x1)/exp(sin(x0)) + (sin(x0) + exp(-31.55 - x1))` — lots of ~1e-14 spurious terms
- **nsimplify(1e-3):** `sin(x0) + sin(x1²)` — exact

## Nguyen-10: `2·sin(x)·cos(y)`

- **Range:** [0, 1], 40 samples, 2 vars
- **Result:** ✅ success — reward 1.0, 10k evals, 0.7s, complexity 8
- **Found:** `sin(x0)·0.9967 / (0.4984 / sin(1.5708 - x1))`
- **nsimplify(1e-3):** `-2·sin(x0)·sin(x1 - 355/226)` where `355/226 ≈ π/2`
- **Equivalent to target via** `sin(a - π/2) = -cos(a)` → `2·sin(x0)·cos(x1)`. nsimplify didn't recognize π/2, but formula is exact.
- **Observation:** MCTS used a trigonometric identity instead of directly producing `2·sin·cos`. `R` was fit to π/2 numerically.

## Nguyen-11: `x^y`

- **Range:** [0, 1], 40 samples, 2 vars
- **Result:** ✅ success — reward 1.0, 22k evals, 1.6s, complexity 10
- **Found:** `exp((x1 + tiny·x0)·log(x0))`
- **nsimplify(1e-3):** `exp(x1·log(x0))` = `x0^x1` — exact
- **Observation:** `**` not in op set, so MCTS used `exp(y·log(x))` — same reformulation trick as Nguyen-8's sqrt.

## Nguyen-12*: `x0⁴ - x0³ + 0.5·x1² - x1`

- **Range:** [0, 10], 40 samples, 2 vars
- **Result:** ✅ success — reward 1.0, 225k evals, 26s, complexity 21
- **Found:** `(x1 + 1.841)·(0.5·x1 - 0.921) - (x1 - 1.696) - x0³·(1 - x0)` (via unclean factorization)
- **expand + nsimplify(1e-3):** `x0⁴ - x0³ + x1²/2 - x1` — exact
- **Note:** required `expand()` *before* `nsimplify()` — the default `simplify()` pipeline in the runner doesn't expand, so coefficients stay entangled in the factored form and nsimplify can't snap them to clean values.

---

---

# Run 2 — Without R (constant token excluded)

Same seed, same hyperparameters, same sample ranges (Nguyen-3 kept at [-10, 10]).
Ops: `+,-,*,/,sin,cos,exp,log` (no `R`).

## Nguyen-1: `x³ + x² + x` — no R

- **Result:** ✅ success — reward 1.0, 13k evals, 179ms, complexity 9
- **Found:** `x0 + x0·x0·(x0² + x0)/x0`
- **expand + nsimplify:** `x0³ + x0² + x0` — exact, no constants needed

## Nguyen-2: `x⁴ + x³ + x² + x` — no R

- **Result:** ✅ success — reward 1.0, 56k evals, 0.9s, complexity 12
- **Found:** `(x0 + x0²)·x0² + x0² + x0`
- **expand + nsimplify:** `x0⁴ + x0³ + x0² + x0` — exact

## Nguyen-3: `x⁵ + x⁴ + x³ + x² + x` — no R (range [-10, 10])

- **Result:** ✅ success — reward 1.0, 98k evals, 1.6s, complexity 14
- **Found:** `(x0² + x0)·(x0/x0 + x0²)·x0²/x0 + x0`
- **expand + nsimplify:** `x0⁵ + x0⁴ + x0³ + x0² + x0` — exact
- **Observation:** used `x0/x0 = 1` trick to synthesize the integer 1 without `R`.

## Nguyen-4: `x⁶ + x⁵ + x⁴ + x³ + x² + x` — no R

- **Result:** ✅ success — reward 1.0, 145k evals, 2.7s, complexity 16
- **Found:** `(x0³ + x0 + x0/x0/x0 - (x0 - x0))·x0·(x0 + x0²)`
- **expand + nsimplify:** `x0⁶ + x0⁵ + x0⁴ + x0³ + x0² + x0` — exact
- **Observation:** uses `(x0 - x0) = 0` and `x0/x0/x0 = 1/x0` token tricks.

---

## Summary

All 12 Nguyen cases succeeded (reward 1.0) and recovered the exact target after `expand() + nsimplify(tolerance=1e-3)`. Key observations:

- **Nguyen-3 on original range [-1, 1]** numerically succeeded but with a structurally wrong formula exploiting the narrow range (`exp·sin` faking the linear term). Widening to [-10, 10] forced true recovery and was 16× faster.
- **R (constant token) in default ops** is essential for Nguyen-12* and helps several others. Paper's convention of omitting R is outdated.
- **expand + nsimplify(tolerance)** in post-processing is far cleaner than the runner's default `simplify` without `rationalize_constants`. Would turn cosmetically-messy successes into verified-clean matches.
- **Reformulation tricks:** MCTS can compose unavailable ops — `sqrt` via `exp(0.5·log(x))`, `x^y` via `exp(y·log(x))`, `cos` via `sin(π/2 - x)`.
- **Total time for 12 cases:** ~95 seconds single-core.
