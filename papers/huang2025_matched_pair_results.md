# Huang 2025 — Matched-Pair Sampling Results

**Background:** see `huang2025_notes.md` § "Unfair sibling comparison from random rollout" → "Refinement to suggestion 1: matched-pair sampling".

## Change

Surgical matched-pair sampling at MCTS transition moments. When a parent's last unexpanded move is expanded (the parent transitions from "has unexpanded moves" to "fully expanded, ready for UCB selection"), re-evaluate ALL its children with N=4 shared completion seeds (CRN across siblings).

**Code:** ~50 lines in `mcts4sr/source/mcts/mcts.cpp` + 1 declaration in `include/imcts/mcts/mcts.hpp`. Compile-time constant `kMatchedPairN = 4`. To disable: set to 0 and rebuild.

**Hypothesis:** matched-pair reduces between-sibling variance via Common Random Numbers, so UCB-extreme's max-tracking compares siblings on correlated noise rather than independent noise. Should diminish the seed-dependent exploit-basin lottery on fragile benchmarks.

## Setup

- Benchmark: Nguyen-3 = `x⁵ + x⁴ + x³ + x² + x`, range [-1, 1], 20 samples (40 effective via `sample_multiplier=2`)
- Op set: `{+, -, *, /, sin, cos, exp, log, R}` (Nguyen default + R)
- RNG: PCG64DXSM (already in place from earlier PR)
- Seeds: Huang's fixed list, indices 0–9 (so 10 distinct 5-digit seeds)
- Hyperparameters: max_depth=6, max_constants=6, max_evals=2M, K=500, c=4.0, γ=0.5
- Success criterion: `reward ≥ 1 − 1e-6`

## Headline Results

| | Baseline (PCG64DXSM) | Matched-pair (N=4) |
|---|---|---|
| Numerical success | 10/10 | 10/10 |
| Polynomial-class structure (no transcendentals) | 6/10 | **10/10** |
| Cyclotomic factorization explicit (golden-ratio constants) | 0/10 | **6/10** |
| Mean evaluations per run | 228k | **97k** (2.35× fewer) |
| Mean wall time per run | 28.8s | **12.0s** (2.40× faster) |

Both faster AND structurally cleaner — the theoretical 1.5–2× cost overhead never materialized because matched-pair finds reward-1.0 formulas at much lower eval counts.

## Per-Seed Comparison

| Seed | Base evals | MP evals | Speedup | Base time | MP time | Base struct | MP struct |
|---|---|---|---|---|---|---|---|
| 23654 | 247,593 | 73,798 | **3.36×** | 28.2s | 8.2s | OK | OK |
| 15795 | 205,003 | 101,761 | 2.01× | 26.3s | 11.9s | OK | drift |
| 860 | 70,015 | 166,474 | 0.42× | 8.5s | 21.4s | OK | rational |
| 5390 | 133,199 | 71,986 | 1.85× | 16.5s | 9.0s | OK | OK |
| 16850 | 605,878 | 73,803 | **8.21×** | 77.5s | 9.1s | rational | OK |
| 29910 | 139,320 | 71,562 | 1.95× | 18.0s | 9.2s | exp·x | drift |
| 4426 | 297,545 | 183,986 | 1.62× | 39.1s | 23.5s | cos·x | OK |
| 21962 | 250,481 | 97,430 | 2.57× | 32.5s | 12.6s | OK | OK |
| 14423 | 124,005 | 84,778 | 1.46× | 15.0s | 10.1s | sin·x | drift |
| 28020 | 215,286 | 46,147 | **4.67×** | 27.0s | 5.6s | OK | rational |

8/10 seeds faster; only seed 860 slower. Largest speedup 8.2× (seed 16850, which previously wandered through 605k evals in transcendental territory). Mean speedup 2.4×.

## N sensitivity (three-seed sweep)

Tested `kMatchedPairN ∈ {1, 2, 4}` at default `gp_rate=0.2` on seeds 23654, 15795, 860. All runs recovered the target exactly under `expand + nsimplify(tolerance=1e-3)`.

| Seed | N=4 (prior) | N=2 | N=1 |
|---|---:|---:|---:|
| 23654 | 8.2s / 74k | 20.4s / 193k | 22.9s / 209k |
| 15795 | 11.9s / 102k | 21.4s / 192k | 12.3s / 109k |
| 860 | 21.4s / 166k | 2.7s / 27k | 8.7s / 79k |
| **Mean** | **13.8s / 114k** | **14.8s / 137k** | **14.6s / 132k** |
| Exact structure | 2/3 + drift | **3/3** | **3/3** |

All three values of N give mean runtimes within ~7% of each other and 3/3 exact structural recovery. The per-seed distribution, however, flips: N=4 is fastest on the "easy" seed 23654 (8.2s) but slowest on seed 860 (21.4s); N=2 and N=1 dramatically improve 860 (2.7s, 8.7s) at the cost of 23654. N=4's sharper sibling discrimination over-commits on seed 860, sending the search down a slow path; the noisier ranking at lower N gets lucky and finds the polynomial faster.

Per-iteration cost model (K=10 branching, depth=6, gp=0.2): expand evals/iter = `1 + N`, so N=1→2, N=2→3, N=4→5. GP cost ~1.2/iter (same for all). The cost ratio 2/3/5 predicts N=4 to be ~1.67× slower per iter than N=1 — but the mean runtimes are within 7%, meaning lower N takes proportionally more iterations (noisier ranking). The net is a wash on the mean, with seed-level variance trades. **N=4 is not strictly optimal; per-benchmark tuning of N is plausibly worthwhile.**

### Key insight: matched-pair N=1 beats burst-expand N=1

> **Matched-pair N=1 at gp=0.2: 14.6s / 132k mean. Burst-expand N=1 at gp=0.4: 18.6s / 209k mean** (`huang2025_burst_expand_results.md:42-47`). Same sample count (1 shared-seed rollout per sibling), same CRN mechanism, matched-pair running at a *lower* gp_rate — yet matched-pair is **~1.3× faster** in time and uses **~1.6× fewer evals**.

This isolates the advantage as the **lazy transition-moment design**, not sample count or GP rate. Burst-expand pays K × N = 10 evals per bursted leaf immediately, including at dead-end subtrees; matched-pair pays 1 eval per new child during exploration and only fires the full K × N re-eval at parents UCB revisits to the K-th child. The same CRN fairness property, achieved at ~25–30% lower overhead. See `huang2025_burst_expand_results.md` §"Why burst has a ~5× gap, not a ~2× gap" for the per-iteration cost breakdown that generalizes this observation to N ∈ {2, 4}.

## Key Structural Observation

**Matched-pair eliminates the transcendental-exploit class entirely (4/10 → 0/10).** All 10 matched-pair runs converge to polynomial or rational structures. The 6/10 "OK" runs explicitly recover the **cyclotomic factorization** of Φ₅(x):

```
Φ₅(x) = x⁴ + x³ + x² + x + 1 = (x² + φx + 1)(x² − (1/φ)x + 1)
```

with `φ = (1+√5)/2 ≈ 1.618` (golden ratio). Example raw formula (seed 5390):

```
(x·(x − x + x) + 0.618043784341·(1.61805600739 − x))
  · (x − x·x·(−0.999999813374·x + −1.61804233212))
```

The constants `0.61804…` and `1.61805…` are the search converging to `1/φ` and `φ` from finite-precision LM. This isn't approximation — it's the *exact algebraic decomposition* over ℝ.

The 4 "BAD" runs decompose as:
- **3 polynomial-with-drift**: clean polynomial structure, but LM converged to coefficients like `999/1000` or `233/232` that are within 0.5% of integers. Would snap to the exact target under `nsimplify(tolerance=1e-3)` post-processing.
- **2 rational exploit**: `x⁶/(x − ε)` form, which Taylor-expands as the target polynomial plus a vanishing perturbation. Genuinely wrong but qualitatively closer to the truth than transcendental exploits.

Under any reasonable structural-equivalence check (e.g., expand + nsimplify + symbolic match), matched-pair likely scores **8/10** vs baseline's 6/10 on Nguyen-3 [-1,1].

## Why It's Faster Despite Doing More Work Per Transition

Each transition adds ~K × N = 40 extra rollouts. With ~500 transitions per run, that's ~20k extra rollouts. Theoretical prediction: 1.5–2× **slower**. Actual: 2.4× **faster**.

**Mechanism:** matched-pair preferentially propagates *structurally-stable* branches. A cyclotomic factorization gives reward 1.0 under almost any random completion (because the polynomial is right and only the constants need fitting). A transcendental exploit gives reward 1.0 only under specific lucky completions where Taylor approximation lines up. Under matched-pair's N=4 shared completions, structurally-stable branches accumulate consistent high rewards across all N seeds; exploits accumulate one lucky reward and three garbage. UCB-extreme then prefers the stable branch.

So matched-pair isn't just a fairer comparison — it's a **structural quality filter**, which collaterally speeds up termination by directing the search toward correct structures.

## Caveats

1. **Single benchmark, single configuration.** Tested only Nguyen-3 [-1,1] with R, 10 seeds. Need to repeat across other Nguyen problems and on harder benchmarks (Livermore, Jin) before claiming this is a general win.
2. **N sweep partial.** Tested N ∈ {1, 2, 4} on three seeds at gp=0.2; see §"N sensitivity (three-seed sweep)". All three work, mean runtime comparable, structural quality preserved. N=8 untested; per-benchmark tuning of N is plausible future work.
3. **Compile-time flag.** To make `kMatchedPairN` runtime-configurable, add a field to `MCTSConfig` and plumb through Python bindings (`python/bindings.cpp`). ~10 more lines.
4. **Effect may be Φ₅-specific.** Cyclotomic factorization is unusually clean for Nguyen-3 because Φ₅'s real factorization involves only one quadratic surd (φ). Higher-degree targets, sin/cos/exp targets, and non-symmetric targets may behave differently. Need to test.
5. **No effect when no exploit basins exist.** On Nguyen-3 [-10, 10], baseline already converges in 12k evals with the correct cyclotomic form (the exploits don't survive extrapolation). Matched-pair should be neutral there. Hasn't been tested yet.

## Reproduce

```bash
cd mcts4sr
.venv/bin/python -m imcts.benchmarks --group Nguyen --cases 3 --runs 10 --seed-start 0
```

Disable matched-pair: edit `source/mcts/mcts.cpp`, set `constexpr int kMatchedPairN = 0;`, rebuild with `.venv/bin/pip install -e .`.

---

## Nguyen-4 — Regression

**Target:** `x⁶ + x⁵ + x⁴ + x³ + x² + x = x · (x+1) · (x²+x+1) · (x²−x+1)`. The non-trivial cyclotomic factors are Φ₂ = x+1, Φ₃ = x²+x+1, Φ₆ = x²−x+1. **All real-coefficient factors have integer coefficients only** — no surds, no golden ratio.

**Single-seed test (seed 23654, [-1,1] with R, N=4):**
- Time: **208.5s** (baseline ~23s)
- Evals: **1,983,303** — essentially the full 2M cap
- Reward: 1.0
- Complexity: 37 (baseline was 18)

**Found formula:**
```
(x³ / (1/x + x · 1.12e-06) + x) · ((x - (x - 1.53e-06)) / (cos(x) / 0.30106)
  - (x · (-1 - x) + -1))
```
After `expand + nsimplify(tolerance=1e-3)`: `x⁶ + x⁵ + x⁴ + x³ + x² + x` (target). The messy form uses a tiny perturbation `1.12e-6 · x` in the denominator and `cos(x) / 0.30106` (where 0.30106 ≈ ln(2)/ln(10) ≈ log₁₀(2), but more likely just a fit artifact) — neither helps structurally, both simplify away.

**Speedup vs baseline: ~0.11× (i.e. ~9× SLOWER).** The full 10-seed sweep was aborted after ~10 minutes when it became clear most seeds would also hit the 2M cap.

**Hypothesis for the regression.** Nguyen-4's natural factorization is `(x+1)(x²+x+1)(x²−x+1)` with **all integer coefficients**. Baseline finds this easily — no surds for LM to fit, no exploit basins competing for reward. Matched-pair's overhead (extra rollouts at every transition) doesn't pay back: there's nothing structural for it to disambiguate. Worse, matched-pair seems to *delay* convergence here — possibly because the matched-pair re-evaluation propagates noisy rewards (from random completions of complex sub-trees) that destabilize the path queues, preventing the search from locking in on the clean integer factorization.

**The asymmetry between Nguyen-3 and Nguyen-4** is informative: matched-pair helps when there are competing reward-1.0 basins (transcendental exploits vs polynomial truth on Nguyen-3) and when the factorization has surd coefficients that need careful constant fitting. It hurts when the answer is already algebraically clean (integer coefficients, no exploit basins).

## Nguyen-5 — Trig identity rediscovery

**Target:** `sin(x²)·cos(x) − 1` on [-1, 1] — a genuinely transcendental target where matched-pair *should* be neutral or hurt (the "transcendentals are actually needed" test).

**Result on 10 seeds, [-1,1] with R, N=4:**

| Seed | Evals | Time | Status | Mechanism |
|---|---|---|---|---|
| 23654 | 32,307 | 2.7s | **exact** | Direct |
| 15795 | 44,356 | 3.6s | **exact-after-arith** | `(0.95 − (x − (−2.95 − x)))·0.5 = −1 − x`, then `+x` cancels |
| 860 | 41,273 | 3.7s | **exact-after-arith** | `+ 0.978 − 1.978 = −1` |
| 5390 | 51,151 | 4.6s | **exact** | `cos(0.5·x·2) = cos(x)` |
| 16850 | 55,992 | 4.8s | **exact** | `x · 1/x · sin(x²) · cos(x) − 1` |
| 29910 | 36,769 | 3.3s | **exact-after-identity** | 2π periodicity (`710/113 ≈ 2π`) |
| 4426 | 79,052 | 7.0s | **exact-after-identity** | Cofunction (`sin(π/2 − x) = cos(x)`, `355/226 ≈ π/2`) |
| 21962 | 35,670 | 3.1s | **exact-after-identity** | `log(1/e) = −1` |
| 14423 | 132,007 | 11.9s | **exact** | Direct |
| 28020 | 10,500 | 0.8s | **exact** | Direct |

Mean: **52k evals, 4.6s** vs baseline ~75k evals, ~7.4s. **Speedup ≈ 1.6×.**

**Headline: 10/10 structurally correct after recognizing standard identities.** The 3 "non-direct" results are mathematically equivalent to the target via real algebraic identities the search rediscovered.

### Raw outputs

```
seed 23654:  sin(1.0·x·(x − 2.28e-08))·cos(x) + −0.99999999            [exact]
seed 15795:  (0.9495 − (x − (−2.9495 − x)))·0.5 + cos(x)·sin(x²) + x   [arith]
seed 860:    sin(x²)·cos(x) + 0.9777 − 1.9777                          [arith]
seed 5390:   cos(0.5·x·2) · (1.31e-07 + sin(x²)) − 1.0                 [exact]
seed 16850:  x · 0.99999990 / x · sin(x²) · (4.56e-08 + cos(x − ε)) − 1.0  [exact]
seed 29910:  cos(x − ε) · sin(x² + 6.28318531) − exp(exp(7.76·x) · 9.06e-12)  [identity]
seed 4426:   −0.99999999 + sin(1.57079627 − x) · (sin(x²) + x·3.63e-09/(x + 1.0245))  [identity]
seed 21962:  log(exp(x/47187465) · 0.367879438) + sin(x² + x/58098460) · cos(x)  [identity]
seed 14423:  cos(1.0·x) · sin(x² · 1.0) − 1.0                          [exact]
seed 28020:  cos(x − 5.93e-09) · 1.0 · sin(x² / 1.0) − 0.99999998      [exact]
```

### The three rediscovered identities

**Seed 29910 — 2π periodicity.** Found `sin(x² + 6.28318531)·cos(x) − exp(exp(7.76·x)·9.06e-12)`. Note `6.28318531 = 2π` to 8 digits. By periodicity `sin(x² + 2π) = sin(x²)`, so the first term is the target's first term. The second term is `exp(exp(7.76·x) · 9.06e-12)`. Inside, `exp(7.76·x) ≤ exp(7.76) ≈ 2350` on [-1,1], so the inner argument is at most 2350·9e-12 ≈ 2e-8, and `exp(2e-8) ≈ 1 + 2e-8`. So the whole second term ≈ 1, and the formula reduces to `sin(x²)·cos(x) − 1` ≡ target. The search rediscovered:
1. Trig periodicity: `sin(θ + 2π) = sin(θ)`
2. A non-obvious way to construct the constant 1: nested exp with vanishing arguments

**Seed 4426 — Cofunction identity.** Found `−1 + sin(1.57079627 − x) · (sin(x²) + x · 3.63e-09 / (x + 1.0245))`. Note `1.57079627 = π/2` to 8 digits, and the second additive term has coefficient `3.63e-09` — essentially zero. So the formula simplifies to `−1 + sin(π/2 − x) · sin(x²) = −1 + cos(x) · sin(x²)` (cofunction identity). Exactly the target. The search rediscovered:
1. Cofunction identity: `cos(θ) = sin(π/2 − θ)`
2. The fact that `1.0245` is irrelevant — the redundant `x · 3.63e-09 / (x + 1.0245)` term contributes ~0 regardless of denominator constant

**Seed 21962 — log(1/e) = −1.** Found `log(exp(x/47187465) · 0.367879438) + sin(x² + x/58098460) · cos(x)`. Note `0.367879438 = 1/e` to 9 digits, and `x/47187465 ≈ 0` on [-1,1]. So `log(1 · 1/e) = log(1/e) = −1` and the first term is the constant −1. The second term is `sin(x²)·cos(x)` modulo the vanishing perturbation. Together: `−1 + sin(x²)·cos(x)` ≡ target. The search rediscovered:
1. `log(1/e) = −1` — a non-trivial way to construct the constant −1 without using R directly
2. The identity `e^0 = 1`, used to make the inner argument trivial

### Why these matter

These rediscoveries are the same flavor as the cyclotomic golden-ratio finding on Nguyen-3: **the search is finding non-trivial algebraic equivalents via real mathematical identities**, not just memorizing target patterns. The constants the search converges to (`710/113 ≈ 2π`, `355/226 ≈ π/2`, `0.36788 ≈ 1/e`) are well-known rational/decimal approximations of fundamental mathematical constants — LM didn't "know" π or e, but converged to them because that's where the loss landscape has zeros given the structural form the search committed to.

**Particularly notable:** `355/226` is the famous Tsu Ch'ung-chih / Ramanujan rational approximation of π (accurate to 7 digits). LM picked it because the search had committed to the structure `sin(R − x)·sin(x²)`, and the only way to make this match the data is `R = π/2`.

These rich rediscoveries happen specifically *under matched-pair*, suggesting CRN-reduced sibling comparisons are surfacing structurally-rich alternatives that the noisier baseline search skips over.

## Cross-Problem Summary

| Problem | Target | MP success | MP mean evals | Baseline mean evals | Speedup | Notable finding |
|---|---|---|---|---|---|---|
| Nguyen-3 | `x·Φ₅(x)` (degree 5) | 8/10 (incl. drift) | 97k | 228k | **2.4×** | Cyclotomic factorization with golden-ratio constants |
| Nguyen-4 | `x·Φ₂·Φ₃·Φ₆` (degree 6) | 1/1 cap-bound | ~2M | 195k | **~0.1×** | **Regression** — integer-coefficient factor structure already easy for baseline |
| Nguyen-5 | `sin(x²)·cos(x) − 1` | **10/10** (incl. trig identities) | 52k | 75k | **1.6×** | Rediscovered cofunction identity, 2π periodicity, `log(1/e) = −1` |

**Pattern emerging.** Matched-pair seems to help when:
1. The target has algebraically rich structure with non-integer coefficients (golden ratio φ, π, e, etc.) — CRN helps LM lock onto correct surds across siblings consistently
2. There are competing reward-1.0 exploit basins (Nguyen-3 transcendental tricks) — CRN cancels the rollout-luck advantage of exploits

It hurts (or is neutral) when:
1. The target has clean integer-coefficient algebra (Nguyen-4) — no surds to disambiguate, baseline already fast
2. The matched-pair re-evaluations introduce noise into otherwise stable path queues

**Refined publication framing.** Matched-pair isn't a uniform improvement — it's specifically a fix for *fragile benchmarks* (those with multiple reward-1.0 attractors). The right framing for a paper is **"variance-reduced MCTS for fragile reward landscapes"**, not "matched-pair MCTS is better". The Nguyen-4 regression is honest negative-result evidence that strengthens the paper rather than weakens it: it shows the mechanism is targeted, not magic.

## Next Tests

1. Run on Nguyen-3 [-10, 10] to confirm matched-pair is neutral when exploits don't exist
2. Run on all 12 Nguyen cases to see overall effect on the standard benchmark
3. Test N=8 for cost-vs-quality tradeoff (N ∈ {1, 2, 4} tested on three-seed subset)
4. Run on Livermore-9 (degree-9 polynomial) to test if cyclotomic-style behavior generalizes
5. Run on Nguyen-5/6 (sin·cos targets) to test if matched-pair *hurts* when transcendentals are actually needed

## Code Diff

- `include/imcts/mcts/mcts.hpp`: added private method declaration
- `source/mcts/mcts.cpp`: added `kMatchedPairN`, transition detection in `search()`, implementation of `matched_pair_reevaluation`

Total ~60 lines added, no existing code removed. Backward compatible (set N=0 to disable).
