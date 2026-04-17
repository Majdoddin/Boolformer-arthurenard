# Near-Constant Subtree Detection

Detecting subtrees whose output barely varies across the training data, and replacing them with constants. The literature calls this "numerical simplification" (Kinzett 2008 et seq.). Context: implementing canonicalization for MCTS-4-SR search tree (see `huang2025_notes.md`).

**Note on naming:** the replacement is with the subtree's mean, not necessarily zero. A subtree like `cos(x/1000000)` on x ∈ [-1,1] evaluates to ≈1 everywhere — near-constant but non-zero. Brush replaces it with a leaf constant of value 1. Only when the mean is near zero does the additive identity (`0+x→x`) actually shorten the tree.

## Papers covered

1. **Kinzett, Zhang, Johnston (2008)** — "Using Numerical Simplification to Control Bloat in Genetic Programming", SEAL 2008
2. **Kinzett, Zhang, Johnston (2010)** — "Investigation of simplification threshold and noise level", IEEE CEC 2010
3. **Johnston, Liddle, Zhang (2010)** — "A Relaxed Approach to Simplification in Genetic Programming", EuroGP 2010
4. **Javed, Gobet, Lane (2022)** — "Simplification of genetic programs: a literature survey", Data Mining and Knowledge Discovery
5. **Rockett (2020)** — permutation-test variant, referenced in Javed 2022

All closed-access on Springer/IEEE except Javed 2022 (Creative Commons, PDF saved locally).

**Citation correction:** the "relaxed approach" paper is Johnston, Liddle, Zhang (2010), NOT Kattan & Poli as I previously wrote. Kattan & Poli published a different paper at the same EuroGP 2010 conference.

## 1. Kinzett 2008 — the original algorithm

Applied online every few generations. Two **independent** per-node tests during a single tree-evaluation pass:

**(a) Contribution-to-parent test:** compute the numerical impact of subtree `c` on its parent `p`'s output (e.g., for `p = a + c`, contribution ≈ variation in `p` caused by `c`). If impact < threshold → delete subtree `c`.

**(b) Range test:** if `max(subtree_output) − min(subtree_output) < threshold` → replace subtree with constant = mean.

**Key point:** Brush implements only (b). The contribution test (a) is a separate mechanism they skipped. The contribution test catches cases where a subtree varies meaningfully on its own but its *effect on the parent* is negligible — different from "subtree output is flat."

**Guarantees:** none. Heuristic with empirical validation. Failure mode (confirmed): near-constant subtree replaced by mean can break upstream operators (division by near-zero → exact zero, log of near-zero → log(0), etc.).

**Runtime:** O(nodes × samples) for one forward pass, then O(nodes) to compute statistics. Same cost profile as Brush.

## 2. Kinzett 2010 — threshold investigation

Empirical rule (direct quote from abstract):
> "A lower bound for the optimum simplification threshold is equal to the noise level and an upper bound is five times the noise level."

No new algorithm, no guarantee. Pure regression over experimental runs on noisy problems. Only useful if you know or can estimate the noise level. **Does not apply to clean synthetic benchmarks** (Nguyen, Livermore) where noise = 0.

## 3. Johnston/Liddle/Zhang 2010 — "Relaxed Approach"

Two-stage pipeline.

**Proposer:** suggests a candidate simplification. Uses linear regression of the parent node's output against the child node's output over the training set. If the fit is good, a rule like `a × c ≈ constant × c` or `a + c ≈ constant` can be proposed.

**Evaluator (the "several levels up" part):** applies the candidate simplification, then re-evaluates the tree at an ancestor **k levels above** the proposal site. Computes MSE between simplified and original output at that ancestor. Accepts only if MSE < configurable threshold.

**What "several levels up" means operationally:**
- k=0 would be the local test (Brush-style, Kinzett range)
- k=1 checks the immediate parent
- k=full-depth checks the tree root
- The paper tested multiple k values on three classification datasets

**Guarantees:** still heuristic (MSE threshold at ancestor is an empirical knob). But **strictly stronger than Kinzett 2008 / Brush**: explicitly quantifies the effect at a chosen upstream point, catching the "division by near-zero becomes division by exact zero" failure mode.

**Runtime:** one extra partial forward pass per candidate, from replacement site up to k-th ancestor. O(k × samples) per candidate. Across the whole tree: O(nodes × k × samples).
- k=3 → roughly 3× Brush's cost
- k=full depth → O(nodes² × samples) worst case

**Direct conclusion from abstract:**
> "The success of linear regression is dataset dependent, looking further up the tree can catch ineffective simplifications, and CPU time can be significantly reduced while maintaining classification accuracy."

Their own experiments confirm that "several levels up" materially catches problems local checks miss.

## 4. Javed 2022 — survey

Does NOT crown a winner. Organises literature along orthogonal axes:
- online vs offline
- syntactic vs numerical vs semantic
- probabilistic vs all-individuals vs top-k%
- domain-specific vs domain-independent

**Relevant observations:**

- Numerical simplification is positioned between cheap-but-limited **syntactic/rule-based** and expensive-but-powerful **semantic simplification**. Semantic methods (Naoki 2009, Chu & Nguyen 2017/2020, Nguyen & Chu 2020) search for a semantically-equivalent smaller subtree — much more expensive than numerical, no better for upstream-operator failure cases.

- **Rockett (2020)** proposes statistical permutation tests on the subtree's contribution, validated on a hold-out set — a more principled variant of Kinzett that explicitly addresses over-pruning risk.

- Threshold selection "is not straightforward and depends on the nature of the application" (§6.3). Recommends further work on domain-independent approaches.

- **No method with mathematical guarantees identified as of 2022.** All numerical/semantic simplification remains heuristic with empirical validation.

## 5. Rockett 2020 — permutation tests

**What it does:** instead of a fixed variance/range threshold, runs a permutation test on the subtree's contribution. Null hypothesis: the subtree's output is irrelevant to the parent / tree output. Reject → keep subtree. Accept → prune.

**Why Javed still calls threshold "open":** Rockett transforms the threshold problem rather than solving it. You replace the numerical threshold with a significance level α, which is:
- still a hyperparameter (α=0.05 vs 0.01 changes aggressiveness)
- still application-dependent (depends on noise floor, signal scale)
- requires a hold-out set (not always available)

Compare to e.g. Bonferroni correction (controls family-wise error rate with a derivation). Rockett doesn't reach that principled level — it's statistical machinery on top of a still-heuristic choice.

**Applicability to fixed-sample benchmarks:**
- Clean synthetic benchmarks (Nguyen, 20 exact samples) have noise = 0 → no scale to calibrate against
- Only 20 samples → limited permutation power (at most 20! permutations, high test-statistic variance)
- Hold-out further reduces training signal — for a 5-order polynomial, 15 train points is thin
- Overall: Rockett's machinery adds overhead without adding information for our case

For our use case (MCTS-SR on clean benchmarks), a simple absolute threshold like Brush's 1e-5 works fine. The `sin(x/100000)` decorative noise terms have variance ~1e-10, orders of magnitude below the threshold.

## Comparison table

| Approach | k in ancestor check | Guarantee | Runtime (vs nodes × samples) | Catches upstream failure? |
|---|---|---|---|---|
| Kinzett 2008 range test (≈ Brush) | 0 (local) | heuristic | 1× | No |
| Kinzett 2008 contribution test | 1 (parent) | heuristic | 1× (same pass) | Partially (parent only) |
| Johnston 2010 relaxed, k=1 | 1 | heuristic | ~2× | Partially |
| Johnston 2010 relaxed, k=depth | full | heuristic | O(depth) | Yes (whole tree) |
| Rockett 2020 permutation | local, statistical | heuristic (α) | O(permutations × samples) | No (local only) |
| Semantic simplification | N/A | heuristic | much higher | Depends on method |

## What we implemented

Our `simplify_to_prefix()` in `mcts4sr/include/imcts/core/simplify.hpp` implements the Brush version of Kinzett 2008's range test — `variance(subtree_output) < threshold`. After LM fitting, runs one forward pass over the tree on training data, checks per-node variance, emits simplified prefix tokens. Controlled by `simplify_threshold` config (0 = disabled).

**Post-processing:** only additive identities (`0+x→x`, `x+0→x`, `x-0→x`). These clear dead constant nodes left by Brush replacements. Absolute error bounded by eps regardless of the other operand.

**NOT applied:** multiplicative identities (`1*x→x`, `0*x→0`) — their error `|coeff−identity| × |f(x)|` is unbounded for large `f(x)`. Example: `0.0001 * exp(exp(100*x))` is huge, not zero. Same failure mode as the division-by-near-zero edge case Johnston 2010 addresses. The variance check handles cases where the product is actually near-constant on the data.

**Key caveats:**
1. **Threshold:** configurable absolute variance threshold. Brush uses 1e-5. Relative thresholding is an open problem.
2. **NaN/Inf:** implicit safety — `NaN < threshold` is false, so affected subtrees are skipped.
3. **Requires LM-fitted constants:** the algorithm evaluates the tree on data, so constant values must be fitted first. Without fitting, all constants are 1.0 and pruning decisions would be wrong.

### TODO: ~1×f(x) dedup

`~1.0001 * f(x)` and `f(x)` are essentially the same formula but occupy different tree paths (`* R f` vs `f`). Brush doesn't simplify this — the product has the same variance as f(x), not near-constant. The multiplicative identity `1*x→x` was removed because error = |R-1| × |f(x)| is unbounded for large f(x). No known safe approach yet.

## Implementation plan — two-tier LM with risky-path check

This plan reflects the full design discussion and is meant to be self-explanatory for re-reading after context compaction.

### Motivation: three interacting concerns

**1. Structural-compensation vs decorative terms.** Two kinds of "bad" terms in a formula:
- **Structural-compensation** — wrong structure hidden by twisted LM-fitted constants. E.g., true formula is `x²+x`, search proposes `a·x² + b·x + c·sin(x)·cos(x)`, LM finds (a≈1, b≈1, c≈small) to hide the sin·cos term.
- **Decorative (dead) terms** — structurally added but LM pushes their coefficient to extremes to kill them. E.g., `true + sin(x/R)` with R fitted to 6M so that `sin(x/6M) ≈ 0` everywhere.

**Reduced LM (upstream's 100→50 change) helps filter the first kind** — LM can't converge the twisted-but-fitting constants under limited budget, so correct structures win by Occam pressure.

**But reduced LM undermines detecting the second kind** — decorative terms need LM to push coefficients to extremes. Without enough iterations, R stays at e.g. 100 instead of 6M, `sin(x/100)` still has meaningful variance, and our near-constant detector doesn't fire.

**2. Reliability of variance measurements depends on LM convergence.** Our method measures `variance(subtree_output)` on training data. With poorly-fitted constants, the current tree's behavior isn't the "optimal" behavior — pruning decisions made on the current tree may differ from those that would be made on the optimally-fitted tree. Same structure can be simplified in one formula but not another, depending on how much LM has converged each.

**3. Risky upstream operators.** The local variance check assumes upstream operators are Lipschitz-bounded (which bounds whole-tree error by subtree error × L^depth). For bounded ops (+, -, sin, cos, tanh) this holds. For `/`, `log`, `exp` the Lipschitz constant is unbounded — the `0.0001 * exp(exp(100x))` failure mode.

### Two-tier LM strategy

Decouple the search's reward signal from the simplification's reliability requirement:

**Tier 1 — Low LM (50 iter) — everywhere in search.** All rollouts, GP mutations, crossovers evaluated at LM=50. Standard upstream backpropagate and top-N insertion use the low-LM reward. This is what biases search toward simpler structures (point 1 above).

**Tier 2 — Warm-start LM refinement (≤50 extra iter) — triggered only on top-N qualification.** When a formula F's low-LM reward qualifies it for top-N at some node (the trigger condition), run the expensive canonicalization pipeline:

1. **Structure-hash cache check** — if this exact tree structure (ignoring constant values) has been canonicalized before, reuse cached canonical form + reward. Skip steps 2–6.
2. **Refine constants by warm-starting LM from Tier 1's fitted tree.** Pass Tier 1's `last_optimized_tree_` (which already carries the 50-iter constants) back into LM with an extra budget of ≤50 iterations. **This is warm-start, not fresh fit:** Tier 1's constants are the initial guess. In the quadratic regime, only ~3–10 iterations typically needed to reach double-precision floor. The `maxfev=50` cap is a safety net for ill-conditioned cases, not an expected cost.
3. **Numerical simplification** — Brush variance check on refined tree → simplified prefix tokens.
4. **Risky-path check** (see below) — for each candidate simplification beyond basic variance replacement, verify the upstream path doesn't amplify the approximation error.
5. **Commutative canonicalization** — C++ swap `+` and `*` args to canonical order.
6. **Re-evaluate canonical form under LOW LM (50 iter)** → fresh low-LM reward `R_C`. **Critical for consistency:** the canonical form's reward must be on the same scale as all other top-N entries (which got low-LM rewards).
7. **Propagate canonical form from root** with `R_C`.
8. **Cache** canonical form + `R_C` by `structure_hash(F)` for future hits.

**Cost accounting:**

| Candidate | Tier 1 | Tier 2 refine | Tier 2 eval (step 6) | Total LM iters |
|---|---|---|---|---|
| Non-qualifying (majority) | 50 | — | — | 50 |
| Qualifying, cache-hit | 50 | — | — | 50 |
| Qualifying, cache-miss | 50 | ~5–10 (cap 50) | 50 | ~105–110 |

Compare to the naive alternative "always fit at 100 iterations": **100 per candidate, no cache, no refinement gating.** Two-tier with warm-start is cheaper on average *and* gives better precision where it matters.

**Empirical finding (from unit tests):** Eigen's LM almost never hits the `maxfev`
budget. Even with `lm_iterations=1` (maxfev=3) on a 2-constant affine formula,
LM exits via `ftol` or `xtol` — it declares "converged" after 3 function
evaluations whether the fit is good or terrible. Only formulas with many
constants (≥6) and maxfev < ~10 actually trigger `TooManyFunctionEvaluation`.
With the real budget of 50 (maxfev=52), a 6-constant nonlinear formula
converged in 13 evals during warm-start.

**Implication for the "reduced LM filters bad structures" argument:** the
50-iter cap is not the mechanism that filters structurally-bad formulas.
LM exits via tolerance regardless of budget. What actually filters is the
**quality of the local minimum** — wrong structure → LM converges to a poor
local optimum → low reward. The budget matters only at the margin (constants
many OOM from 1.0, ill-conditioned Jacobians). For Nguyen-scale problems,
50 is ample.

This gives two propagations per qualifying formula:
- **First** (upstream behavior, unchanged): F propagated normally with its low-LM reward via `backpropagate` + `propagate`.
- **Second** (our addition): canonical form C propagated from root with its own low-LM reward.

### Risky-path check (replaces multiplicative identities safely)

The Lipschitz argument gives a cheap decision rule. At each candidate simplification (subtree N → replacement), walk up the ancestor chain once:

- **Path from N to root contains only bounded ops** (+, -, sin, cos, tanh, bounded *): local variance check is mathematically sound, accept the simplification.
- **Path contains any of `/`, `log`, `exp`, `sqrt`**: do a bottom-up incremental **root-deviation check** (see below).

The risky-path check subsumes what Kinzett 2008 contribution-to-parent would give us and extends beyond the parent to the full risky chain. Decision rule is simpler: "does the path from N to root contain any risky op?"

This also enables safely applying the multiplicative identity `1*f(x)→f(x)` when the path above is bounded. It handles our earlier concern: `1.0001 * sin(x)` → path above is just root, bounded, safe; `0.0001 * exp(exp(100x))` → path contains exp(exp(...)), risky, escalate to root check which rejects.

### Incremental bottom-up root-deviation check (when risky path detected)

Despite the "root-MSE" shorthand elsewhere in this file, **the acceptance statistic is `max_i |root_before[i] − root_after[i]|` across the N training samples** — max absolute deviation at the root, not MSE. This matches the benchmark's own `RMSE ≤ succ_error_tol · σ_y` criterion: if the maximum per-sample change is below that bound, no individual residual can have grown beyond tolerance.

Naive full-tree re-evaluation per candidate is O(nodes² × samples). The smart version is O(nodes × depth × samples):

- Precompute values[] for all nodes (one forward pass — already done)
- For each candidate simplification at node N:
  1. Apply the proposed replacement locally (update N's values column)
  2. Walk up the ancestor chain N → root, recomputing values at each ancestor using existing sibling values (which didn't change)
  3. Compute `max_i |root_before[i] − root_after[i]|`
  4. If max|Δ| < `sqrt(simplify_threshold) = succ_error_tol · σ_y`, accept; else revert
- O(depth × samples) per candidate

For typical trees (depth ≤ 6, ~15 nodes), this is negligible.

**Example traces:**
- `exp(cos(R·x/1000000))`: inner subtree ≈ 0, replacement → `exp(cos(0)) = e ≈ 2.718`. Original ≈ 2.718 with tiny wiggles. `max|Δ|` tiny → accept.
- `exp(1/near_zero_subtree)`: replacement → `exp(1/0) = exp(∞) = ∞`. `max|Δ| = ∞` → reject. Local variance check would have wrongly accepted.

### TODO — empirical-Lipschitz pre-check before root-deviation fallback

**Motivation.** The current binary rule "if chain contains any risky op, fall through to root-deviation check" is coarse: a risky op may be *empirically tame* on this dataset (e.g., `exp(x)` with argument range `[-1000, -500]` never exceeds `e⁻⁵⁰⁰`), and the full root-deviation re-evaluation is unnecessary work. For larger datasets (thousands of samples, deeper trees) this wasted fallback adds up.

**Refined rule.**

```
if chain contains any risky op:
    compute L_chain · subtree_ε  (cheap: uses already-cached values[])
    if L_chain · subtree_ε < succ_error_tol · σ_y:
        accept  (empirical Lipschitz bound proves safety)
    else:
        fall through to root-deviation check  (expensive empirical test)
else:
    accept  (statically bounded chain)
```

Three-tier cascade: static-bounded (free) → empirical-Lipschitz (cheap) → root-deviation (more expensive).

**Per-operator empirical Lipschitz from forward-pass values.**

For a node whose operator `f` sees input range `[a_emp, b_emp]` observed across training samples:

| Op | Analytical `L` on `[a, b]` | Cheap empirical variant |
|---|---|---|
| `sin, cos, tanh` | `≤ 1` | Static; no data needed |
| `+, −` | `= 1` | Static |
| `abs` | `= 1` | Static |
| `*` | `max(\|a\|, \|b\|)` per arg | Needs both operands' empirical ranges |
| `sqrt(x)` | `1 / (2√a_emp)` if `a_emp > 0`; else ∞ | From cached child values |
| `log(x)` | `1 / a_emp` if `a_emp > 0`; else ∞ | From cached child values |
| `exp(x)` | `exp(b_emp)` | From cached child values |
| `x / y` | bound by `max(\|y\|) / min(\|y\|)²` per operand-range scheme | Needs operand empirical ranges |

All ranges are already in `values[][]` from the existing forward pass — zero additional passes.

**Chain multiplication.** Walk ancestors `N → root`, multiply per-node Lipschitz constants: `L_chain = ∏_i L_i`. For unary ops, one factor per ancestor. For binary ops, take the Lipschitz w.r.t. the input that lies on the N-to-root path (the sibling input is unchanged, so its Lipschitz doesn't contribute to the chain). O(depth) multiplications per candidate.

**Acceptance condition.**
```
L_chain · sqrt(subtree_variance)  <  succ_error_tol · σ_y
```

This is a **sound but not tight** bound: if the chain Lipschitz proves safety, the root deviation is provably below `eps`. If it doesn't, fall through — the bound may just be loose.

**Practical pitfalls.**

1. **Infinite Lipschitz short-circuits correctly**. When `a_emp ≤ 0` for `log(x)` or `sqrt(x)`, or when an operand of `1/y` contains zero, the empirical L is infinite. The bound fails; fall through to root-deviation check. No special-casing needed.

2. **Binary ops need care**. For `x * y`, the relevant Lipschitz factor depends on which operand is on the N-to-root path. If N's chain goes through the left operand, the Lipschitz is `max|y|`; if through the right, `max|x|`. Encode in the chain-walk.

3. **Empirical ≠ true Lipschitz**. Training-set empirical range is a *lower bound* on the true operating range at future evaluations. For training-only decisions (our case), this is sound. If ever used for test-time guarantees, would need a safety margin — not our current concern.

4. **Chain product can overflow/underflow**. For chains with `exp` on large arguments, `L_chain` can exceed double range. Use log-space accumulation: `log L_chain = Σ log L_i`, compare `log L_chain + 0.5 · log(subtree_var) < log(eps)`. Robust.

5. **Cache the per-node Lipschitz alongside values**. Adds one `double` per node; amortized across all candidates using that node as an ancestor.

**When to implement.** After measuring the root-deviation fallback cost on realistic workloads (not Nguyen). If profiling shows fallback count × depth × samples dominates Tier 2 time, implement the pre-check. Expected speedup: 2–10× on Tier 2 depending on how often risky ops are empirically tame on the dataset.

**Expected value on different workloads.**
- **Clean synthetic (Nguyen)**: root-deviation already cheap, gain marginal.
- **Blackbox with large σ_y variation across x**: chain Lipschitz often provably safe, gain meaningful.
- **Chains with cascading exp/log/div**: rarely provably safe, both paths lead to fallback — no gain but no harm.

Zero risk: this is a *speed* optimization. Correctness is unchanged because unprovable cases still get the full root-deviation empirical test.

### Threshold — derived from benchmark pass criterion

Currently configured as a free parameter. Principled default:

```
simplify_threshold = (succ_error_tol × σ_y)²
```

Reasoning: benchmark passes if `NRMSE ≤ succ_error_tol` (typically 1e-6). NRMSE = √MSE / σ_y. Simplification changes predictions by some Δ; added MSE ≈ mean(Δ²); added NRMSE ≈ max|Δ| / σ_y. For simplification not to push a passing formula into failing:

```
max|Δ| < succ_error_tol × σ_y   ⟹   variance threshold ≈ (succ_error_tol × σ_y)²
```

For `succ_error_tol = 1e-6` and `σ_y ≈ 1`: threshold ≈ 1e-12. Much tighter than Brush's 1e-5. But safe: simplification only fires on truly-negligible subtrees, results are comparable to un-simplified for reward purposes.

Compute automatically in the evaluator; allow manual override via config for experimentation.

### Tolerances summary — what depends on input data

One root parameter, everything else derives mechanically:

| # | Parameter | σ_y-dependent? | Derivation |
|---|---|---|---|
| 1 | `simplify_threshold` (variance cutoff) | **Yes** | `(succ_error_tol · σ_y)²` |
| 2 | `eps` for near-zero identity (0+x→x, etc.) | Yes, derived | `√simplify_threshold = succ_error_tol · σ_y` |
| 3 | Risky-path root-deviation cutoff | Yes, derived | Same `eps` as #2 |
| 4 | LM ftol (Eigen default) | No (relative by construction) | — |
| 5 | Top-N qualification threshold | No (reward is already nrmse-normalized) | — |
| 6 | Tier 2 extra `maxfev` | No | — |
| 7 | Structure-hash cache | No tolerance | — |

**Implementation:** compute `simplify_threshold` once at Evaluator construction; every downstream cutoff reads it or its square root. Single knob, single derivation. Allow manual override for experimentation.

**Other input-data dependencies considered and rejected:**
- **`|ȳ|` (y mean)**: relevant only for FP cancellation when `|ȳ|/σ_y ≫ 1`. Fix is **centering** (subtract ȳ from residuals before LM), not a tolerance change. Documented in the Appendix.
- **N (sample count)**: variance is already per-sample, so N is absorbed.
- **x-ranges**: effect on `‖∇_θ f‖` is absorbed into σ_y via scale invariance.

### Summary of changes to current code

| Component | Change |
|---|---|
| `RegressorConfig` | Add `high_lm_iterations` (default 50, interpreted as **extra** budget on top of Tier 1); existing `lm_iterations` = Tier 1 (50) |
| `Evaluator` | Add `refine_last_high_lm()` that warm-starts LM from `last_optimized_tree_` with the extra budget and returns refined tree. Not a fresh fit — seeds from Tier 1's fitted constants. |
| `MCTS` | At backpropagate+propagate insertion, detect top-N qualification; trigger Tier 2 pipeline |
| `simplify.hpp` | Add risky-path detection; add incremental ancestor-chain re-evaluation for root-MSE check; extend identity application to safely allow `1*x→x` when path is bounded |
| Structure-hash cache | New: map `structure_hash → (canonical_prefix, R_C)`. Populate on Tier 2 execution, consult at Tier 2 entry. |
| Threshold | Auto-compute from `succ_error_tol` and dataset σ_y; allow override |

### Implementation order

1. **Benchmark current implementation under LM=50** — measure how often simplification fires on current setup (LM=50, threshold=1e-5). Establishes baseline for judging whether the additional complexity is worth it.
2. **Add auto-derived threshold** — simplest change, should make existing code more principled without needing two-tier.
3. **Add two-tier LM with top-N trigger + structure-hash cache** — the main architectural addition.
4. **Add risky-path check and extend multiplicative identity handling** — the "finally safe `1*x→x`" piece.
5. **Benchmark full pipeline** — compare iteration count to pass, wall-clock, simplification hit rate.

### TODO — per-node LM warm-start from ancestor's best rollout

Orthogonal optimization to the two-tier pipeline. Not required for correctness; pure speedup for LM convergence across rollouts.

**Idea.** MCTS expansion is **append-only** (`exp_tree.hpp:33`, `op_list_.push_back`). Child's prefix = parent's prefix + exactly one token. Any R that exists at byte position `i` in the parent's prefix sits at the **identical byte position and ordinal index** in every descendant. So ordinal indices of shared R's cannot drift — no tree-structural matching needed.

**Scheme.**
- Each `MCTSNode` stores `warm_constants: std::vector<float>` — the LM-fitted values (in prefix order) from that node's best-reward rollout so far.
- Before each rollout's LM fit, initialize the first `k` R's in the rolled-out tree from `parent.warm_constants[0..k]` (`k = min(parent's count, child's R count)`). Remaining R's default to 1.0.
- After LM fit, if the rollout's reward is the new best at the node, overwrite the node's `warm_constants` with the freshly fitted values.

**Why it helps.** LM from seed `1.0` spends `~log₁₀(|c*|)` iterations in the linear/damped phase before quadratic convergence. Warm-starting from an ancestor's ~correct value skips that. Expected savings: **3–10 iterations per qualifying constant per rollout**, larger on benchmarks with constants at non-O(1) scales (blackbox datasets; decorative `sin(x/6M)` terms).

**Caveats.**
- **Joint optimization means "same position" ≠ "same optimum".** A's best R₀ fitted in A's completion isn't guaranteed to equal B's best R₀ in B's different completion — the optima depend on the sibling subtrees. For early R's near the root (offsets, leading coefficients), optima are usually stable. For deep R's, less so.
- **Worst case is a slightly suboptimal initial guess**, which costs a few extra LM iterations. LM is monotone-convergent; warm-start within ~1 OOM of the optimum is always a strict win over default seeding.
- **Cost:** one `vec<float>` per node (~5 floats typical). O(100 KB) for a 25k-node tree — negligible.

**Position-based variant (safer).** Instead of ordinal index, key by byte position of the R-token in the prefix. Since MCTS is append-only, R's at position `≤ |A's prefix|` are structurally identical between A and any descendant. Only R's added later might semantically drift. This variant is defensively correct even if we ever add non-append-only operations.

**When to implement.** After step 5 (pipeline benchmarked). Measure: does LM hit `maxfev=50` frequently? If yes → warm-start likely valuable. If LM converges comfortably within budget → benefit is marginal, skip.

**When it's most valuable.**
- Non-O(1) constants (benchmarks outside Nguyen's clean polynomial range)
- Deep trees where R's accumulate orders of magnitude
- With Tier 2 warm-start already covering same-formula re-fits, this one covers parent→child reuse

### TODO — adaptive per-formula threshold

Currently (and in the plan above) `simplify_threshold` is **per-dataset, fixed once**: `(succ_error_tol · σ_y)²`. This is conservative — it assumes the formula is *exactly at* the success boundary, so any simplification error could tip it over.

In practice most formulas reaching Tier 2 fit much better than the boundary. If current RMSE `r_f` has slack `s = succ_error_tol · σ_y − r_f > 0`, simplification can eat that slack without affecting success. Adaptive rule:

```
simplify_threshold(F) = max(0, s_F)²
                     = max(0, succ_error_tol · σ_y − RMSE(F))²
```

- High-quality formulas (RMSE ≪ boundary) → looser threshold → more aggressive pruning → finds more decorative terms.
- Borderline formulas (RMSE near boundary) → threshold shrinks toward 0 → conservative, preserves existing fit.
- Failing formulas (RMSE > boundary) → threshold = 0 → no simplification. Correct: we don't canonicalize failures anyway.

**Practical pitfalls to avoid.**

1. **RMSE must be computed on the same data and after the same LM tier that generated the simplify candidate.** Mixing Tier 1's RMSE with Tier 2's post-refinement tree would under-estimate slack. Use the Tier-2-refined tree's RMSE explicitly.

2. **Don't use ratios or percentages.** `threshold = α × current_MSE` (multiplicative slack) is tempting but wrong: it scales threshold with current error, not with remaining budget. Near the boundary this *grows* threshold (bad); far from it, it shrinks (also bad).

3. **Monotonicity, not re-tuning.** An adaptive threshold per formula is fine; a threshold that changes *within* a single formula's simplification pass (e.g., after each candidate accepted, recompute slack) is dangerous — can cause cascading pruning where each accepted simplification enables the next, beyond any single-step safety margin. Rule: **compute slack once per Tier 2 entry, use it for all candidates in that entry's pass, then freeze**.

4. **Account for compounding across multiple candidates.** If `k` candidates each just-barely pass under `max|Δ| < s`, the combined deviation can be up to `k × s`. Either:
   - (a) Divide: allocate `s/k_max` per candidate (Bonferroni-like); or
   - (b) After each accepted candidate, recompute the current tree's RMSE and update slack. (This conflicts with pitfall #3 — *per-candidate* re-computation is fine; what's dangerous is making *the threshold itself* adaptive mid-pass without re-measuring.)
   
   Option (b) is correct but requires re-evaluating the tree after each accepted simplification. O(candidates × samples) per Tier 2 — still cheap.

5. **RMSE computation needs to be already available.** LM produced it as a byproduct of the fit — `workspace_.result()` has `y_pred`, subtract `y`, take norm. Zero new forward passes.

6. **Guard against degenerate σ_y**. The existing `σ_y floor of 1.0` (see Appendix) means adaptive threshold also degrades gracefully on constant-y datasets — slack just becomes `succ_error_tol − RMSE(F)` with σ_y=1.

7. **Interaction with cache.** Canonical form is cached keyed by `structure_hash`. But adaptive threshold means the *simplification decisions* depend on the formula's current fit quality, which differs across calls with the same structure hash. Either:
   - Cache per `(structure_hash, rounded_slack)` bucket; or
   - Accept that adaptive threshold may produce slightly different canonical forms across runs — which breaks dedup. Probably undesirable.

   Safer: stick with the fixed per-dataset threshold for the cache key and for "safe" canonicalization; use adaptive only for identifying *additional* simplifications beyond the safe set. Two passes: safe-canonical (cache-eligible) + aggressive (per-call, not cached).

**When to implement.** After step 5 (full pipeline benchmarked with fixed threshold). Measure how often formulas reaching Tier 2 have meaningful slack vs. sit at the boundary. If most have slack > 10× tighter than the threshold, adaptive pays off. If most are near-boundary, fixed threshold is fine.

**Expected benefit.** More aggressive decorative-term removal on clean benchmarks (Nguyen synthetic data, exact fits possible → large slack). Minor benefit on noisy/blackbox data where RMSE sits near the feasibility boundary.

### TODO — structure-gated global optimization for nonlinear constants

LM is a local optimizer. Constants that appear **inside** nonlinear functions
(`sin(R*x)`, `exp(R*x)`, `1/(R+x)`) have multi-modal loss landscapes where
LM gets stuck in wrong basins regardless of iteration budget. Constants that
appear **linearly** (`R*f(x) + R`) are always convex — LM is optimal.

Empirically confirmed: `R*sin(R*x)` fitting `3*sin(freq*x)` from seed (1,1)
fails for freq ≥ 5. LM converges in ~7 evals to a wrong basin and declares
"done" via ftol. 500 iterations produce the same wrong answer.

**Proposal.** Gate a short CMA-ES (or similar global optimizer) on the
formula's **structure**, not its reward:

```
if any R is inside a nonlinear unary op (sin/cos/exp/log/sqrt):
    run CMA-ES(20 gens) + LM     (~150-300 function evals)
else:
    run LM only                   (~3-13 function evals)
```

Reward-based gating (top-N) doesn't work: LM stuck in bad basin → low
reward → never qualifies → never gets global optimization. Structure-based
gating avoids this chicken-and-egg.

**Cost with caching.** Same structure hash → same CMA-ES result. Pay once
per unique structure. Estimated 2-5% total overhead.

**Detection reuses existing infrastructure.** Walk the tree once checking
if any R leaf is a descendant of a nonlinear op — same pattern as the
risky-path check in `simplify.hpp`.

**When to implement.** After benchmarking the current pipeline. Measure
how often the basin problem actually hurts results on Nguyen/blackbox
benchmarks. MCTS search-level diversity (many structurally different
formulas) may already compensate.

Full analysis: `papers/constant_optimization_local_minima.md`.

### What we get at the end

- **Decorative terms removed reliably** under well-fitted constants (Tier 2's high LM)
- **Near-constant non-zero subtrees** (`cos(x/1000000) → 1`) replaced with constants, and when those constants are 1 and in multiplicative context with a bounded path, further eliminated via safe `1*x→x`
- **Rewards consistent across tree** — all top-N entries on the low-LM scale
- **Dedup along canonical paths** — equivalent formulas converge to the same tree path regardless of how they were discovered
- **No incorrect simplification** — risky paths get root-MSE check; failures rejected

### What we explicitly leave out

- **Kinzett 2008 contribution-to-parent as a separate mechanism** — subsumed by risky-path-only root check, which is a strict superset.
- **Rockett 2020 permutation tests** — not applicable on 20-sample benchmarks (insufficient statistical power).
- **Johnston 2010 full k-levels-up for all candidates** — risky-path gating makes this only fire when needed.
- **Snapping constants to rationals** (0.618 → (√5−1)/2) — a separate problem (constant recognition), not negligible-subtree detection. No clear need for MCTS dedup; can revisit if evidence suggests.

## Appendix — LM convergence, ftol, and σ_y

Context: the simplify threshold is derived from the benchmark success criterion
`success ⟺ best_reward ≥ 1 − tol`, which is equivalent to `MSE ≤ (tol·σ_y)²`
(see `mcts4sr/source/evaluator/evaluator.cpp`). Whether LM can actually deliver
that precision depends on its stopping tolerances and iteration budget.
This appendix settles why it works — or doesn't — and what breaks it.

### Setup

- Reward: `r = 1 / (1 + nrmse)`, `nrmse = √MSE / σ_y`.
- Pass: `r ≥ 1 − tol` ⟺ `RMSE ≤ tol·σ_y / (1−tol) ≈ tol·σ_y`.
- σ_y: population std. dev. of y on the training set (floor 1.0).
- LM: Eigen's MINPACK port (`Eigen::LevenbergMarquardt`, `unsupported/Eigen/LevenbergMarquardt`).
  Default relative tolerance `ftol ≈ √eps_double ≈ 1.5·10⁻⁸`. Only `maxfev` is set
  in our code (`optimizer.cpp:88`, `= lm_iterations + 2`, 50 or 100).
- Constants seeded at 1.0 (`bridge.cpp:30`), then LM-optimized.

### Why relative ftol suffices (in infinite precision, well-posed case)

MINPACK's ftol termination guarantees bounded relative parameter error:
```
‖θ_final − θ*‖ / ‖θ*‖  ≲  ftol
```

Near the optimum, RMSE linearizes:
```
RMSE  ≈  ‖∇_θ f‖ · ‖θ_final − θ*‖  ≈  ‖∇_θ f‖ · ftol · ‖θ*‖
```

For a well-posed SR problem, `‖∇_θ f‖ · ‖θ*‖` is the sensitivity product that
determines the signal's variation — i.e., order `σ_y`:
```
RMSE  ~  ftol · σ_y
```

Required: `RMSE ≤ tol·σ_y`. Condition reduces to:
```
ftol  ≤  tol
```

σ_y cancels. For our defaults, `1.5·10⁻⁸ ≤ 10⁻⁶` ✓ — LM delivers ~100× tighter
precision than the benchmark demands, regardless of σ_y's absolute scale.

**Not a coincidence:** scale invariance of the well-posed SR problem. σ_y is a
joint consequence of `‖∇‖ · ‖θ*‖`, so both sides of the success inequality
scale identically under `y → c·y`.

### Iteration count (maxfev in context)

LM in the quadratic regime: `SSR_{k+1} ~ C · SSR_k²`. Reaching any target
`SSR_target` from the linear-regime ceiling takes `O(log log(1/ε))` iterations
— practically 3–5.

Linear-regime iteration count:
```
k  ~  log(SSR₀ / SSR_min) / log(1/r)
```
Both `SSR₀ ~ N·σ_y²` (formula at seed) and `SSR_min ~ N·(tol·σ_y)²` (passing
threshold) scale with `σ_y²`, so the ratio is `1/tol²` — **σ_y-independent**.

**What maxfev = 50 actually caps is seed-to-truth distance**, not σ_y:
- Seed `c = 1`, true `c* ~ O(1)` → few iterations
- Seed `c = 1`, true `c* = 10⁻⁶` → ~`log(10⁶)/log(1/r)` ≈ 20–30 iterations
  crossing scales in the linear/damped phase, before quadratic kicks in
- maxfev = 50 covers this with headroom; 100 is safer when constants cross many
  orders of magnitude

So the 50/100 limit is orthogonal to σ_y. It's a budget against bad seed choice
and ill-conditioning, not variance.

### When the rescue breaks

The `σ_y cancels` argument assumes `‖∇_θ f‖ · ‖θ*‖ ~ σ_y`. Four cases where this fails:

1. **Large y-offset** (`|ȳ| ≫ σ_y`). Example: `y = 1000 + 10⁻³·f(x)`.
   - σ_y = 10⁻³ but constants include the 1000 offset with `∂f/∂offset = 1`.
   - Required absolute precision on the offset: `tol·σ_y = 10⁻⁹`.
   - Relative precision on the offset: `10⁻¹²` — below Eigen default `ftol`.
   - **Fix: centering** (subtract ȳ), not relative tolerance.

2. **Floating-point cancellation** in `ŷ − y` when y has large magnitude.
   With `double`'s ~16 digits, residuals below `|y|·eps` are noise.
   When `σ_y < |ȳ|·√eps`, LM minimizes noise, not the signal.
   **Not fixable in the optimizer**; requires higher precision or centering.

3. **Ill-conditioned Jacobian** (`κ(JᵀJ) ≫ 1`). LM's damping slows progress;
   linearization bound `RMSE ≈ ‖∇‖·Δθ` weakens. Ftol-parameter-precision
   guarantee degrades by `κ`. Independent of σ_y.

4. **Non-convexity**. LM converges to a local, not global, minimum.
   Infinite precision and tight ftol don't help. Formula-structure problem.

### Practical implications for our code

- **Default Eigen ftol is adequate** for Nguyen-style benchmarks where
  σ_y ~ O(1) and |ȳ|/σ_y is modest. No need to tighten ftol.
- **maxfev = 50 is tight** when true constants are far from 1.0; 100 is
  defensibly safer. The two-tier plan (LM_low=50 for search, LM_high=100 for
  simplification-qualifying candidates) uses this asymmetry deliberately:
  search can tolerate sub-optimal fits, simplification decisions cannot.
- **No need to scale residuals by σ_y inside LM** — relative tolerances make
  this a no-op. (Briefly considered, rejected after inspecting the code.)
- **Centering y is the real defense** against large-offset cases.
  `mcts4sr/source/evaluator/evaluator.cpp:21-24` computes mean + variance but
  only uses variance for σ_y. If benchmarks with large `|ȳ|/σ_y` appear
  (some blackbox datasets may), subtract ȳ before computing residuals.
- **σ_y floor of 1.0** (`evaluator.cpp:24`): when `σ_y < 1e-10`, the code
  sets `σ_y = 1.0`. This implicitly switches the benchmark criterion from
  relative (`RMSE < tol·σ_y`) to absolute (`RMSE < tol`). Without the floor,
  `tol·σ_y → 0` as `σ_y → 0`, demanding sub-double-precision accuracy
  (e.g., threshold `(1e-6·1e-11)² = 1e-34`) — the benchmark becomes
  unsolvable for floating-point reasons, not formula quality. The floor
  avoids this by capping the threshold at `(tol·1.0)² = tol²`. Not standard
  in the literature; a pragmatic guard for a degenerate case absent from
  standard benchmark suites.

### One-line verdict

> Relative ftol gives relative parameter precision; natural SR problems have
> `σ_y ∝ ‖∇_θ f‖·‖θ*‖`; therefore absolute RMSE scales with σ_y automatically.
> The "fixed ftol" applies to a dimensionless quantity, and dimensionless
> quantities don't need to know σ_y. Breaks under large y-offset, FP noise,
> ill-conditioning, or non-convexity — none of which are σ_y-specific.

## Sources

- [Javed, Gobet, Lane 2022](https://link.springer.com/article/10.1007/s10618-022-00830-7) — full PDF in `papers/javed2022_simplification_survey.pdf`
- [Kinzett, Zhang, Johnston 2008](https://link.springer.com/chapter/10.1007/978-3-540-89694-4_50) — SEAL 2008
- [Kinzett, Zhang, Johnston 2010](https://ieeexplore.ieee.org/document/5586181/) — IEEE CEC 2010
- [Johnston, Liddle, Zhang 2010](https://link.springer.com/chapter/10.1007/978-3-642-12148-7_10) — EuroGP 2010 ("Relaxed Approach")
- [Kinzett, Johnston, Zhang 2009](https://link.springer.com/article/10.1007/s12065-009-0029-9) — Evolutionary Intelligence (journal version of 2008)
- [EuroGP 2010 TOC](https://dblp.uni-trier.de/db/conf/eurogp/eurogp2010.html) — confirms authorship
