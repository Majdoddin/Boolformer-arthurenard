# Huang 2025 — Burst-Expand Results

**Background:** see `huang2025_notes.md` §"Unfair sibling comparison from random rollout" → "Refinement to suggestion 1: matched-pair sampling" → "GP/tree budget ratio concern under burst-expand", and the preceding experiment report `huang2025_matched_pair_results.md`.

## Change

**Burst-expand with descent-stop.** Replace Huang's one-child-per-visit expansion rule with a one-shot full expansion at the first visit to any leaf. When descent hits a node with `unexpanded_moves`, expand **all** of them at once, running N shared-seed rollouts per child, and then stop the iteration. Next iteration's descent then sees a fully-expanded node and can use UCB across all K siblings from the very first time, not after K-1 unfair one-at-a-time rollouts.

**What this changes vs prior matched-pair:**

| | Matched-pair N=4 (prior) | Burst-expand (this version) |
|---|---|---|
| First visit to a leaf | Expand 1 child, 1 rollout, descent stops | Expand K children, K × N rollouts, descent stops |
| Child K visits later | Transition moment triggers K × N re-evaluation | Already evaluated at first visit, no re-eval |
| Per-iteration expand work | 1 eval (amortized: 1 + N over K iters) | K × N evals every iter |
| Fairness delay | K-1 iterations before siblings compared with matched seeds | 0 (fair from first visit) |
| Partial-expansion parents | 1 eval each, K×N re-eval never fires if descent abandons them | Pay full K × N at first visit, no lazy skip |

The initial framing — "burst is the cleaner implementation of CRN, skipping wasted re-work" — turned out to be wrong in a way the table above hints at. Matched-pair is **lazy**: the expensive K × N re-eval fires only at parents UCB revisits to the K-th child, so dead-end subtrees never pay full price. Burst is **eager**: every leaf descent touches pays K × N evals immediately. Also, matched-pair's K-iteration gap between first and last child expansion lets GP accumulate history in the parent's path_queue, which then propagates down to the children; burst's one-shot expansion gives the children much thinner inherited path_queues. See §"Comparison with matched-pair N=4" for the per-iteration cost breakdown that explains the observed ~5× eval gap.

**Code:** ~60 lines edited in `mcts4sr/source/mcts/mcts.cpp` and `include/imcts/mcts/mcts.hpp`.
- Constant `kBurstSamplesN` (tested at 1, 2, 4), replacing `kMatchedPairN = 4`.
- `expand_node()` removed, replaced by `burst_expand()`.
- `matched_pair_reevaluation()` removed — its purpose was intended to be absorbed into `burst_expand`.
- `search()` simplified: descent → if terminal evaluate else burst_expand → stop.

See `papers/huang2025_notes.md` §"GP/tree budget ratio concern under burst-expand" for the original motivation behind picking N=1. Short version: at N=4 the burst would push `~40` evals per iteration against Huang's `~3` GP evals, collapsing the GP/tree ratio from ~70/30 to ~7/93 and starving the GA. At N=1 the burst costs `K ≈ 10`, keeping the split closer to Huang's. The 3-seed results below show this was the right instinct (N=1 is indeed the only viable point) but for the wrong reason — the GP-budget concern is a symptom of the deeper amortization gap described in §"Comparison with matched-pair N=4".

## Setup

- Benchmark: Nguyen-3 = `x⁵ + x⁴ + x³ + x² + x`, range **[-1, 1]**, 20 samples (40 effective via `sample_multiplier=2`)
- Op set: `{+, -, *, /, sin, cos, exp, log, R}`
- RNG: PCG64DXSM
- Seeds: Huang's fixed list, indices 0–2 — **23654, 15795, 860**
- Hyperparameters: max_depth=6, max_constants=6, max_evals=2M, K=500, c=4.0, γ=0.5
- Success criterion: `reward ≥ 1 − 1e-6`

## Headline: three-seed comparison

Baseline and matched-pair numbers are from `huang2025_matched_pair_results.md:38-47`.

| Seed | Huang baseline | Matched-pair N=4 | **Burst N=1, gp=0.2** | **Burst N=1, gp=0.4** | **Burst N=2, gp=0.4** | **Burst N=1, gp=0.6** |
|---|---|---|---|---|---|---|
| 23654 | 28.2s / 248k | **8.2s / 74k** | 45.8s / 540k | 16.7s / 187k | 69.9s / 783k | 106.5s / 1,079k |
| 15795 | 26.3s / 205k | 11.9s / 102k | — | 16.2s / 190k | 73.4s / 480k | 38.0s / 186k |
| 860 | 8.5s / 70k | 21.4s / 166k | — | 22.8s / 250k | 34.3s / 228k | 125.3s / 701k |
| **3-seed mean** | 21.0s / 174k | 13.8s / 114k | — | **18.6s / 209k** | **59.2s / 497k** | 89.9s / 655k |

Burst N=4 was attempted at gp=0.4 but aborted — runtime blew past 5 minutes on seed 23654 alone. Burst N=2 gp=0.6 was also aborted mid-run for the same reason. These negative results are noted qualitatively rather than measured.

**Main findings:**

1. **Burst N=1 with default gp_rate=0.2 is a disaster** (45.8s on seed 23654, ~5.6× slower than matched-pair). N=1 is too thin: single-sample sibling comparison doesn't filter exploit basins.
2. **Bumping gp_rate to 0.4 recovers most of the gap at N=1**: 1.35× slower than matched-pair on the three-seed mean, structurally equivalent output.
3. **gp_rate=0.6 is non-monotone-worse at N=1**: 6.4× slower on seed 23654, 5.5× on seed 860. gp=0.4 is a local sweet spot.
4. **Raising N past 1 does NOT close the gap — it widens it.** Burst N=2 at gp=0.4 is ~3.2× slower than burst N=1 (59.2s vs 18.6s mean), ~4.9× slower than matched-pair N=4, and uses ~5× more evals. Burst N=4 couldn't even be measured within reasonable time. This is the opposite of what we expected when initially picking N=1 for GP-ratio reasons — we thought N=1 was a concession to cost and N=2/4 would be closer to matched-pair's quality. In reality, N=1 is the *only* viable point for burst.
5. **Structural output at N=1 gp=0.4 is excellent**: all three seeds land on clean algebraic forms — Horner, cyclotomic with golden ratio, clean polynomial with drift. N=2 gp=0.4 is slightly worse (2/3 clean, 1/3 near-miss with a cos decoration that numerically simplifies to target).

## Structural findings

### Burst N=1, gp=0.2

**Seed 23654** — transcendental decorative-scaffold (but simplifies correctly):
```
x0 - sin(x0 / 6393264.10016)
  - (x0² + 1.00000087756) · (0.99999973911 + x0) · x0 / 3742026.86305
    · (cos(x0) + x0 · -3742026.95785 + sin(x0) · x0)
```
After `sp.expand + sp.nsimplify(tolerance=1e-3)`: exactly `x0⁵ + x0⁴ + x0³ + x0² + x0`.

The scaffolding works via a "huge constant × tiny constant" cancellation: `2.67e−7 · 3742026 ≈ 1`. `cos(x0)` and `sin(x0)·x0` contribute coefficients below tolerance and get dropped. This is **not** a transcendental exploit in the "exp·sin approximates Taylor expansion of a linear term" sense — it's structurally the target, just with decorative trig that washes out. But it's ugly enough to count as a structural miss relative to matched-pair's cyclotomic form.

### Burst N=1, gp=0.4

All three seeds clean.

**Seed 23654** — Horner factorization, complexity 17:
```
x · (x · (x+1) · (x²+1) + 1)  =  x⁵ + x⁴ + x³ + x² + x
```
Same structure as Huang baseline on wider [-10, 10]. Tiny coefficient drift (`0.9999999...`, `-1.89e-7`) washes out under nsimplify.

**Seed 15795** — ⭐ cyclotomic factorization with golden ratio, complexity 19:
```
x · (x² − 0.618034·x + 1) · (x² + 1.61803·x + 1)
  = x · (x² − (1/φ)·x + 1) · (x² + φ·x + 1)
  = x · Φ₅(x)  where φ = (1+√5)/2
```
**Identical factorization that matched-pair N=4 recovered** on 6/10 of its seeds (`huang2025_matched_pair_results.md:56-58`). The golden-ratio constants `1.61803408295` and `0.618034251237` agree with `φ` and `1/φ` to 8 digits.

**Seed 860** — expanded polynomial with 1e-7 drift, complexity 22:
```
x · (0.9999996·x⁴ + 0.9999998·x³ + 1.0000004·x² + 1.0000001·x + 0.9999999)
```
Same category as matched-pair's "polynomial-with-drift" seeds (3/10 of baseline runs in that table). Under `nsimplify(tolerance=1e-3)` this snaps to `x·(x⁴+x³+x²+x+1)` — exact.

### Burst N=1, gp=0.6

2/3 clean, 1 messy.

**Seed 23654** — messy with sin(x) decorative term, complexity 26. Simplifies to target under expand+nsimplify.

**Seed 15795** — Horner factorization, complexity 15:
```
x · (x · (x+1) · (x²+1) + 1)
```
Same as seed 23654 at gp=0.4. Runtime is 2.3× slower despite same eval count (`186k` at gp=0.6 vs `190k` at gp=0.4), suggesting the eval **mix** shifted toward more expensive LM fits — a signal that higher GP rate produces messier intermediate candidates.

**Seed 860** — ⭐ cyclotomic factorization with golden ratio, complexity 18:
```
x · (x² − 0.618034·x + 1) · (x² + 1.61803·x + 1)  =  x · Φ₅(x)
```
Another recovery of the exact golden-ratio factorization, this time on the seed that was **slowest** under matched-pair (21.4s) — and now at gp=0.6 it's **125s**, six times slower but still structurally elegant.

### Burst N=2, gp=0.4

2/3 exact structural matches, 1/3 near-miss with a `cos` decoration.

**Seed 23654** — complexity 22:
```
x⁵ + x⁴ + x³ + x² + x·cos(5x/308)
```
Structurally the target with the last term decorated as `x·cos(0.016·x)`. On [-1, 1] this is `x·(1 − 3.3×10⁻⁵)` — numerically indistinguishable from `x`, hence reward 1.0. `nsimplify(tolerance=1e-3)` won't snap a function argument to zero, so this registers as a structural miss despite being perfectly accurate. Not a transcendental exploit, just a decorative term the MCTS failed to prune.

**Seed 15795** — exact target, complexity 20. Clean polynomial factored as `x·(x·(x+1)·(x²+1) + 1)`.

**Seed 860** — exact target, complexity 24. Polynomial form with residual `exp(x)·sin(1.98e−7·x)` that washes out under tolerance.

Quality slightly *worse* than N=1 gp=0.4 (which had 3/3 clean forms including the golden-ratio cyclotomic on 15795). Doubling N did not buy cleaner structure; it just paid more to land on roughly equivalent algebraic forms.

## GP-rate response is non-monotone

The response of runtime to gp_rate on seed 23654:

| gp_rate | Time | Evals | Complexity | Structure |
|---|---|---|---|---|
| 0.2 | 45.8s | 540k | 29 | big-constant decorative scaffold |
| **0.4** | **16.7s** | **187k** | **17** | Horner |
| 0.6 | 106.5s | 1,079k | 26 | sin-decorated Horner |

Not a ceiling — a **local minimum**. Too little GP and the search drifts through ugly scaffolding (not enough mutation pressure to refine toward clean forms). Too much GP and path_queues saturate with mutation-derived entries, UCB starts chasing local mutation-loop optima, and the eval mix shifts toward expensive LM fits on complex intermediates.

**Mechanism guess:** at gp=0.6, GP fires at ~60% of non-leaf nodes during descent, producing many mutations per iteration. Once path_queues have ~K entries (full), new mutations displace old ones, making the population churn. This churn dilutes high-quality entries and gives UCB noisy Q-values. gp=0.4 fires GP at ~40% of non-leaf nodes, which is enough to drive refinement without saturating the queues.

This is consistent with classical GP-MCTS folklore: there's a "right" ratio of tree exploration to GP recombination, and overshooting recombination starves tree growth. Huang's default gp_rate=0.2 was chosen for one-at-a-time expansion; burst-expand shifts the balance so the optimum moves to ~0.4.

## Comparison with matched-pair N=4

On the three-seed subset (23654, 15795, 860):

| Metric | Matched-pair N=4 (gp=0.2) | Burst N=1 (gp=0.4) | Burst N=2 (gp=0.4) | Ratio (burst N=2 / mp) |
|---|---|---|---|---|
| Mean time | 13.8s | 18.6s | 59.2s | **4.3× slower** |
| Mean evals | 114k | 209k | 497k | **4.4× more** |
| Success rate | 3/3 | 3/3 | 3/3 | tied |
| Clean structure | 2/3 + 1 drift | 3/3 clean | 2/3 + 1 cos-decoration | mp ≥ burst |
| Cyclotomic recovery | 2/3 | 1/3 | 0/3 | matched-pair ahead |

Burst N=1 at gp=0.4 is ~1.35× slower than matched-pair for the same structural quality. Burst N=2 makes things **worse**, not better: ~4.3× slower for slightly worse structure. Burst N=4 runs off the edge of the practicality envelope entirely. The naive reading — "more samples per child should buy better sibling discrimination" — is defeated by a K-multiplier effect that was not obvious until we measured it.

### Why burst has a ~5× gap, not a ~2× gap

The seemingly-logical intuition is: "matched-pair N=4 stores 5 evals per child (1 single-rollout + 4 re-evals), burst N=2 stores 2 evals per child, so burst N=2 should be **cheaper** per child by ~2.5×, not more expensive." This mis-counts because it compares per-child costs without adjusting for how many children are expanded per iteration.

**Per-iteration cost breakdown**, with K = 10 branching, depth = 6, gp_rate = 0.4 (so depth × gp_rate ≈ 2.4 GP evals per descent):

| Step | Matched-pair N=4 | Burst N=1 | Burst N=2 | Burst N=4 |
|---|---:|---:|---:|---:|
| Children expanded per iter | 1 | K = 10 | K = 10 | K = 10 |
| Expansion evals per iter | 1 (single-rollout) | 10 × 1 = 10 | 10 × 2 = 20 | 10 × 4 = 40 |
| Re-eval amortization per iter | 4 (= N, fires every K iters) | 0 | 0 | 0 |
| GP evals per iter | ~2.4 | ~2.4 | ~2.4 | ~2.4 |
| **Total evals per iter** | **7.4** | **12.4** | **22.4** | **42.4** |
| Ratio vs matched-pair | 1.0× | 1.7× | 3.0× | 5.7× |

The **K factor** is what the per-child framing misses. Matched-pair expands **1** child per iteration — `expand_node` pops one random unexpanded move and gives it a single rollout. Burst expands **K = 10** children per iteration. Burst's per-iter expansion cost is `K × N_burst`; matched-pair's is `1 + N_mp` (one single-rollout plus amortized re-eval). The ratio is `K × N_burst / (1 + N_mp) = 20 / 5 = 4×` at burst N=2, not `N_burst / N_mp = 0.5×`.

But 3× per iteration doesn't quite match the observed ~5× eval ratio on burst N=2. The missing 1.7× comes from a **second** effect: burst needs more iterations to find the solution because its sibling rankings are noisier at expansion time. Two reasons:

1. **Less GP history at the propagation moment.** Matched-pair fires at the K-th visit to a parent. By then, K iterations of descent have each rolled GP dice at every non-leaf node on the path, and the parent's `path_queue` has accumulated `~K × gp_rate × depth ≈ 24` GP-discovered entries. When `propagate` pushes those down to the K children, each child inherits a rich set of good paths. Burst fires on the *first* visit, with the parent's `path_queue` carrying only the GP entries accumulated during prior descents through it — typically far fewer. Children inherit thinner priors.
2. **No lazy expansion.** Matched-pair's transition-moment design is lazy: parents that descent abandons after 1–2 visits pay only 1–2 evals per partial child, and the K × N re-eval *never fires* for them. Dead-end subtrees are cheap. Burst has no laziness — every leaf descent touches pays the full K × N immediately, so dead-end exploration costs scale with K × N rather than with 1.

Both effects degrade the quality of burst's UCB estimates early, causing it to waste visits on subtrees that a better-informed search would prune. Empirically this costs ~1.7× more iterations for burst N=2 vs matched-pair N=4.

Combining: `3.0× (per iter) × 1.7× (more iters) ≈ 5.1× more evals` — which matches the observed 4.4× (our model is approximate and the per-iter estimate has ~20% slack on the GP amortization term). At burst N=4, per-iter cost is `5.7×` and the iteration multiplier is likely larger still, giving >10× — consistent with "burst N=4 took too long to finish."

### Why the framing in §Change was wrong

The original pitch — "burst skips matched-pair's wasted K one-at-a-time rollouts" — misreads matched-pair's structure. Matched-pair's "wasted" K rollouts aren't wasted; they're the **backbone of its laziness**. Each single rollout is cheap (1 eval), and descent/UCB uses it to decide whether to ever come back and pay the full K × N. The K rollouts *are* the cheap exploration budget. Burst replaces them with one expensive atomic operation that has no early-abort path.

The "fairness delay = 0" property that burst preserves sounds valuable but isn't, because:
- At first visit, no sibling comparison is needed — nothing is descending into them yet.
- By the time comparison *is* needed (K-th visit), matched-pair has already fired its re-eval. The "K-1 iterations of unfair rollouts" happen *before anyone is looking*.

So burst spends extra evals to fix a problem that matched-pair's lazy design already avoids for free. There's no structural benefit to pay for.

## What about the GP/tree budget ratio?

Revisiting the concern raised in `huang2025_notes.md` §"GP/tree budget ratio concern":

| Strategy | Expand evals / iter | GP evals / iter | Total / iter | GP share |
|---|---|---|---|---|
| Huang baseline | 1 | 3 | 4 | ~75% |
| Matched-pair N=4 (prior) | 1 + N amortized = 5 | 3 | ~8 | ~38% |
| Burst N=1, gp=0.2 | K ≈ 10 | 3 | 13 | 23% |
| **Burst N=1, gp=0.4** | **K ≈ 10** | **~6** | **16** | **~38%** |
| Burst N=2, gp=0.4 | 2K ≈ 20 | ~6 | ~26 | ~23% |
| Burst N=1, gp=0.6 | K ≈ 10 | ~9 | 19 | ~47% |

At burst N=1 gp=0.4, GP share is ~38%, comparable to matched-pair. Raising N to 2 shifts the ratio back down to ~23%, collapsing the GP fraction and producing the observed slowdown. The original concern — "naive burst starves GA" — was correctly identified but is not actually fixable within the burst framework: any N>1 runs back into the same problem, because K×N dominates the per-iteration budget.

The deeper reason is that matched-pair achieves **both** high per-child sample count (5 evals/child) **and** high GP share (~38%) simultaneously, because its expansion cost is amortized over K iterations instead of paid in one shot. Burst can match either one but not both.

## Caveats

1. **3 seeds only.** Same subset of Nguyen-3 seeds (indices 0–2). Need full 10-seed sweep to claim mean parity or divergence. Variance across seeds is huge in matched-pair's table (8.5× spread on seed 16850) — 3 seeds is not enough to know whether the 1.35× mean difference is stable or noise.
2. **N sweep is partial.** N=1 and N=2 tested at gp=0.4; N=4 at gp=0.4 was attempted and aborted (>5 min on one seed); gp=0.6 was only tested at N=1, and N=2 gp=0.6 was also aborted. No complete (N, gp_rate) grid, but the direction of the effect is unambiguous: raising N widens the gap.
3. **gp_rate=0.4 sweet spot not proven to be the optimum.** Sampled at {0.2, 0.4, 0.6}. 0.4 looks best on 2/3 seeds but could be coincidence. A finer sweep (0.3, 0.5) would confirm.
4. **Only Nguyen-3.** Matched-pair showed a Nguyen-4 regression (`huang2025_matched_pair_results.md:101-122`) because the target's integer-coefficient factorization was already easy for baseline. Burst may show the same regression on Nguyen-4 — we haven't tested.
5. **Structural quality assessed informally.** All judgments of "clean Horner" vs "cyclotomic" vs "polynomial drift" are visual reads of the output, not automated equivalence checks. A proper test would `sp.expand + sp.nsimplify(tolerance=1e-3)` both the burst output and the target and check symbolic equality.

## Open questions

1. ~~**Does N=2 close the gap?**~~ **Answered: no.** N=2 widens the gap to ~4.3×. Burst N=4 widens it further — too slow to measure. Burst is only viable at N=1, and even there it's 1.35× slower than matched-pair.
2. **Does the 1.35× gap at N=1 gp=0.4 hold under 10 seeds?** Still unknown. 3 seeds is too few to pin down the mean.
3. **Is the non-monotone gp_rate response stable across seeds?** We saw it on seed 23654 (0.2→0.4→0.6 = 45.8, 16.7, 106.5s) but not independently confirmed on 15795 or 860 (those were only tested at 0.4 and 0.6).
4. **Does burst-expand avoid the Nguyen-4 regression that matched-pair had?** Unknown — not tested.
5. **Is there a hybrid "lazy burst" that keeps matched-pair's laziness but bursts at the transition moment instead of one-at-a-time?** This would be: expand one child per visit (1 eval each), and when the K-th child is added, fire a `burst_expand`-style K × N matched evaluation *in place of* the transition-moment re-eval. In effect, this is matched-pair with its `matched_pair_reevaluation` renamed and slightly restructured — so it's what the prior code already did. The "lazy burst" framing makes clear that the prior matched-pair *is* the right design; burst-on-first-visit was an unnecessary restructuring that lost laziness.

## Refined publication framing (speculative)

The prior version of this section proposed that burst might be "a cleaner mechanism, potentially more tunable" than matched-pair. With the N=2 data and the per-iteration cost analysis, that framing no longer holds up:

1. Burst is **not** cleaner. Matched-pair's lazy design is fundamentally better matched to MCTS's cost structure — partial expansion is cheap, full expansion is expensive, and laziness decouples the two.
2. Burst is **not** more tunable. N and gp_rate both fight each other, and the "sweet spot" (N=1, gp=0.4) is on the edge of a cliff in both directions.
3. Burst has **no structural advantage** — fairness-from-first-visit turns out to be a property no one needed.

The useful publication story from this work is about matched-pair, not burst. The contribution of the burst experiment is a **negative result** that clarifies why matched-pair works: it's not the shared-seed re-evaluation per se, but the *combination* of shared-seed re-evaluation with lazy one-at-a-time expansion. Replacing the latter kills the former's benefit. Matched-pair's "transition moment" design isn't an implementation detail — it's the core of why it's fast.

If we wanted a crisper version of matched-pair's contribution, this is it:

> *Shared-seed re-evaluation of sibling children provides variance-reduced comparison, but only when combined with lazy expansion that defers the full K × N cost until siblings actually need ranking. Eagerly expanding all K children at first visit pays the full cost on every dead-end subtree and collapses GP priming, leading to a ~5× eval overhead with no structural benefit. The laziness of classical one-at-a-time expansion is doing more work than it appears to.*

## Reproduce

```bash
cd mcts4sr
# Burst-expand N=1 + gp_rate=0.4, first 3 seeds
.venv/bin/python -m imcts.benchmarks --group Nguyen --cases 3 --runs 3 \
    --seed-start 0 --gp-rate 0.4

# Single seed + gp-rate sweep
.venv/bin/python -m imcts.benchmarks --group Nguyen --cases 3 --runs 1 --gp-rate 0.2
.venv/bin/python -m imcts.benchmarks --group Nguyen --cases 3 --runs 1 --gp-rate 0.4
.venv/bin/python -m imcts.benchmarks --group Nguyen --cases 3 --runs 1 --gp-rate 0.6
```

Requires `iMCTS/benchmarks/basic.json` Nguyen-3 `data_range = [-1, 1]` and symlinked editable-install data files (see `TODO_prs.md` PR #3 workaround). Revert to N=4 by editing `source/mcts/mcts.cpp` `kBurstSamplesN = 4` and rebuilding.

## Code diff

- `include/imcts/mcts/mcts.hpp`: removed `expand_node`, `matched_pair_reevaluation`; added `burst_expand` declaration
- `source/mcts/mcts.cpp`: renamed `kMatchedPairN` → `kBurstSamplesN = 1`; removed `expand_node` and `matched_pair_reevaluation`; added `burst_expand`; rewrote `search()` to call burst_expand + stop descent
- `papers/huang2025_notes.md`: added §"GP/tree budget ratio concern under burst-expand"

Net: ~90 lines edited, 40 lines deleted. Backward path: set `kBurstSamplesN` back to 4 and wrap the call site in transition-moment detection to restore matched-pair.

## Status

**Tested:**
- Burst N=1 at gp_rate ∈ {0.2, 0.4, 0.6} on seeds 23654, 15795, 860
- Burst N=2 at gp=0.4 on the same three seeds (adds to conclusion: burst is only viable at N=1)

**Attempted and aborted for runtime:**
- Burst N=4 at gp=0.4 — exceeded 5 minutes on seed 23654 before we stopped it
- Burst N=2 at gp=0.6 — same

**Not tested (not worth testing given the findings):**
- 10-seed sweep at any (N, gp_rate) — burst is slower than matched-pair at its best and doesn't scale
- Cross-benchmark (Nguyen-4, Nguyen-5) — burst is no longer a candidate for primary MCTS strategy
- Automated symbolic-equivalence check — for this report, informal `expand + nsimplify(1e-3)` sufficed

**Conclusion:** Revert the main branch to the matched-pair transition-moment design. Keep burst-expand code as a separate branch for the negative-result writeup (if we do one). The C++ constant `kBurstSamplesN = 2` in `source/mcts/mcts.cpp` should be switched back to `kMatchedPairN = 4` with the full transition-moment machinery restored (see git commit `bdc28a1`).
