# Huang 2025 — Discussion Notes

**Paper:** Improving Monte Carlo Tree Search for Symbolic Regression (NeurIPS 2025)
**Code:** github.com/PKU-CMEGroup/MCTS-4-SR (C++, cloned to `mcts4sr/`)
**Full transcription:** `huang2025_paper.md`

## What it does

Pure classical MCTS for symbolic regression. **No neural network, no pretraining, no learned parameters.** Two innovations on top of standard MCTS:

1. **UCB-extreme** — bandit selection tracking max reward ever seen per branch (not mean). Justified because SR evaluation is deterministic: once a formula fits the data, the reward is reproducible. Proven finite-time optimality under polynomial reward-decay assumptions.

2. **State-jumping** — GP mutation/crossover operators injected during MCTS tree traversal. Operates on complete formulas stored in per-node top-N priority queues.

Per-problem, from scratch: 2M expression evaluations, ~2 min single-core. Competitive with neural-guided methods (DSR, NGGP) and GP (PySR) on standard benchmarks.

## Upstream default changes (commit 04e143a, 2026-04-15)

Laivirt123 changed three defaults in `RegressorConfig` and `basic.yaml` without explanation (commit message: "Refactor benchmark runner and tune default search config"):

| Parameter | Paper / old default | New default | Effect |
|---|---|---|---|
| `gp_rate` | 0.2 | **0.5** | GP mutation/crossover fires at 50% of non-leaf nodes instead of 20% |
| `lm_iterations` | 100 | **50** | Levenberg-Marquardt constant-fitting budget halved |
| `max_constants` | 6 | **10** | More `R` tokens allowed per formula (YAML only, C++ default unchanged) |

**Tension with our findings:** our matched-pair experiments on Nguyen-3 [-1,1] showed lower GP share helps (matched-pair's benefit was partly an effective gp_rate reduction from ~0.2 to ~0.15 equivalent). Upstream went the opposite direction. Our experiments were single-benchmark; upstream may have optimized for the broader suite. The `lm_iterations` cut is likely just compute savings — most fits converge in <50 iterations on 20-sample problems.

## Architecture: one flat tree

Unlike AlphaZero (two levels: real state + search tree per move), Huang uses **one tree for the entire problem**. Each node is a partial prefix expression; each edge appends one token. 2M iterations, each starting from root, each expanding exactly one new node.

No policy network → expansion among unexpanded children is **uniformly random**. This is why 2M iterations are needed — brute force compensates for lack of learned guidance.

## Evaluation pipeline

1. Partial expression at leaf → **random completion** (fill tokens until syntactically complete)
2. **Levenberg-Marquardt** fits numerical constants (paper says BFGS; code uses LM via Eigen)
3. NRMSE computed → reward = `1/(1+NRMSE)` ∈ (0, 1]
4. Reward + token path backpropagated to root

Constants are placeholder tokens (`constant`, arity 0). Structural placement by search/mutation; numerical values fitted fresh each evaluation, then discarded. Top-N queues store `(token_path_suffix, reward)` only — no fitted values retained.

Evaluation cache keyed by structural hash avoids redundant LM fits.

## State-jumping mechanics

Happens **during descent**, not after expansion. At each node visited, with depth-dependent probability `p_s` (decreasing with depth → fires mostly near root):

- **Mutation** (4 types, uniform random):
  - `node_replace` — swap token for same-arity token (length unchanged)
  - `shrink_mutate` — remove operator, keep one child subtree (shorter)
  - `uniform_mutate` — replace random subtree with random new subtree (length varies)
  - `insert_mutate` — wrap subtree with new operator (longer)

- **Crossover** — pick random subtree from each of two top-N formulas, swap them. Two offspring, both evaluated.

Results don't create new tree nodes. They update top-N queues via bidirectional propagation (up to root, down to existing matching descendants). Descent continues normally after state-jumping.

## Weaknesses and improvement opportunities

### Max discards basin information
UCB-extreme reports the single best reward per branch. But one branch hitting 60% once vs. many rollouts hitting 55-65% are treated identically. The second case indicates a **fertile basin** — mutations are likely to find improvements nearby. Max throws this away. A quantile-based or max+variance selection could exploit it.

**Note on noise aggregation.** Classical UCT propagates *means* because the LLN cancels rollout variance, giving concentration-bound regret proofs (Kocsis & Szepesvári 2006). UCB-extreme propagates *max*, giving up noise cancellation entirely — justified only because SR reward is deterministic given structure, so "best formula found" is ground truth, not a noisy estimate. But the noise doesn't vanish; it shifts to **proposal-order variance** (which formulas get sampled first), which is why Huang shows seed sensitivity and why variance reduction must come from CRN / matched-pair techniques rather than averaging.

### Forced full-expansion before descent
`is_leaf()` returns true if a node has ANY unexpanded children. Descent stops there and must expand before going deeper. With ~11 actions (operators + constant + variables), every node on the path needs all 11 children expanded (one random child per visit, so 11 visits) before the tree grows deeper through it. In contrast, AlphaZero/PUCT gives unvisited actions a finite prior-weighted score — low-prior actions can be skipped entirely. This wastes budget on unpromising siblings.

### Blind expansion wastes the top-N signal
Compounding the above: when expanding an unexpanded child, selection is uniformly random among remaining moves. But the top-N queue at the parent already contains complete formulas specifying exactly which token should come next. This information is available but unused. Trivial improvement: prioritize expanding the child matching the best formula in the queue. No neural network needed.

### Mutated formulas have no tree presence
A brilliant formula from mutation lives only in top-N queues — it gets no node in the tree. Forward propagation (`propagate()` in code) walks down through existing children matching the path; if no matching child exists (`if (!found) break;`), propagation stops. Future UCB-extreme selection will be drawn toward the branch by the high reward, but the actual path to recreate that formula may not exist as tree nodes.

Partially mitigated: when a node is later expanded, its parent pushes top-N entries down to the new child. So the signal eventually reaches new nodes — but with delay, and only if expansion happens to choose the right child (which is random).

**Proposed fix — canonical simplification before top-N insertion:**

When any formula enters a node's top-N queue (from rollout, GP mutation, or crossover — not just GP), additionally:

1. Inline fitted `R` constants into the expression → SymPy expr
2. `nsimplify(expr, tolerance=1e-2)` — **no `expand()`** (see below)
3. Convert back to prefix tokens: operators/variables preserved, remaining constants → `R`
4. Walk the simplified formula's prefix path through existing tree nodes
5. At each matching node, offer the simplified formula to that node's top-N (re-evaluate with fresh LM fit for honest reward)

**Why no `expand()`:** `expand` destroys good structure. The cyclotomic factorization `x·(x² + φx + 1)·(x² − x/φ + 1)` is depth ~5 in the prefix tree. `expand` turns it into `x⁵ + x⁴ + x³ + x² + x` — a nested `+` chain at depth ~11, past `max_depth=6`. The golden-ratio constants vanish, replaced by integer coefficients. The expanded form cannot be represented in the tree and loses the structural discovery matched-pair makes. `nsimplify` alone preserves the operator skeleton, only snapping constants (`0.618034 → (√5−1)/2`, `0.9997 → 1`, `6393264 → ∞` → term coefficient becomes 0 → term drops).

**What nsimplify does at each stage of search:**

- *Early/mid search* (reward < 0.99): formulas are imperfect. Extra terms have **substantial** coefficients (e.g., `0.15·sin(x)`) because they actively compensate for the main structure being wrong (`1.2·x²` instead of `x²`). No tolerance drops these — they're not close to 0. **nsimplify's main job here is constant-snapping for dedup**: `0.9997 → 1`, `1.618034 → φ`. Two formulas with identical operator trees but slightly different LM-fitted constants map to the same canonical form. This is the primary dedup source.

- *Late search* (reward ≈ 1.0): main structure is correct. Decorative terms shrink to tiny coefficients (`sin(x/6393264)`, `exp(x)·9e-12`). Now nsimplify drops them entirely — `sin(x/6393264) → sin(0) → 0`, the term vanishes. This is genuine cleanup, producing shorter formulas.

**Tolerance `1e-2`:** chosen to catch both cases. For constant-snapping, even `1e-3` works (`0.9997` is within `1e-3` of `1`). For term-dropping, `1e-2` catches coefficients like `0.003` that `1e-3` would miss. The safety net: the **original unsimplified formula is already in top-N** via normal propagation. The simplified version is an additional entry competing for a slot, not a replacement. If simplification is too aggressive (snaps `0.618` to `5/8`), the simplified version gets a worse reward on re-evaluation and doesn't make top-N. No harm done.

**Depth guarantee:** `nsimplify` without `expand` never increases depth. It can only: (a) snap constants (same token count), (b) reduce terms to 0 and drop them (fewer tokens, less depth). The max-depth overflow concern is eliminated.

**Dedup mechanism:** algebraically equivalent formulas found via different search paths — Horner form `x·(x·(x+1)·(x²+1)+1)`, direct polynomial `x⁵+x⁴+x³+x²+x`, cyclotomic `x·Φ₅(x)` with slightly different fitted constants — all produce the same canonical form after constant-snapping. They compete for the same top-N slots along the same tree path instead of fragmenting across separate branches.

**Re-evaluation:** the simplified formula must get a fresh LM fit before entering top-N, because nsimplify may have changed constant values or dropped terms. Inheriting the original's reward would let bad simplifications enter with falsely high rewards. One extra LM fit per top-N entry — still cheap (~10ms) relative to the dedup benefit.

**Cost:** one `nsimplify` call (~1ms) + one LM fit (~10ms) per top-N insertion. With K=500 queue size and ~500k iterations, total overhead is bounded by the number of actual top-N updates (queue insertions, not attempts), which is much smaller. Dwarfed by the ~2M baseline LM fits.

**Implementation:** Python ↔ C++ bridge via pybind11 (already in use). The `nsimplify` call is in Python; the MCTS loop is in C++. On each top-N insertion, call back into Python with the inlined expression string, receive simplified prefix tokens + re-fitted reward. Latency ~11ms per call — acceptable if batched or if top-N insertions are infrequent relative to iterations.

**SymPy round-trip caveat:** SymPy rewrites `x0*x0` as `Pow(x0, 2)`, which is not in the MCTS operator set `{+,-,*,/,sin,cos,exp,log,R}`. After nsimplify, run `expand_integer_powers()` to convert `Pow(x, n)` back to `Mul(x, x, ..., evaluate=False)` before converting to prefix tokens. Verified: canonical ordering survives this round-trip.

**SymPy already handles commutative canonicalization:** `Add` and `Mul` arguments are stored in a deterministic canonical order (verified: variables sorted alphabetically, consistent ordering for functions and nested expressions). So nsimplify produces commutatively canonical output — no separate commutativity pass needed. The during-expansion detection (§ below) is an optimization that prevents non-canonical nodes earlier, saving LM fits.

### TODO: Commutativity canonicalization during expansion (not yet implemented)

**Problem:** `+ a b` and `+ b a` (and `* a b` / `* b a`) occupy different tree paths but are algebraically identical. The search wastes budget exploring both.

**Mechanism — incremental comparison at expansion time:** during descent through the second argument of a commutative op, maintain a token-by-token comparison against the first argument (which is already complete). At each **expansion point** (when `expand_node` is about to create a new child):

- New token < first arg's corresponding token → **non-canonical detected**. Don't create the node. Evaluate the canonical formula (`+ a b` with args swapped), `propagate()` its reward through existing canonical nodes (option A — no forced node creation).
- New token > first arg's corresponding token → canonical confirmed, stop checking.
- Tokens equal → undetermined, create node normally, continue checking at next expansion.

**Why detection is at expansion, not during UCB descent:** descent through existing nodes is passive — those nodes already have stats and serve multiple formulas. Action (redirect or not) only matters when creating new nodes.

**Shared intermediate paths are not wasted:** the path `+ → b_tokens → a_partial_tokens` serves ALL formulas `+ b (a_prefix ...)` where the completion is `≥ b` (canonical). Only the specific non-canonical completion gets redirected. Example: `+ → sin → cos → x1 → sin → cos` serves both `+ sin(cos(x1)) sin(cos(x2))` (canonical, x2 > x1) and the redirected `+ sin(cos(x1)) sin(cos(x0))` (non-canonical, x0 < x1).

**Nested commutativity is not a special case:** in prefix notation, inner subtrees complete before outer ones. The arity-completion stack processes inside-out. By the time any outer commutative op's args are both complete, all inner ones have already been canonicalized.

**Combines with nsimplify:** nsimplify at top-N time catches everything (algebraic equivalences, constant drift, commutativity). Expansion-time detection is an early filter that prevents building non-canonical paths and saves LM fits. Both feed `propagate()` with canonical token paths — they reinforce each other.

### Structural analysis note for future runs

The benchmark CSV has three expression columns: `expression` (raw prefix tokens), `materialized_expression` (constants inlined), and `simplified_expression` (after `sp.expand(sp.simplify(...))`). The `simplified_expression` column destroys structural discoveries — `expand` flattens factored forms (cyclotomic, Horner) into flat polynomials.

**For structural analysis, apply `nsimplify(materialized_expr, tolerance=1e-2)` without `expand` or `simplify`.** This snaps constants (0.618034 → rational or surd) and drops decorative terms (sin(x/6393264) → 0) while preserving the formula's operator skeleton. The structural findings in our matched-pair/burst-expand reports (golden-ratio cyclotomic factorizations, Horner forms, trig identity rediscoveries) came from reading `materialized_expression`, not the SymPy-destroyed `simplified_expression`.

### Near-constant subtree detection

See [near_constant_subtree_detection.md](./near_constant_subtree_detection.md) — literature survey (Kinzett 2008, Johnston 2010, Rockett 2020, Javed 2022 survey), our Brush-based implementation in `simplify.hpp`, caveats (multiplicative identities removed for safety), and the open `~1×f(x)` dedup problem.

### TODO: replace backpropagate + propagate with single root propagate

`backpropagate` walks UP from a node to root, prepending moves incrementally. `propagate` walks DOWN from a node through matching children. In mutation/crossover, both are called — two walks covering the same nodes. A single `root->propagate(full_path, reward)` (plus `root->path_queue.append` for root itself) covers all existing nodes in one downward walk.

**Why this works:** every formula that reaches a child also reaches the parent (backpropagate walks up, propagate walks down from ancestors). Parent's candidate pool ⊇ any child's pool. With equal queue capacity K, parent's best ≥ child's best — a mathematical invariant of the top-K selection. So top-down propagation from root loses nothing.

**Benefits:** one walk instead of two, simpler code, and the full token sequence is naturally available for canonicalization (just `canonicalize(full_path, pset)` before propagating). The early-stop optimization in backpropagate (`if (!append) break`) is valid but saves at most ~10 failed appends per formula — negligible vs LM cost.

**Current code has the full path available at every call site:** `state.get_op_list()` (descent + expansion) + `path` (rollout suffix). No reconstruction needed.

### Missing DGSR+MCTS comparison
Kamienny's DGSR+MCTS (ref [25]) is cited in related work but **not benchmarked**. The paper compares against DSR, NGGP, GEGL, PySR — all weaker than DGSR+MCTS. Notable omission. Direct comparison would be informative: zero-training MCTS vs neural-guided MCTS on identical benchmarks.

### Unfair sibling comparison from random rollout

All three are genuinely good ideas. Let me think through each:

**1. Expand all siblings in one iteration with shared completion**
- Strong idea. It turns the sibling comparison into a **common random numbers** experiment — same context, only the sibling choice differs. Directly addresses the unfairness you spotted.
- Cost: one iteration does 11× evaluations at that depth. But the signal is much cleaner — fewer iterations may be needed overall.
- Implementation: straightforward modification of the expand step. Instead of expanding one random child and rolling out, expand all unexpanded children and roll out each with the **same** random seed for the completion.

**2. Least-harm completion instead of random**
- Brilliant, but the "right" completion depends on the operator. Your `x*1=x` and `x+0=x` examples are identities. Generalizing:
  - `a * ?` → `?=1`
  - `a + ?` → `?=0`
  - `a - ?` → `?=0`
  - `a / ?` → `?=1`
  - `log(?)` → `?=e` (gives 1) or `?=1` (gives 0)
  - `sin(?)` → `?=0`
  - `exp(?)` → `?=0` (gives 1)
- **Even better generalization:** replace every unfilled position with `R` (constant token) and let LM fit them all. The partial formula gets evaluated at its **best possible completion** under the committed structure. This turns MCTS evaluation into "what's the best this structural commitment can achieve?" instead of "what does one random completion look like?"
- This is actually a profound change. A partial formula becomes a **family of complete formulas parameterized by learnable constants**, and the reward is the best in that family.

**3. Re-evaluate sibling nodes**
- Directly fixes the unfairness in Nguyen-3 (`x0` was evaluated with an incomplete POLY, `sin` with a more complete one).
- Light version: **invalidate cached rewards** when a branch above changes, so the next UCB-extreme query triggers fresh evaluation. Lazy rather than eager.
- With suggestion 2, re-evaluation is cheap because the completion is deterministic — just redo the LM fit on the new context.

**Combining all three:** Expand all siblings together, complete with `R` tokens let LM fit, re-evaluate lazily when structure changes. This gives fair, informative, consistent comparisons. The search becomes much more like "structural commitment testing" instead of "random rollout lottery."

**Caveats:**
- Suggestion 2 (R-fill + LM fit) makes rollout expensive — each partial formula triggers an LM run instead of a cheap numerical evaluation. Huang's current setup does LM anyway on terminal formulas, so it's not worse than the final cost, but now it happens at every node.
- Suggestion 1 and 3 are additive fixes. Suggestion 2 is transformative and might need its own paper.

**My ranking for impact/effort:**
1. Suggestion 1 (same completion, all siblings) — cheapest win, fair comparisons
2. Suggestion 2 variant: "never rollout randomly, always use LM fit on R-filled" — transformative
3. Suggestion 3 — nice-to-have once 1 and 2 are in

These would be a solid paper on top of Huang. Quality-of-search improvements, not just a new benchmark number.

#### Refinement to suggestion 1: matched-pair sampling over *multiple* shared completions

Suggestion 1 (common completion across siblings) is a special case of **Common Random Numbers (CRN)** — the variance-reduction technique from simulation/statistics. Standard CRN uses a *single* shared completion. That's not fully fair in SR for a subtle reason: siblings that differ structurally also differ in what they "absorb" of the target, so they want *different* optimal contexts for the rest of the formula.

Concrete example — Nguyen-3 on [-1,1]. Say `f = + [poly] <hole>`, and we're comparing sibling `x` vs sibling `sin(x)`:

- With sibling **x**, the best `[poly]` is exactly `x² + x³ + x⁴ + x⁵`.
- With sibling **sin(x)**, Taylor gives `sin(x) ≈ x − x³/6 + x⁵/120`, so the best `[poly]` is *slightly different*: `x² + (x³ + x³/6) + x⁴ + (x⁵ − x⁵/120)` — it has to compensate for what `sin(x)` deviates from `x`.

Same target, different best decomposition. CRN with a *single* shared `[poly]` completion will favor whichever sibling that completion happens to flatter. Even with CRN, the comparison is biased.

**Fix — Version B: CRN with multiple completion sets, per-sibling max.**
1. At expansion time, draw N random completions of the "rest of f" (call them `C₁, …, C_N`).
2. For each unexpanded sibling, evaluate it under **each** of the N completions.
3. Record the *max* reward per sibling over N completions.
4. Compare siblings by their max.

Each sibling gets a fair chance to find its best supporting context, and the comparison uses correlated noise (CRN between siblings) so between-sibling variance shrinks. This is known in simulation literature as **matched-pair Monte Carlo** — the statistically optimal way to compare K alternatives under random perturbation when each might have a different optimal context.

**Cost:** `N × K` LM fits per expansion instead of 1. With N=4, K≈6, that's 24 LM fits per expansion. At ~500k expansions it comes to ~12M LM fits — ~6× Huang's current rollout budget of 2M. Feasible but not free. For easier problems it's overkill; for the fragility cases (like Nguyen-3 [-1,1]) it directly targets the failure mode, because exploit branches win by a narrow rollout lottery that matched-pair sampling eliminates.

**Further refinement — only pay matched-pair cost where it matters.** Huang's descent stops at any node with unexpanded children, so to *pass* a node all K children must be expanded, which means descent must reach it K times. Each of those K visits is UCB-directed at ancestors, so "fully expanded" is itself evidence the branch is promising — the node has survived K UCB-selections at the parent level. This means the **parent-level sibling unfairness** (which of the parent's children survives) is largely absorbed by the 10-visit survival pressure: unpromising branches never get fully expanded in the first place, and their "unfair" comparisons at the top never matter.

What's NOT absorbed is the **child-level sibling unfairness**: once a node's K children are all expanded, UCB-extreme at this node compares them by their individual max rewards, each based on 1 unmatched random-completion rollout. That's where matched-pair pays off.

**Corollary:** run matched-pair sampling only at the *transition moment* — the expansion that takes the node from "has unexpanded moves" to "all moves expanded, ready for UCB selection." At that moment, re-evaluate all K children with N shared completions and overwrite each child's initial max. This targets exactly the child-level unfairness without paying matched-pair cost elsewhere. The number of transition moments is much smaller than total expansions (roughly 1 per passed node, not 1 per new child), so the amortized cost is well below the naive ~6× estimate — probably closer to 1.5–2× current budget. Still not free, but much cheaper and surgical.

**How it plays with UCB-extreme:** UCB-extreme already tracks max over many rollouts over the full search, so the *asymptotic* distribution is the same either way. The difference is the *early* comparison — a sibling whose first rollout is unlucky gets fewer visits, compounding. Eager multi-rollout at expansion breaks this: every newly expanded branch is guaranteed N samples before UCB selection sees it, so no branch is penalized for bad early luck.

**Updated ranking:**
1. Suggestion 1 refined → **matched-pair CRN (N shared completions)**: cheapest non-trivial win that actually targets the asymmetry.
2. Suggestion 2 (R-fill + LM, transformative): still the best long-term answer because it makes rollout deterministic and evaluation = "best achievable under structural commitment".
3. Pure single-CRN (original suggestion 1) is dominated by matched-pair.
4. Suggestion 3 (lazy re-evaluation) was already subsumed by 1; under matched-pair CRN it's even more redundant.

**Caveat on "short random rollouts":** an earlier idea in this file was to cap rollout length to reduce noise. On reflection that's not a general improvement — classical MCTS is *built* on random rollouts (UCT, MoGo, Crazy Stone, etc.), and random play gives informative averages when the space is large enough to explore. For SR specifically it's a weaker version of the right fixes (matched-pair CRN or R-fill + LM), because it reduces noise at the cost of also reducing exploration. Leaves-only rollout at `sin(?)` can only produce `sin(R)` or `sin(x0)`, never `sin(x²)`, so it forbids discovering useful substructures during rollout. Not pursued.

#### GP/tree budget ratio concern under burst-expand

Huang's baseline allocates roughly **70% of the 2M eval budget to GP ops** and 30% to tree-expansion rollouts. The mechanism is probabilistic: during each descent, at every non-leaf node, a GP op fires with probability `gp_rate = 0.2`. With typical depth ~10, that's ~2 GP events × ~1.5 evals each = ~3 GP evals per iteration, against 1 tree-expansion rollout. This matches the folklore that **GAs need many evaluations to make progress** — GP relies on population-wide accumulation, not per-op wins, so Huang deliberately (or implicitly) spends most of the budget on GP churn, not on growing the tree.

Naive burst-expand (expand all K children of a leaf with N shared-seed rollouts each) inverts this: per iteration the burst costs `K × N ≈ 40` tree evals, vs unchanged ~3 GP evals from descent. **Ratio collapses from 70/30 GP-heavy to ~7/93 tree-heavy.** The GA side is starved ~10×. Under the 2M budget, ~47K iterations instead of ~500K — fewer but fatter.

**Can raising `gp_rate` compensate?** Partially. `gp_rate` is bounded at 1.0, so the ceiling is `depth × 1.0 × 1.5 ≈ 15 GP evals per descent`. At the N=4 burst cost of 40 tree evals per iteration, max `gp_rate = 1.0` gets to 73/27 tree/GP — nowhere near the 30/70 target.

**Recipe used in our implementation: `kBurstSamplesN = 1` + default `gp_rate = 0.2`.** With burst N=1 the per-iteration tree cost is `K ≈ 10`, and at `gp_rate = 0.2` the GP evals are ~3. Ratio ~77/23 tree/GP — still tree-heavier than Huang, but within factor-of-3 range. If experiments show GP starvation, bump `gp_rate` to 0.5 (→ 10 tree + ~8 GP = 55/45) or 1.0 (→ 10 + 15 = 40/60, close to Huang). Structural change and budget-balance knob are kept orthogonal: first measure the pure effect of burst-expand at default settings, then tune `gp_rate` if needed. Additional "post-burst GP ops at the parent" and "progressive burst across iterations" are deferred as future knobs, not defaults.

#### Idea: automatic hyperparameter adaptation

The `(N, gp_rate)` sweet spot is clearly problem-dependent (Nguyen-3 vs Nguyen-4 in the matched-pair notes), and probably also *phase*-dependent within a single run: early exploration wants low GP (fresh path_queues, not much to mutate from), late refinement wants high GP (queues are populated, mutation is productive). A single global setting picked by hand will always be suboptimal for some combination. A few ways to close this without a grid search:

1. **Bandit over hyperparameter values.** Treat `gp_rate ∈ {0.2, 0.3, 0.4, 0.5, 0.6}` as arms of a multi-armed bandit, allocate search budget via EXP3 or UCB based on observed "improvement rate" (Δ best_reward per 1000 evals). Per-problem autotuning with O(1) overhead.
2. **Schedule.** Start with `gp_rate` low (favoring tree growth) and ramp up as path_queues populate, similar to learning-rate schedules in DL. Natural answer to the early-vs-late phase asymmetry.
3. **Signal-based local adaptation.** Monitor path_queue saturation per node — if entries are close to K and churning fast, lower gp_rate at that node; if the queue is stable with stale entries, raise it. Local per-node gp_rate rather than global.
4. **Meta-search.** Run a pilot search at low budget across 3–5 hyperparameter points, pick the best setting, run the full search with it. Simple and robust but pays a constant fraction of budget as overhead.

Option 1 is the principled version; option 2 is cheapest to implement. The same ideas apply to `kBurstSamplesN`, `exploration_rate`, and `c` — any knob where we currently pick a number and hope for the best.

### RNG choice is not the cause of seed fragility

Huang's C++ core uses `std::mt19937_64`. MT19937 has known problems — poor small-seed diffusion (nearby seeds produce correlated state), TestU01 BigCrush failures, 2.5 KB state — and was replaced as NumPy's default in 1.17 (2019) by PCG64. Swapping to PCG64DXSM (the self-correlation-fixed variant from `imneme/pcg-cpp`) is a trivial one-line change in `types.hpp`.

We tested it. On Nguyen-3 [-1,1] with R, success rate went from 4/8 to 6/10 — within noise. More telling: **the "wrong" seeds under PCG64DXSM use DIFFERENT exploits** than under MT19937 (exp·x, cos·x, rational 1/(1+x), sin-linear) rather than converging to the truth. So swapping the RNG just shuffles which seeds land in exploit basins; it doesn't reduce the total size of those basins.

The seed fragility on Nguyen-3 [-1,1] is **not** an RNG quality problem. It's benchmark under-determination: the narrow range admits many transcendental Taylor-approximations of `x⁵ + x⁴ + x³ + x² + x` that all hit reward 1.0. Any MCTS with random rollouts will find one of them some fraction of the time, regardless of PRNG. The real fixes are (a) widen the range to [-10, 10] (which kills the exploits because they extrapolate badly), or (b) the sibling-fairness proposals above.

Worth keeping PCG64DXSM anyway for code-quality reasons — see `mcts4sr/TODO_prs.md` PR #4.

### No generalization across problems
Each problem solved independently from scratch. No knowledge transfer between problems. A learned prior (policy network) would amortize experience — but that's exactly what this paper avoids.

## Comparison with Gumbel MuZero (mctx)

| | Huang UCB-extreme | Gumbel MuZero |
|---|---|---|
| Simulations needed | ~2M | 8-16 |
| Prior | None (random) | Learned policy network |
| Selection among unexpanded | Random | Prior-weighted + Gumbel noise |
| Budget allocation | UCB exploration term | Sequential Halving |
| Evaluation cost per sim | Cheap (string ops + LM) | Expensive (neural net forward pass) |
| Improves over time | No (no learning) | Yes (policy improves → Gumbel gets better) |

Gumbel is the right choice when simulations are expensive (neural network in loop). UCB-extreme is the right choice when simulations are near-free and there's no learned prior.

## Relevance to our work

**As a zero-training baseline:** If classical MCTS with no learning matches neural methods, it sets the bar the Assessor-trained model must clear. Running Huang's code on our benchmarks would establish this baseline cheaply.

**The revolutionary observation:** Competitive results with zero pretraining, zero learned parameters, against methods that train transformers for hours/days on GPUs. This validates MCTS as a strong amplification operator — and suggests that the bottleneck in neural-guided SR may be the quality of the prior, not the search algorithm.

**Practical for our Boolformer MCTS:** The "blind expansion" weakness is directly relevant — our policy head serves exactly the role that Huang's method lacks. If our policy head is good, we should dominate; if it's bad, we're paying neural-net cost for Huang-level performance. This gives a concrete diagnostic: compare policy-guided vs uniform-random expansion with same simulation budget.
