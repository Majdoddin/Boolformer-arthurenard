# Huang 2025 — Discussion Notes

**Paper:** Improving Monte Carlo Tree Search for Symbolic Regression (NeurIPS 2025)
**Code:** github.com/PKU-CMEGroup/MCTS-4-SR (C++, cloned to `mcts4sr/`)
**Full transcription:** `huang2025_paper.md`

## What it does

Pure classical MCTS for symbolic regression. **No neural network, no pretraining, no learned parameters.** Two innovations on top of standard MCTS:

1. **UCB-extreme** — bandit selection tracking max reward ever seen per branch (not mean). Justified because SR evaluation is deterministic: once a formula fits the data, the reward is reproducible. Proven finite-time optimality under polynomial reward-decay assumptions.

2. **State-jumping** — GP mutation/crossover operators injected during MCTS tree traversal. Operates on complete formulas stored in per-node top-N priority queues.

Per-problem, from scratch: 2M expression evaluations, ~2 min single-core. Competitive with neural-guided methods (DSR, NGGP) and GP (PySR) on standard benchmarks.

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

### Forced full-expansion before descent
`is_leaf()` returns true if a node has ANY unexpanded children. Descent stops there and must expand before going deeper. With ~11 actions (operators + constant + variables), every node on the path needs all 11 children expanded (one random child per visit, so 11 visits) before the tree grows deeper through it. In contrast, AlphaZero/PUCT gives unvisited actions a finite prior-weighted score — low-prior actions can be skipped entirely. This wastes budget on unpromising siblings.

### Blind expansion wastes the top-N signal
Compounding the above: when expanding an unexpanded child, selection is uniformly random among remaining moves. But the top-N queue at the parent already contains complete formulas specifying exactly which token should come next. This information is available but unused. Trivial improvement: prioritize expanding the child matching the best formula in the queue. No neural network needed.

### Mutated formulas have no tree presence
A brilliant formula from mutation lives only in top-N queues — it gets no node in the tree. Forward propagation (`propagate()` in code) walks down through existing children matching the path; if no matching child exists (`if (!found) break;`), propagation stops. Future UCB-extreme selection will be drawn toward the branch by the high reward, but the actual path to recreate that formula may not exist as tree nodes.

Partially mitigated: when a node is later expanded, its parent pushes top-N entries down to the new child. So the signal eventually reaches new nodes — but with delay, and only if expansion happens to choose the right child (which is random).

**Proposed fix:** materialize tree nodes for mutated/crossed formulas so they're directly reachable by descent. But first canonicalize (`expand()` + `nsimplify(..., tolerance=1e-3)` for constants — tolerance is required, default `nsimplify` only finds exact rationals and won't snap `0.500012` to `1/2` or drop tiny spurious coefficients like `0.000342`) so algebraically equivalent variants — `2x + x` and `3x`, or the same expression with `0.500012` vs `1/2` constants — map to the same canonical path rather than creating duplicate branches. This also doubles as a global dedup mechanism for the wider search: multiple branches generating equivalent formulas get unified at a single node. Cost: one simplify call per mutation (~ms), dwarfed by LM cost per evaluation.

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
