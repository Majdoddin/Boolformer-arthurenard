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
`is_leaf()` returns true if a node has ANY unexpanded children. Descent stops there and must expand before going deeper. With ~11 actions (operators + constant + variables), every node on the path needs all 11 children expanded before the tree grows deeper through it. In contrast, AlphaZero/PUCT gives unvisited actions a finite prior-weighted score — low-prior actions can be skipped entirely. This wastes budget on unpromising siblings.

### Blind expansion wastes the top-N signal
Compounding the above: when expanding an unexpanded child, selection is uniformly random among remaining moves. But the top-N queue at the parent already contains complete formulas specifying exactly which token should come next. This information is available but unused. Trivial improvement: prioritize expanding the child matching the best formula in the queue. No neural network needed.

### Mutated formulas have no tree presence
A brilliant formula from mutation lives only in top-N queues — it gets no node in the tree. Forward propagation (`propagate()` in code) walks down through existing children matching the path; if no matching child exists (`if (!found) break;`), propagation stops. Future UCB-extreme selection will be drawn toward the branch by the high reward, but the actual path to recreate that formula may not exist as tree nodes.

Partially mitigated: when a node is later expanded, its parent pushes top-N entries down to the new child. So the signal eventually reaches new nodes — but with delay, and only if expansion happens to choose the right child (which is random).

### Missing DGSR+MCTS comparison
Kamienny's DGSR+MCTS (ref [25]) is cited in related work but **not benchmarked**. The paper compares against DSR, NGGP, GEGL, PySR — all weaker than DGSR+MCTS. Notable omission. Direct comparison would be informative: zero-training MCTS vs neural-guided MCTS on identical benchmarks.

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
