# Kamienny — DGSR+MCTS notes

Source: `kamienny2023_phd_thesis.pdf` (PhD, Sorbonne, 2023) Chapter 7 + Kamienny et al.
ICML 2023 `kamienny2023_dgsr_mcts.pdf` (arXiv 2302.11223).

## What the thesis is (and isn't)

The thesis is about **efficient adaptation in RL via meta-learning**, not symbolic
regression per se. Two parts:

- Part I (Ch 3–4): model-free meta-RL exploration.
- Part II (Ch 5–8): symbolic regression as a tool to build **symbolic world models**
  for model-based RL. Ch 8 applies SR to fit interpretable dynamics instead of NN
  world models.

Recurrent-formula work (arXiv 2201.04600) is in **Appendix F** — older, sidelined.
The main SR contribution is **Chapter 7: DGSR+MCTS** (published as Kamienny ICML 2023).
That is what we discussed.

## The framing argument Kamienny makes

His stated case against token-by-token seq2seq SR + MCTS (Boolformer-style):

> "such left-to-right blind way of decoding does not allow for accurate planning;
> using accuracy objectives to guide the decoding would be difficult, since
> intermediate sequences of tokens are not valid expressions."

The claim is that intermediate token prefixes can't be scored, so the search has no
signal except at terminal leaves. His fix: make every MCTS node a *syntactically
complete expression* so you can evaluate it at any depth.

**This framing is weaker than he sells it.** A learned critic (which he uses) is
perfectly happy to score partial sequences — AlphaZero values mid-game positions
the same way. The real lesson of his paper is in the ablations, not the framing
(see below).

## How it actually works

### State / action definition

- **MCTS node** = a full, valid, evaluable expression tree `f`. Initial node is a
  trivial expression (`∅` / single constant). Never a half-parsed prefix.
- **MCTS edge** = a mutation triple `⟨A, op, B⟩`:
  - `A` = index of an existing node in the current tree `f`
  - `op` = one of Table 7.1 operators: unary wraps (`sin`, `cos`, `log`, `exp`,
    `A^2`, `A^0.5`, `A^-1`, etc.) or binary combiners (`A+B`, `A-B`, `A*B`, `A/B`,
    `B+A`, ...)
  - `B` = a new sub-expression (only for binary ops)
- Mutations **strictly grow** the tree — they never shrink or replace. No crossover.

Example of one mutation with a 10-token `B`:

```
current:  f = y
mutation: ⟨root, A → A + B, B = exp(x₁) · (sin(x₂) + cos(x₃ − x₁))⟩
          B in prefix: [*, exp, x₁, +, sin, x₂, cos, -, x₃, x₁]   (10 tokens)
result:   f' = y + exp(x₁) · (sin(x₂) + cos(x₃ − x₁))
```

One MCTS edge commits 11 tokens atomically.

### The transformer Mθ (policy)

Same encoder-decoder architecture as Kamienny's 2022 E2E SR model (close cousin of
Boolformer).

- **Encoder input**: dataset `D` (tokenized (x,y) pairs) **and** current expression
  `f` in prefix notation.
- **Decoder output**: a flat sequence
  `ω = [A_idx, op_id, B_token_1, B_token_2, ..., B_token_k, EOS]`, decoded
  autoregressively until EOS. So the "atomic" mutation is atomic only at the MCTS
  level — internally it's still token-by-token generation.
- **Critic Cψ** shares the trunk with Mθ and adds one extra value head.

**No precomputed table, no lookup.** Just a decoder that generates `B` one token at
a time given `(D, f)`.

### The three roles Mθ plays in MCTS

Unlike AlphaZero where the policy is only a prior over a fixed action set, here Mθ
does three jobs because the action space `⟨A, op, B⟩` is effectively infinite:

1. **Action enumerator**: at expansion, sample `K ∈ [8,16]` mutations from Mθ.
   Those K samples *are* the children of the expanding node. Any mutation Mθ
   didn't sample simply doesn't exist in the tree.
2. **PUCT prior**: selection uses
   `V(f') + p_uct · E(f') · Mθ(f' | f, D)`
   where `Mθ(f' | f, D)` is the decoder's log-likelihood of the ω sequence that
   produced `f'`.
3. **Expert-iteration target**: successful mutation sequences are stored and used
   to fine-tune Mθ via imitation learning between search trials.

### Leaf evaluation

- If `R²(f', D) ≥ 0.99` (optionally after BFGS constant fit) → `v(f') = 1.0`
  (ground-truth override).
- Otherwise → `v(f') = Cψ(D, f')` (neural critic).

Crucially, **R² is not the MCTS value**. He tried using R² directly as v and
explicitly rejected it:

> "a few mutations that lead to less accurate expressions (akin to sacrificing
> pieces in chess) may be needed before a very accurate expression is found."

Raw accuracy as a value function is deceptive — it punishes necessary setup moves.
The critic is trained to predict *will this lead to a solution*, not *how accurate
is it right now*.

### Training

- **Pretraining** of Mθ: generate ground-truth `f*`, procedurally dismantle each
  one (remove a node + one subtree, reconnect), record the reverse sequence of
  mutations `m₁, ..., m_L` that would build `f*` from `∅`. Train Mθ supervised on
  `(D, f⁽ˡ⁻¹⁾) → mₗ` with cross-entropy on the flat ω.
- **Mutation size** is controlled by *where* the dismantling picks nodes — higher
  in the tree means bigger removed chunks, so bigger `B` in the reversed mutation.
- **Expert iteration**: repeat {1000 MCTS sims on current datasets, collect
  successful `(m₁...m_L, f*)`, update Mθ by imitation and Cψ by visit-weighted
  value}. Continue sampling from the synthetic pretraining distribution during
  RL to prevent mode collapse.
- **Multi-dataset simultaneous training**: search runs on many SR instances at
  once so gradients are shared across tasks.

## Results

### The mutation-size ablation (Table 7.3) — the key finding

Test set: 1000 synthetic expressions the model has never seen, split
500 **in-domain** (same generator as pretraining, ≤25 operators) /
500 **OOD** (same operators but bigger expressions, up to 40 operators).
A dataset counts as solved if the best returned formula hits `R² ≥ 0.99` on the
training split. Numbers are % of datasets solved.

| Mutation size | In-domain | OOD | Interpretation |
|---|---|---|---|
| @1 (≈1 token per edge) | 52.2 | 26.8 | Token-level — too fine-grained, search can't cover the combinatorial space in budget |
| @10 (≈10 tokens per edge) | **74.8** | **44.0** | Sweet spot |
| @∞ (whole formula in one shot) | 72.4 | 16.8 | ≡ pure E2E Boolformer. In-domain fine, **OOD collapses** |

Two things to take away:

- **@10 beats @1 by ~20 points on both splits.** Bigger atomic action steps win.
- **@∞ loses catastrophically OOD.** Pure seq2seq is only competitive when test
  data looks like training data. Search buys OOD robustness, but only at a useful
  granularity.

**Why @10 beats @1 — same machinery, different locus of planning.** All three
rows use the *same* transformer architecture, same critic, same PUCT selection,
same K ∈ [8,16], same 1000 sims. The only difference is the pretraining
curriculum. The reason the gap is ~20 points is not "more tokens per edge", it's
**where the multi-step compositional reasoning lives**:

- In **@10**, the autoregressive decoder is trained to emit coherent ~10-token
  sub-expressions. The transformer's own attention does the multi-token planning:
  when it emits `sin`, it already has distributional beliefs about the argument.
  Each decoded B hangs together as a meaningful unit because that was the
  supervised target. MCTS then picks between K coherent candidate chunks.
- In **@1**, the decoder is only trained to predict 1-token futures. It has no
  signal that "these 10 tokens belong together." All multi-step composition
  burden shifts onto MCTS, which has to chain many locally-ranked single-token
  decisions into globally coherent sub-expressions. Same transformer, weaker
  prior over long-horizon moves.

@10 doesn't have *more* search power — it has a **stronger differentiable prior
that pre-compresses the search problem**. Secondary amplifiers: (i) effective
tree depth — 1000 sims can only meaningfully explore a few atomic decisions deep
with K ≈ 12, so @10 covers ~10× more token-structure depth at the same budget;
(ii) credit assignment — a node with `sin(x₁)+x₂²` gives the critic a far more
informative signal than a node with `sin(0)`.

**Could @1 catch up with more compute?** In principle yes (same reachable space —
size-1 mutations are complete for any expression). In practice no with any
realistic budget. If @10 finds solutions in ~5 MCTS steps, @1 needs ~50. With
branching K ≈ 12, a full tree of depth 5 has ≈ 2.5×10⁵ nodes; depth 50 has ≈
10⁵⁴. Even assuming a strong PUCT prior prunes the effective branching to ~3,
the gap is still 3⁵ ≈ 240 vs 3⁵⁰ ≈ 7×10²³. The real obstacle isn't just tree
size — it's that @1's policy was never trained to rank long-horizon composability,
so its PUCT prior is weakest precisely where @1 needs it most. Adding sims lets
@1 flail longer with a weak prior; it doesn't make the prior stronger.

The only way @1 meaningfully closes the gap is with a dramatically stronger
**critic** that can rescue bad local token choices via backtracking — but
training such a critic is itself the hard problem AlphaZero needed millions of
self-play games to solve, and that's a parallel pretraining cost, not a
"just throw more sims at it" fix.

### Constant optimization ablation (Table 7.4)

Separate lever, doesn't apply to Boolean formulas but worth noting as the
magnitude of a "small" design choice:

| Constant opt | In-domain | OOD |
|---|---|---|
| Never | 74.8 | 44.0 |
| Best expr only | 77.2 | 59.4 |
| All expressions | **79.6** | **66.2** |

He stores the *non-optimized* expression in the MCTS tree (optimized constants are
OOD w.r.t. Mθ) but uses the BFGS-fitted version for scoring. Two-track design.

### SRBench headline numbers (Table 7.2 / Fig 7.2)

| | Black-box R² | BB size | Feynman solve (R² ≥ 0.99) | Feynman size |
|---|---|---|---|---|
| E2E Kamienny 2022 (seq2seq) | 0.797 | 61 | **87%** | 121 |
| DGSR+MCTS 2023 | **0.846** | 41 | 80% | **33** |

Two-sided result: **MCTS wins on black-box (harder, OOD-ish) but loses on
Feynman solve rate to pure seq2seq.** On Feynman the MCTS payoff shows up as
much simpler formulas (33 vs 121 tokens), not as higher solve rate. So "MCTS
beats seq2seq" is only unambiguously true on OOD/noisy data; in-distribution,
pure seq2seq is stronger in solve rate and MCTS trades solve rate for
simplicity.

**Not the single best SR model, even in 2023.** DGSR+MCTS is on the rank-0
Pareto front of SRBench, but so is **GP-GOMEA** — by Kamienny's own words *"GP-GOMEA
and DGSR+MCTS seem to be the best approaches for achieving simple-yet-accurate
models, and interestingly switch place in their trade-off between the two
metrics on the black-box and Feynman datasets"*. AI-Feynman dominates Feynman
outright. As of Huang 2025 (NeurIPS, our most recent paper), **Operon (GP) leads
test accuracy** and Huang 2025 claims #2, with DGSR+MCTS folded into "prior
MCTS-based approaches". GP has never been displaced at the top of SRBench —
transformers + search joined the front, they didn't sweep it.

**Relevance caveat for Boolformer:** SRBench is real-valued SR. None of these
benchmarks include Boolean function inference, so "who is #1 on SRBench"
doesn't directly tell us what to do for our plateau. The transferable lesson
is mechanistic: size-10 mutations >> size-1 mutations, not "copy Operon".

## Critical read

### What Kamienny *claims* is the advantage

"Intermediate states are always valid expressions, so we can evaluate them with
ground-truth accuracy at every node."

### What actually drives the gains

**Multi-token atomic action steps, period.** His own ablation is unambiguous: a
model with size-1 mutations (effectively token-level) gets 52/27; a model with
size-10 mutations gets 75/44. Same architecture, same training loop, same critic,
same number of sims — the only change is how many tokens each MCTS edge commits.

The "always-valid" framing is post-hoc justification. You could take a token-level
MCTS in Polish prefix notation, extend the action space to include "emit next k
tokens", and recover most of Kamienny's benefit without changing the state
representation.

Constant optimization is a second real advantage (Table 7.4) but it's orthogonal
to the search design and doesn't apply to Boolean formulas.

## vs Huang 2025 (`huang2025_improving_mcts_sr.pdf`, NeurIPS 2025)

Both are MCTS for SR but architecturally different:

| | Kamienny 2023 | Huang 2025 |
|---|---|---|
| MCTS state | Complete expression tree | Partial pre-order token sequence |
| MCTS edge | Semantic mutation `⟨A, op, B⟩`, grows tree | Append one symbol to sequence |
| Atomic action size | ~10 tokens (policy-proposed chunks) | 1 token (but with state-jumping escape hatch) |
| Policy | Pretrained transformer Mθ, fine-tuned by expert iteration | None — classical MCTS, no learning |
| Value | Neural critic Cψ | Bandit statistics |
| Bandit rule | PUCT (AlphaZero) | **Extreme bandit** with finite-time regret bounds under polynomial reward decay |
| Action types | Monotonic growth mutations only | **Mutation + crossover** on top-N stored trajectories per node |
| Escape-local-optima | Larger action size | **State-jumping**: pick top-N trajectories at a node, GP-mutate/crossover, bidirectionally propagate |

They're complementary. Kamienny = better policy/value guidance; Huang = better
bandit + richer action set + global jump mechanism. Huang's base MCTS is actually
closer to our Boolformer setup than to Kamienny's.

## vs our Boolformer-MCTS setup

Our current setup (from `mctx/examples/boolformer/train.py`, CLAUDE.md):

- Token-level MCTS in prefix notation
- Depth 4, **8 sims per step**, `selfplay_batch_size = 256`
- Action = append one token from the Boolean vocabulary
- Reward = raw IoU on truth table at terminal leaves
- Policy head + value head share the transformer encoder
- Value head trained to predict IoU directly (MSE loss)
- Plateau at ~13% success for 50+ iters; random init baseline is ~10%

Cross-referencing with Kamienny's findings:

1. **We are in the @1 row of his ablation** (token-level, single-token actions),
   with an even tighter sim budget than his experiments (8 vs 1000). @1 on his
   benchmark was 52/27 — on ours it's ~13. The ablation predicted the plateau.
2. **Our value head is doing exactly what Kamienny warned against**: predict
   accuracy (IoU is the Boolean analog of R²). He showed this gives deceptive
   rewards — the value head mirrors the accuracy landscape instead of the
   solution-path landscape. Matches our CLAUDE.md diagnosis ("model predicts,
   doesn't solve; no mechanism to push for better solving").
3. **No mode-collapse anchor**: our training has no equivalent of "keep sampling
   from the pretraining distribution". Kamienny explicitly does this to prevent
   the policy from collapsing onto what the critic currently likes.

## Practical actions we could take

Listed cheapest → most invasive:

1. **Decouple value from IoU.** Train the value head to predict "will this
   trajectory lead to a solved formula" (binary or exponentially-decayed
   solved-future), not the current-step IoU. Closest analog to Cψ. Fixes the
   deceptive-reward problem without touching MCTS structure.
2. **Multi-token atomic actions** (the cheap version of Kamienny's @10).
   Train / fine-tune the policy to emit `k`-token chunks (k ≈ 3–5 for short
   Boolean formulas). Each MCTS edge commits the whole chunk. Tree gets
   effectively deeper for the same sim count. No state-representation change.
3. **Pool anchoring**: during selfplay iterations, keep a fraction of the
   training batch drawn from the supervised pretraining distribution (fresh
   synthetic formulas + their solutions, policy trained by cross-entropy).
   Prevents value-head collapse, matches Kamienny's expert-iteration recipe.
4. **Huang-style state-jumping**: keep top-N trajectories per MCTS node,
   periodically mutate/crossover them, bidirectionally propagate. Drop-in on
   top of token-level search, no retraining.
5. **Post-hoc canonicalization** (Boolean analog of Kamienny's constant
   optimization): simplify the emitted formula with a Boolean simplifier before
   scoring, but keep the raw token sequence in the policy/value inputs.
   Two-track design like Kamienny's constants.
6. **Full rewrite to expression-level mutations** (Kamienny @10 proper).
   Highest gain in theory but biggest engineering cost: new action
   representation, new policy decoder output (A_idx, op_id, B), new pretraining
   data pipeline (dismantle ground-truth trees).

(1) and (3) are both one-file changes to `train.py`. (2) requires touching the
policy head and how MCTS expands children, but the state is unchanged. (4) is
additive. (5) needs a Boolean simplifier. (6) is a major refactor.

## Compute cost sanity check

Kamienny's @10 runs ~K×10 ≈ 100 forward passes per MCTS expansion.
Ours (@1, depth 4, 8 sims) runs ~32 per episode. So @10 is ~3× the decoder
compute per episode for the 20-point accuracy jump. Our RTX 5090 currently shows
0% util (known nvidia-smi artifact with short burst workloads, but also some
genuine idle) — compute headroom is not the bottleneck.
