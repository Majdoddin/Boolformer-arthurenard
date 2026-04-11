# AI for Math Proposal — Revision Notes

Discussion notes from 2026-04-10. The original proposal
(`AI_for_Math_Proposal.md`, March 2025) is technically correct in its core
claims; the issue is that the **load-bearing points are buried under framing
that invites attack**. A reviewer skimming the abstract will go after the
weakest framing (AlphaZero analogy, GAN comparison) and miss the strongest
substance (Fagin diminishing-returns diagnosis, shared-weight multitask design).
This document captures what's actually strong, what was initially misread, and
the open design questions.

## What's actually strong (and should be foregrounded)

### S1. The diagnosis — random synthetic data hits a Fagin-style wall

The non-obvious empirical claim: **state-of-the-art synthetic-data ATP systems
are bleeding compute on samples their model already knows.** Receipts:

- AlphaGeometry2: 20% of training data → 21/30 IMO; remaining 80% → only +4
  problems (25/30). That's not a scaling curve, it's a brick wall.
- Theoretical hook: Fagin's Zero-One Law for finite model theory — random
  structures concentrate around trivial outcome distributions. Translated:
  as the prover grows stronger, the *fraction* of random samples sitting at
  its frontier shrinks toward zero.

This is the wedge. The proposal currently mentions it mid-paragraph in
section 2; it should be the **opening** of the executive summary. The pitch
is "we have receipts that the field is hitting a data-distribution wall, and
we have a frugal answer," not "we like AlphaZero."

**Empirical strengthening (PI B, direct experiment).** First-hand
reproduction of random formula samplers from prior work shows that **most
sampled formulas are duplicates.** This is stronger than diminishing-
difficulty — it's *diminishing-existence*. The sampler runs out of distinct
outputs well before it runs out of compute. For Boolean functions the
collapse is especially brutal: the target space is finite (`2^(2^n)`
distinct functions for `n` inputs), and at modest depths syntactic formulas
collapse onto a tiny subset (constants, single-variable identities, common
patterns like majority/parity). For real-valued SR the same effect is
softer but present (symbolic equivalences `x+x = 2x`, constant collisions,
common subtree patterns).

**This is better ammunition than the AG2 numbers** because it is first-hand
and unambiguous. Action: produce a "samples generated vs distinct samples"
curve — semantic dedup (truth-table identity for Boolean, R²-identity for
real-valued), not just syntactic — and put it in the proposal as Figure 1.
The flattening *is* the Fagin wall made empirical, and makes the diagnosis
impossible to wave away.

This also feeds the plateau diagnosis directly: if the Boolformer training
pool is mostly duplicate truth tables, the model is being asked to fit the
same handful of functions over and over — exactly the "predicts accurately
but doesn't solve better" equilibrium currently observed. Pre-action: count
distinct truth tables in the current training pool.

### S2. The intervention — single multitask model, not GAN

Critical clarification missed on first read of the proposal: Prover and Assessor
**share weights**. They are one transformer instructed to play either role.
This matters because:

- A GAN has two networks with opposed objectives → notorious instability,
  especially for discrete sequence outputs (SeqGAN etc. all lost to LM
  training). The proposal currently says "parallels GANs," which invites this
  exact objection.
- A shared-weight multitask model has **one** parameter set updated jointly.
  When the Prover's representation of theorem difficulty improves, the
  Assessor *automatically* inherits that representation — same encoder, same
  weights. There is no adversarial gradient flow between two networks because
  there is only one network.

This is genuinely closer to AlphaZero (one network, two roles) than to GANs.
The proposal makes this point but doesn't make it load-bearing. It should.

### S3. Compute-frugal positioning

The intervention is about **sample distribution**, not scale. A small team
beats a big lab not by buying more GPUs but by spending each sample at the
prover's frontier. For a fund explicitly looking for breakthrough ideas over
scale-ups, this is the right pitch and should be explicit.

## What was initially misread (and why)

These are mistakes a reviewer will repeat. Listed for actionability.

| Misread | Why it happened | Fix |
|---|---|---|
| "Asymmetry between proving and proposing is unaddressed." | Asymmetry is diagnosed in §2 but the Assessor-as-answer logic is one sentence. | Make the asymmetry → Assessor link the structural backbone of §2, not a transition. |
| "Two-model GAN — discrete sequence GANs are unstable." | "Parallels GANs" appears before "single multitask model." Reader anchors on GAN. | Lead with the multitask architecture; mention GAN only as a *contrast* ("unlike GANs, we share weights"). |
| "Assessor doesn't know what's hard for the Prover." | Shared-weight property is implicit in "multitask," never spelled out. | Explicitly: "Because Prover and Assessor share parameters, the Assessor's representation of difficulty updates automatically when the Prover updates. There is no separate generator learning to track a moving target." |
| "AlphaZero comparison is rhetorical." | The headline says AlphaZero but the proposal never names AlphaZero's actual amplification mechanism (MCTS as policy-improvement operator). | See open question Q1. Either add the amplification operator and earn the analogy, or qualify the headline. |

## Critiques that still hold

- **AlphaZero analogy in the abstract.** Even with the shared-weight clarification,
  AlphaZero's defining feature is **MCTS as policy-improvement operator**: search
  produces a policy strictly better than the raw network prior, and you distill
  the gap. Without naming an amplification operator, the analogy is partial. See Q1.
- **Three-domain scope.** SR → Lean tactic for Lie algebra → compiler optimization
  in 24 months is wide. Each is its own field. Recommend deprioritizing the
  compiler arm to a "future work" mention; depth on Lean wins reviewer points.
- **Differentiation from expert iteration is implicit.** DeepSeek-Prover, AlphaProof
  etc. already use expert iteration on success/failure — that's an implicit
  curriculum (model only learns from problems it can *almost* solve). Need an
  explicit argument that adaptive **generation** beats adaptive **filtering**.
  See Q4.
- **Boolformer plateau is current evidence.** Branch `mctx` sits at ~13% success
  after 50+ iters. The 4-month SR pilot is supposed to de-risk the ATP work, but
  the foundation isn't yet learning *without* Assessor. Adding Assessor co-training
  on top of a non-learning loop multiplies risk.

## Open design questions

These are the things to decide before the next proposal revision.

### Q1. What is the amplification operator?

AlphaZero's loop only works because **MCTS produces a policy better than the
network prior**, and the network is trained to match the improved policy.
Without an analog, the self-play loop has nothing to climb on — it just
distills its own outputs.

Options for ATP:
- MCTS over proof tactics (Kamienny-style, with Mθ as prior)
- Best-of-N proof attempts + verifier as ground truth
- Symbolic engine (lean-egg / nlinarith) as the amplification — model proposes,
  symbolic engine verifies and possibly extends, distill back

**Decision needed.** Whichever is chosen, name it explicitly in the proposal
and connect it to AlphaZero's MCTS so the analogy is mechanically defended,
not just thematic.

**How the training signal flows out of the amplification operator** (settled
by Q2 above): whatever the operator is, the training signal for the network
is **cross-entropy on the filtered successful trajectories**, not a
reward-weighted gradient. AlphaZero does this (visit counts + winner as CE
targets). Kamienny does this (Table 5.1 "Selection & Imitation"). Huang 2025
does the opposite extreme — no learned model at all, pure search + classical
bandit — and still gets SRBench-competitive results, which sets a useful
"zero-pretraining" lower bound on what the amplification operator alone can
achieve. Both endpoints are useful anchors.

User position: open to MCTS, undecided.

### Q2. Assessor training objective — LOCKED

Resolved 2026-04-11. Confirmed against Kamienny 2023 thesis
(pp. 61, 79, 82, 84–85).

**Recipe**: plain seq2seq per-token cross-entropy, with an expert-iteration
loop that populates a replay buffer from a generate-then-filter step. Same
pattern as Kamienny's Mθ training and AlphaZero's policy head.
**No REINFORCE. No policy gradient. No batch loss. No scalar-reward gradient
weighting.**

**Input structure**: the Assessor takes a scalar difficulty target
`d_i ∈ [0, 1]` as a per-sample input (extra token or continuous embedding).
Each sample in a batch gets its own `d_i`. **Per-sample loss, not batch loss.**
Batch loss was an anti-cheat hack from before we had `d_i` conditioning; with
conditioning it's redundant and has worse credit assignment.

**Phase 1 — Pretraining.** Synthesize `(d, formula)` pairs where difficulty
is known by construction (complexity proxies: depth, operator count, variable
count, etc.). Train with plain per-token CE. Gives a warm-started `d → formula`
map before the Prover-in-the-loop phase. Analog of Kamienny's procedural
dismantling.

**Phase 2 — Expert iteration.** Loop:
1. For each `d_i` in the batch, sample `k` candidate formulas from the current
   Assessor (temperature > 0).
2. Measure each candidate's actual Prover-difficulty. **Must be continuous**
   (truth-table distance for Boolean, R² for real-valued). Binary
   success/failure only supports a two-bucket curriculum (`d ∈ {easy, hard}`),
   not a graded one.
3. **Filter**: keep candidates where `|measured − d_i| < ε`. Discard the rest.
4. Add kept `(d_i, formula)` pairs to a replay buffer.
5. Train Assessor with plain per-token CE on the buffer.
6. Repeat.

The scalar "measured difficulty" enters training **only as a binary
include/exclude filter on the buffer**, never as a gradient weight. Failed
candidates contribute zero gradient. Same discipline as Kamienny's
failed-trajectory discard.

**Primary-source citation** (Table 5.1, Kamienny thesis p. 61), worth quoting
verbatim in the proposal:
- **DGSR+MCTS [Kam+23]**: Update θ, ψ via "Selection & Imitation"
- **DSR [Pet+19]**: Update θ with "policy gradients"

Kamienny explicitly contrasts imitation learning from filtered rollouts
against DSR's policy gradients in his own summary table. This is the cleanest
single citation for "no REINFORCE needed."

**Why filter-then-CE beats REINFORCE despite the discrete sampling step.**
The seq2seq CE analogy holds at the level of `∇ log p(sequence | input)` —
initially I suggested it didn't and that was wrong. The real advantages of
filter-then-CE over REINFORCE on a scalar sequence reward are:
- **Per-token credit assignment.** CE supervises each token individually;
  REINFORCE assigns one scalar reward to every token in the sequence.
- **Stationary targets.** Filtered demonstrations are fixed once in the
  buffer; REINFORCE's "targets" are whatever the model sampled last, which
  drifts as the model updates.
- **Lower variance.** Bounded per-token prediction error vs. unbounded
  Monte-Carlo policy-gradient estimator.
- **Standard practice.** Filter-then-CE is how AlphaZero trains its policy
  head (MCTS visit counts + game winner as CE targets), and how Kamienny
  trains Mθ (successful trajectories as CE targets).

**`P(d)` is the curriculum control, not the loss.** Start `P(d)` uniform on
`[0, 1]`; make adaptive later (concentrate on `d` where the Assessor shows
learning progress — POET / prioritized-replay trick). Q3's "difficulty band
calibration" lives in `P(d)`, not in the loss function.

**Degeneracy analysis under this locked design** (conditional scalar `d_i`
+ per-sample CE loss + continuous signal + filtered replay buffer):

| Degeneracy | Status | Fix |
|---|---|---|
| D1a: inter-difficulty mode collapse (fixed two-attractor) | ✅ Prevented | Conditioning on `d_i` + per-sample loss forces response to `d_i` |
| D1b: intra-difficulty lookup-table collapse (one formula per `d` bin) | ❌ Still possible | Sample `k > 1` candidates per `d_i` + pairwise distance penalty in loss |
| D2: hard-but-uninformative | ❌ Not addressed | Learning-progress signal (expensive) or value-head uncertainty proxy (cheap). Refinement, not pilot-blocker. |
| D3: non-differentiability | ✅ Handled | Continuous signal + filter + CE; never need gradient through sampling |
| D4: sync lag between Assessor and Prover updates | Soft | Per-sample gradient tracks moving `d → formula` mapping smoothly; still needs relative-rate tuning |
| D5: trivial fixed band | ✅ Gone | Replaced by `P(d)` |
| D6: cold-start (new) | Needs Phase 1 warm-start | Complexity-proxy pretraining solves it |
| D7: covariate shift (new) | ✅ Mitigated | Shared-weight multitask — Prover is the same model, has always seen what Assessor generates |
| D8: mode coverage gap (new) | Needs adaptive `P(d)` | Prioritize `d` where Assessor currently produces on-target candidates |

**Blockers remaining:** D1b (cheap: add pairwise-distance term to loss) and
D2 (refinement: skip for pilot, address post-plateau). Everything else is
either prevented by the design or has a concrete mitigation.

### Q3. Difficulty band calibration over time — PARTIALLY RESOLVED via Q2

Under the Q2 design, the "difficulty band" is no longer a scalar target in
the loss — it's the *shape* of `P(d)`, the distribution we draw `d_i` values
from. Calibration moves from "loss function design" to "sampling
distribution design", which is a much cleaner lever.

Remaining sub-question: **should `P(d)` be static or adaptive?**

- **Static** (uniform on `[0, 1]` or a beta distribution peaked at `0.5`):
  simplest, always covers the full range, never gets stuck.
- **Adaptive** (concentrate on `d` values where the Assessor's samples show
  non-zero learning progress for the Prover): POET / prioritized-replay
  style. Higher ceiling but risks selection-bias collapse if the adaptation
  rule is myopic.

Recommendation for pilot: **start static uniform, switch to adaptive only if
uniform plateaus**. Keep a diversity regularizer on `P(d)` (e.g., floor the
probability mass at all `d` values above some epsilon) to prevent collapse
onto a narrow band.

### Q4. Explicit differentiation from expert iteration — answered via S1's duplicate observation

The proposal currently positions Assessor against **random** generation.
That's a weak baseline. The strong baseline is **expert iteration with
on-the-fly augmentation** (DeepSeek-Prover-V1.5, AlphaProof, and — on the
SR side — Kamienny's DGSR+MCTS itself). All three implement an implicit
curriculum: only successful trajectories become training data.

**The answer to "why does adaptive generation beat adaptive filtering":**
expert iteration filters a **fixed problem distribution**. Once you've
extracted all the near-misses from that fixed pool, you've extracted
everything the pool offers. The Fagin-wall / duplicate-formula observation
from S1 is the concrete form of this: random pools run out of distinct
samples well before they run out of compute. Filtering a pool whose
distinct-sample count has saturated gives you no new training signal.

The Assessor **generates new samples conditioned on where the Prover's
frontier currently sits**, which is a pool that *grows with the Prover's
capability*. This extends the reachable frontier past any fixed pool's
ceiling. That's the one-sentence argument, and it connects Q4 directly to
the S1 diagnosis — they're the same wedge from two angles.

One honest caveat to include: the argument assumes the Assessor's generation
can actually produce novel frontier-level samples and not just memorize the
pretraining distribution. D6 (cold-start) and D8 (mode coverage gap) are the
failure modes where this assumption breaks. Both have fixes in Q2.

### Q5. Pilot success criterion

The 4-month SR pilot is supposed to de-risk the ATP milestones. What
measurable result demonstrates "Assessor works"?

Concrete options:
- **Sample-efficiency ratio**: Assessor variant matches baseline performance
  with X% fewer training samples.
- **Ceiling-breaking**: Baseline plateaus at P%; Assessor variant pushes to
  P + ΔP%, where the baseline cannot reach ΔP regardless of compute.
- **Head-to-head vs. expert iteration**: Same compute budget, Assessor variant
  beats vanilla expert-iteration MCTS.

**Recommendation:** option 3, head-to-head against expert iteration. That's
the comparison reviewers will care about, and option 2 (ceiling-breaking)
is the strongest single-result framing.

### Q6. Boolformer plateau as a precondition

Branch `mctx` is currently at ~13% success rate after 50+ iters of vanilla
expert iteration. Two interpretations:

1. **The loop is fine, the architecture/reward is wrong.** Plateau is a
   debugging issue (value collapse, action granularity, reward shape). Once
   debugged, vanilla expert iteration breaks the plateau; then Assessor is
   added and shows further gains.
2. **The loop has a structural ceiling.** Vanilla expert iteration cannot
   break the plateau without a different sample distribution. Assessor is
   what unblocks it.

These have **different proposal implications**. (1) means the SR pilot is
straightforward execution. (2) means Assessor's value is demonstrated
*directly* by the pilot. Either way, the next experimental step is the same:
**run vanilla expert iteration to convergence and document where it stalls.**
That gives a baseline against which Assessor's gain (if any) can be measured.

## Compute budget reality (for proposal §6)

Verified from primary sources (Kamienny 2022 Ch 6 training section;
Kamienny 2023 thesis §7.3.1, p. 85; `huang2025_paper.md` transcription
Appendix H):

| System | Pretraining | Per-problem inference | Self-funded feasible? |
|---|---|---|---|
| **Kamienny 2022 E2E** (Ch 6) | 32× V100 32GB × ~50 epochs × ~30 min/epoch ≈ several hundred GPU-hours (~hundreds of USD on current cloud pricing) | seconds (forward pass only) | No |
| **Kamienny 2023 DGSR+MCTS** (Ch 7) | 8 GPUs × 12 days pretraining + 8 GPU-days expert iteration ≈ **104 GPU-days** | ≤ 24 h per dataset on 8-node cluster (4 trainers + 4 MCTS workers) | No |
| **Huang 2025** | **Zero** (classical MCTS, no learned model) | 10.6 TFLOPS FP32 machine, 48 h or 500K evals per problem | **Yes — laptop/workstation class** |

**Huang 2025 is confirmed learning-free.** Algorithm 1 is pure classical
MCTS with UCB-extreme bandit + GP-style state-jumping. No transformer, no
pretraining, no learned value/policy. The only numerical optimization is
BFGS for real-valued constants inside candidate expressions. Full confirmation
in `huang2025_paper.md` with page citations.

**Implication for PI B (self-funded).** The proposal's SR pilot must fit on
a single rented GPU (currently RTX 5090 on vast.ai for short bursts). Three
viable routes:

1. **Scale down** — depth-4 Boolean formulas, 59M-parameter model. Already
   running on the `mctx` branch.
2. **Start from a public checkpoint** if one exists (the released Kamienny
   2022 weights would work; check availability).
3. **Classical-MCTS baseline** — run Huang 2025's algorithm (released at
   github.com/PKU-CMEGroup/MCTS-4-SR) on the Boolean problem as a
   zero-pretraining control. **Free experiment.**

Route (3) doubles as the **Q5 pilot baseline**: if Huang-style classical
MCTS breaks the 13% Boolformer plateau without any pretraining, the plateau
cause is search strategy / sample distribution, not model capacity. If it
doesn't, the plateau is in the learned component.

**Budget framing for §6**: the proposal's compute ask (~$15K cloud GPU)
should be framed explicitly in terms of the scaled-down Boolean pilot and
the follow-on Lean tactic, not in terms of matching Kamienny-scale
pretraining. Compare favourably to Kamienny's ~100 GPU-day cost as evidence
of frugal design, not a sign of under-resourcing.

## Action items

In rough priority order:

1. **Run vanilla expert iteration on Boolformer to convergence**, document
   the plateau and the failure mode (value collapse? policy entropy collapse?
   reward saturation?). This is the empirical baseline for Q5/Q6 and is
   independent of any proposal-revision work.
2. **Count distinct truth tables in the current Boolformer training pool.**
   Directly tests the S1 duplicate-formulas hypothesis as a plateau cause.
   One-line script, takes an hour.
3. **Run Huang 2025's classical MCTS on the Boolean problem** as a
   zero-pretraining control. Doubles as the Q5 pilot baseline (see §Compute).
4. **Implement Phase 1 Assessor pretraining** with complexity-proxy
   difficulty labels. Small scope, testable on the existing CPU config. No
   Prover-in-the-loop needed yet.
5. **Implement Phase 2 expert-iteration loop** (generate-filter-train) with
   truth-table distance as the continuous difficulty signal. Add k>1 sampling
   per `d_i` + pairwise distance penalty to close D1b.
6. **Decide Q1 (amplification operator)** — most likely MCTS — and write a
   paragraph in the proposal connecting it explicitly to AlphaZero's policy
   improvement step. The training signal is already locked (CE on filtered
   successes, per Q2); only the search strategy remains to choose.
7. **Reframe abstract** to lead with diagnosis (Fagin + duplicate observation),
   then shared-weight multitask intervention, then compute-frugal positioning.
   AlphaZero stays as a *spirit* reference unless Q1 is resolved with a real
   amplification operator.
8. **Consider deprioritizing the compiler-optimization arm** to deepen the
   Lean tactic milestone. Reviewers reward depth over breadth.
9. **Update `kamienny_notes.md`** with the newly-cited pages (61, 79, 82,
   84–85) and the Table 5.1 quote. Existing notes already cover Ch 7 but
   don't cite Table 5.1 directly — it's the cleanest primary-source citation
   for the "no REINFORCE" design choice.

Action items 1–3 are concretely doable right now and produce data that
de-risks the proposal independently of any writing. 4–5 are the Assessor
pilot implementation. 6–8 are proposal-text edits. 9 is a small notes update.
