# Boolformer MCTS Training - TODOs

## Model Configuration

### 0. Precision: Use FP16 not BF16
**Reference**: Current discussion on x.com

Model parameters and variables should use **FP16** (float16) precision, not BF16 (bfloat16).

Reason: Better numerical stability and performance characteristics for this use case.

## Performance Optimizations

### 1. ~~Memory: Encoder Output Storage~~ DONE (commit 8d19b33)
Removed `encoder_output` and `points` from `BoolformerState`. They are now passed via mctx `params` (not stored per tree node) instead of `embedding` (stored per node). Training re-encodes from `points` to maintain gradient flow through encoder.

### 2. ~~Reward Signal: Softer Gradients~~ DONE
Using mean IoU (Jaccard Index) as reward via `evaluate_formula_balanced_accuracy()`.

## Reward & Training Improvements

### 4. Raw IoU + Above-Average Boost (IN PROGRESS)
Two improvements to break out of the training plateau (~13% success, no improvement over 50+ iters):

**Leg 1: Above-average boost (replacing binary is_perfect boost)**
- Switch from adjusted IoU to **raw IoU** everywhere (MCTS + training)
- Track EMA baseline of mean raw IoU across selfplay iterations
- Pool sampling: boost samples with `value_target > baseline` (not `is_perfect`)
- `min_success_ratio_per_length` reused: ensures ≥30% of training batch is above-average
- Do NOT over-boost perfect samples (they're easy / already known)

**Leg 2: Adaptive push (TODO — implement after Leg 1)**
- For below-baseline episodes, add positive adjustment to value_target
- Prevents value head collapse: can't settle into "predict average and stop"
- Creates continuous pressure to improve (yesterday's average = today's zero)

**Root cause analysis:**
- `protect_l4_successes` filled pool with stale successes (~92%) → value head collapsed to predict 1.0 → FIXED (commit 64642e2)
- Adjusted IoU + reward-weighted policy loss muted 70% of episodes → model only reinforced what it already solved → no improvement
- Value head accurately predicted mediocre performance → low loss → no gradient → stable equilibrium at ~13%
- Model predicts, doesn't solve. Predictions converged. No mechanism to push for better solving.

### 5. ~~Value/Policy Loss Balance~~ DONE (commit b91078e)
10x value loss weight balances backbone gradients (CE policy ~10x larger than MSE value).

## JAX Conversion

### 3. Formula Generation
**Location**: `TODO_JAX_FORMULA_GEN.md`

Formula generation uses Python/PyTorch (upstream Boolformer `src/`). Parallelized with multiprocessing (16 cores, ~2s for 128 formulas). PyTorch is CPU-only to avoid CUDA version conflicts with JAX.

Still TODO — JAX-ify to eliminate Python/multiprocessing overhead:
- Formula tree generation (recursive → iterative with scan/while_loop)
- Formula evaluation (already done in environment.py)
- Token conversion (string → integer IDs)
- Negation logic (count 0s vs 1s)

## Knowledge base

- `papers/INDEX.md` — catalog of PDFs in `papers/` with per-paper status and links to notes files. Canonical navigational layer; keep this up to date when adding papers or writing notes.
- `karpathy.md` — reference for the LLM-maintained knowledge-base workflow we're loosely adopting (raw PDFs → curated md notes, queryable by Claude across sessions).
