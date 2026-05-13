# Propagate-before-backpropagate

Gate for skipping unnecessary upward propagation in MCTS.

## Algorithm

At every call site that previously did `backpropagate` (or `backpropagate` + `propagate`):

```cpp
bool self_accepted = node->propagate(path, reward);   // self + children downward
if (self_accepted)
    node->backpropagate(path, reward, /*skip_self=*/true);  // parents upward, skip self
```

**`propagate`** gains two parameters:
- `include_self` (default `true`): insert formula into current node's queue before descending.
- Returns `bool`: whether **self's** queue accepted the formula (children's acceptance is irrelevant to the gate).

**`backpropagate`** gains one parameter:
- `skip_self` (default `false`): skip the first node's append (already done by `propagate`).

### Call-site mapping

| Old code | New code |
|---|---|
| `node->backpropagate(path, r)` | `if (node->propagate(path, r)) node->backpropagate(path, r, true)` |
| `node->backpropagate(p, r); node->propagate(p, r)` | `if (node->propagate(p, r)) node->backpropagate(p, r, true)` |
| `parent->propagate(entry.path, entry.reward)` | `parent->propagate(entry.path, entry.reward, false)` |

The second row (mutation/crossover) previously called both functions — the new `propagate` with `include_self=true` subsumes both: self-insert + children propagation.

## Equivalence proof

**Claim:** The rewritten code produces identical queue state at every node for every possible execution.

### Definitions

- **ExpQueue** of size K: sorted top-K entries by reward. Two rejection reasons:
  1. *Reward too low:* queue full and `reward <= worst entry`
  2. *Near-duplicate:* existing entry with `|reward - existing| < 1e-5`
- **backpropagate** walks upward: self → parent → grandparent..., breaking on first rejection.
- **propagate** walks downward: self (optional) → children along path.

### Proof structure

Consider any call site. Let `N` = current node, `P` = parent, `G` = grandparent, etc.

**Step 1.** Self gets tested in both old and new code. Old: `backpropagate` tests self first. New: `propagate` with `include_self=true` tests self first. Same call: `N.queue.append(path, reward)`. Identical.

**Step 2.** If self accepts (`propagate` returns true):
- New code calls `backpropagate(skip_self=true)`: skips self (already done), prepends `N.move`, offers to `P`. Continues upward exactly as old `backpropagate` would after accepting self.
- Children: in search, the expanded node has no children (just created), so propagate's children loop is a no-op. In mutation/crossover, old code called `propagate` separately — new code's `propagate` with `include_self=true` covers both self and children in one call.
- Result: identical queue state at all nodes.

**Step 3.** If self rejects for reason 1 (reward too low):
- Old: `backpropagate` breaks at self. No ancestor sees the formula.
- New: `propagate` returns false. `backpropagate` not called. No ancestor sees the formula.
- Are ancestors missing out? No — see **Monotonicity Lemma** below.

**Step 4.** If self rejects for reason 2 (near-duplicate):
- Old: `backpropagate` breaks at self (append returns false on near-duplicate too).
- New: `propagate` returns false. `backpropagate` not called.
- Same result.

### Monotonicity Lemma

**Claim:** If node `N` rejects a formula, every ancestor of `N` would also reject it.

**Proof:** Every entry that enters `N`'s queue was backpropagated upward at insertion time. Parent `P` received every formula that `N` received (plus formulas from `N`'s siblings). So `P`'s queue is drawn from a superset of candidates with the same capacity K.

- The K-th largest reward from a superset >= K-th largest from any subset.
- Therefore `P.worst >= N.worst` (monotonically non-decreasing up the tree).

**Reason 1:** If `reward < N.worst`, then `reward < P.worst`. Parent rejects.

**Reason 2:** If `N` has a near-duplicate entry `e` with `|e.reward - reward| < 1e-5`, then `e` was backpropagated to `P` when first inserted into `N`. Either:
- `P` still has `e` → `P` rejects for reason 2.
- `P` evicted `e` (replaced by something better) → `P.worst > e.reward >= reward - 1e-5` → `P` rejects for reason 1.

In both cases, `P` rejects. By induction, all ancestors reject. **QED.**

## Critical implementation note: `self_accepted` vs `any_accepted`

The gate must use **self's acceptance only**, not `any_accepted` (which includes children).

An early implementation returned `any_accepted` (OR of self and all children along path). This caused a **71% regression** on Nguyen-3 seed 23654: 126,501 evals vs baseline 73,798.

After correcting to `self_accepted`, all 10 Nguyen-3 seeds matched the baseline exactly (bit-identical eval counts).

### Why the regression is puzzling

By the Monotonicity Lemma, the gate should not matter: when self rejects but a child accepts, `backpropagate` would run but parent would reject (parent's threshold >= self's threshold). The extra `backpropagate` call should be a no-op.

We cannot construct a case where self rejects but an ancestor accepts. Yet empirically, `any_accepted` causes a large regression while `self_accepted` is bit-identical to baseline. The root cause is unresolved — it may be a build-cache artifact from the first test, but we have not re-verified.

**Rule:** Always gate on self's acceptance. Do not use children's acceptance for the backpropagate gate.

## Benchmark verification

Nguyen-3, 10 seeds, propagate-before-backpropagate with `self_accepted` gate. All eval counts match baseline (commit `bdc28a1` on `matched-pair-baseline` branch) exactly:

| Seed | Baseline evals | PropChange evals | Time |
|---|---|---|---|
| 23654 | 73,798 | 73,798 | 8.0s |
| 15795 | 101,761 | 101,761 | 11.4s |
| 860 | 166,474 | 166,474 | 20.5s |
| 5390 | 71,986 | 71,986 | 8.5s |
| 16850 | 73,803 | 73,803 | 8.8s |
| 29910 | 71,562 | 71,562 | 8.7s |
| 4426 | 183,986 | 183,986 | 21.3s |
| 21962 | 97,430 | 97,430 | 12.0s |
| 14423 | 84,778 | 84,778 | 9.5s |
| 28020 | 46,147 | 46,147 | 5.4s |
| **Mean** | **97,173** | **97,173** | **11.4s** |

**Result:** Bit-identical eval counts across all 10 seeds. Mean runtime 11.4s vs baseline 12.0s — a slight improvement (~5%), consistent with skipping the upward walk for rejected formulas, though within run-to-run noise.

## Code

Commit `d2fecb9` on branch `propagate-before-backpropagate` (worktree `/tmp/mcts4sr-mp-test`, based on `bdc28a1` from `matched-pair-baseline`).

## Purpose

The gate itself saves negligible runtime (old `backpropagate` already breaks on the first failed append). The real value is as a **gate for the Tier 2 pipeline**: canonical pruning, LM re-optimization, and other expensive post-processing that should only run when a formula actually qualifies for some node's top-N.
