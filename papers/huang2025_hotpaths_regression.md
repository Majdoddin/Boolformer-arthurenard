# Upstream hot-paths commit: functional change disguised as optimization

**GitHub issue:** [PKU-CMEGroup/MCTS-4-SR#4](https://github.com/PKU-CMEGroup/MCTS-4-SR/issues/4)

## Summary

Upstream commit `a5c87f9` ("Optimize evaluation hot paths and add profiling hooks") is labeled as a performance optimization but changes search behavior. It is incompatible with PCG64DXSM, causing up to 39× regression on individual seeds. The pre-selected benchmark seeds mask this problem because they are favorable to MT19937.

## Commits tested

| Label | Commit | RNG | Description |
|---|---|---|---|
| Base | `0d2a771` | MT19937 | Upstream after PR#1+PR#2 merged, hot paths reverted |
| Hot paths | `a5c87f9` | MT19937 | Upstream hot paths re-applied |
| Upstream latest | `73f18d4` | MT19937 | Latest upstream (identical C++ to `a5c87f9`) |
| PCG no hot paths | `4e22c61` | PCG64DXSM | `matched-pair-baseline` branch, before matched-pair |
| PCG + hot paths | `2c4954b` | PCG64DXSM | Main branch, PCG on top of hot paths |
| PCG + hot paths + MP | `21d7435` | PCG64DXSM | Main branch, PCG + matched-pair on hot paths |

All tests: Nguyen-3, `--gp-rate 0.2 --lm-iterations 50 --max-constants 6 --c 4.0 --exploration-rate 0.2 --ops "+,-,*,/,sin,cos,exp,log,R"`, max_evals=2M.

## Changes in `a5c87f9`

1. **exp_queue**: Linear-scan duplicate check → binary search (`lower_bound` + 2-neighbor check)
2. **node.hpp backpropagate**: O(n²) incremental prepend → O(n) pre-built full path with subspan
3. **node.hpp random_child**: Temporary vector → two-pass count-then-index
4. **node.hpp propagate**: Vector copy → span subspan
5. **optimizer.cpp**: Removed `working_tree = tree_template` reset per LM iteration
6. **interpreter.cpp**: Fused Jacobian expressions (separate `local_deriv` → inline multiply-accumulate)

Each appears mathematically equivalent. Combined, they change eval counts and formula structures.

## Experiment 1: Pre-selected seeds, MT19937, before vs after hot paths

Seeds: 23654, 15795, 860, 5390, 16850, 29910, 4426, 21962, 14423, 28020 (the benchmark's first 10).

| Seed | Base (`0d2a771`) evals | Hot paths (`a5c87f9`) evals | Base complexity | Hot paths complexity |
|---|---|---|---|---|
| 23654 | 197,735 | 118,237 | 33 | 33 |
| 15795 | 117,120 | 83,316 | 24 | 24 |
| 860 | 132,295 | 37,313 | 22 | 22 |
| 5390 | 133,751 | 116,466 | 25 | 24 |
| 16850 | 302,843 | 58,205 | 22 | 24 |
| 29910 | 63,440 | 166,683 | 36 | 22 |
| 4426 | 333,790 | 168,183 | 58 | 22 |
| 21962 | 136,779 | 95,706 | 22 | 20 |
| 14423 | 51,067 | 58,776 | 155 | 22 |
| 28020 | 173,614 | 199,601 | 22 | 49 |
| **Mean** | **164k** | **110k** | 42 | 26 |

Structural recovery: 5/10 → 8/10 exact polynomial. Looks like an improvement.

## Experiment 2: Pre-selected seeds, PCG, before vs after hot paths

Same 10 seeds. PCG no hot paths = `4e22c61`, PCG + hot paths = `2c4954b`.

| Seed | PCG no hot paths | PCG + hot paths | Ratio |
|---|---|---|---|
| 23654 | 48,703 | 150,549 | 3.1× worse |
| 15795 | 24,424 | 85,294 | 3.5× worse |
| 860 | 35,941 | 167,786 | 4.7× worse |
| 5390 | 35,568 | 303,766 | 8.5× worse |
| 16850 | 58,295 | 50,035 | 0.9× better |
| 29910 | 47,191 | 120,650 | 2.6× worse |
| 4426 | 17,808 | 204,611 | 11.5× worse |
| 21962 | 12,062 | 95,706 | 7.9× worse |
| 14423 | 54,195 | 372,693 | 6.9× worse |
| 28020 | 39,679 | 1,550,256 | 39× worse |
| **Mean** | **37k** | **321k** | **8.7× worse** |

Hot paths destroys PCG performance. 9/10 seeds worse, one seed 39× worse.

## Experiment 3: Truly random seeds, MT19937, before vs after hot paths

Seeds from `/dev/urandom`: 663110, 5879382, 1162155, 6707899, 5005913, 723294, 2510859, 437735, 2160600, 1411505.

| Seed | Base | Hot paths | Ratio |
|---|---|---|---|
| 663110 | 79,720 | 82,801 | 1.0× |
| 5879382 | 209,754 | 63,064 | 0.3× better |
| 1162155 | 469,251 | 196,299 | 0.4× better |
| 6707899 | 223,930 | 111,734 | 0.5× better |
| 5005913 | 644,613 | 107,513 | 0.2× better |
| 723294 | 35,336 | 122,680 | 3.5× worse |
| 2510859 | 132,969 | 62,119 | 0.5× better |
| 437735 | 364,993 | 248,795 | 0.7× better |
| 2160600 | 135,890 | 110,303 | 0.8× better |
| 1411505 | 278,970 | 1,811,643 | 6.5× worse |
| **Mean** | **258k** | **292k** | **1.1× worse** |

With random seeds, hot paths is neutral on MT19937. The improvement on pre-selected seeds (164k → 110k) does not generalize.

## Experiment 4: PCG + hot paths + matched-pair

Commit `21d7435` (matched-pair on top of hot paths + PCG). Single seed 23654.

| Config | Evals |
|---|---|
| PCG, no hot paths, no MP (`4e22c61`) | 48,703 |
| PCG, no hot paths, MP N=4 (`bdc28a1`) | 73,798 |
| PCG, hot paths, no MP (`2c4954b`) | 150,549 |
| PCG, hot paths, MP N=4 (`21d7435`) | 588,082 |
| PCG, hot paths, MP N=4, lm=100 (`21d7435`) | 2,000,000 (budget) |

Each layer makes it worse. Hot paths alone: 3×. Adding matched-pair: 4× more. Adding lm=100: hits budget.

## Experiment 5: Matched-pair on random seeds — controlled comparison (2026-05-12)

**Question:** Does matched-pair's 2.35× speedup on pre-selected seeds generalize to unbiased seeds?

**Design:** Same code (`bdc28a1`), same RNG (PCG64DXSM), same 10 seeds — only variable is `kMatchedPairN` (0 vs 4). This isolates matched-pair's effect with no confounds.

Seeds from `secrets.randbits(32)`: 996657091, 1903754000, 1494059780, 952885591, 772058839, 3728297749, 988928067, 2251174867, 3170449109, 1817236458.

Branch `experiment/random-seeds-mp-baseline` (commit `7fc9100`).

| Seed | PCG only (N=0) evals | PCG only time | PCG+MP (N=4) evals | PCG+MP time | MP effect |
|---|---|---|---|---|---|
| 996657091 | 151,238 | 17.2s | 747,899 | 87.0s | 4.9× worse |
| 1903754000 | 286,915 | 30.0s | 267,792 | 48.3s | 0.9× better |
| 1494059780 | 217,650 | 24.6s | 96,667 | 16.0s | 2.3× better |
| 952885591 | 143,072 | 15.0s | 316,581 | 55.8s | 2.2× worse |
| 772058839 | 156,776 | 15.9s | 48,648 | 7.2s | 3.2× better |
| 3728297749 | 129,640 | 13.6s | 269,778 | 45.4s | 2.1× worse |
| 988928067 | 83,716 | 8.6s | 415,715 | 70.6s | 5.0× worse |
| 2251174867 | 567,013 | 63.2s | 101,342 | 16.4s | 5.6× better |
| 3170449109 | 77,245 | 8.0s | 2,000,000 | 253.6s | **25.9× worse** |
| 1817236458 | 56,594 | 5.6s | 51,916 | 6.8s | 1.1× better |
| **Mean** | **187k** | **20.2s** | **432k** | **60.7s** | **2.3× worse** |
| **Success** | **10/10** | | **9/10** | | |

Total wall time: PCG only = 201.8s, PCG+MP = 607.2s.

Seed `3170449109` converges easily without MP (77k, 8.0s) but hits the 2M budget with MP (253.6s). 5 seeds are worse with MP, 5 better — but the "worse" cases are far more extreme (up to 26×) than the "better" ones (up to 5.6×).

Seed `1817236458` with PCG-only has `test_r2=0.932` (overfit); with MP it achieves `test_r2=1.000`. So MP sometimes improves structural quality at the cost of higher mean evals.

### Same code, same RNG: pre-selected vs random seeds

| Config | Pre-selected seeds | Random seeds |
|---|---|---|
| PCG only (N=0) | 228k | **187k** |
| PCG+MP (N=4) | **97k** | 432k |
| MP speedup | **2.35×** | **0.43× (2.3× slower)** |

Matched-pair's apparent 2.35× speedup on pre-selected seeds **reverses** to a 2.3× slowdown on random seeds. The pre-selected seeds are a favorable draw for MP, not representative of general performance.

### Reproduce

Branch `experiment/random-seeds-mp-baseline` on `majdoddin` remote (`Majdoddin/mcts4sr`). Hyperparameters same as all experiments in this doc (see §"Commits tested").

```bash
# MP N=4 run
git checkout experiment/random-seeds-mp-baseline
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
pip install -e .
python -m iMCTS.benchmarks --group Nguyen --cases Nguyen-3 \
  --seeds 996657091,1903754000,1494059780,952885591,772058839,3728297749,988928067,2251174867,3170449109,1817236458 \
  --gp-rate 0.2 --lm-iterations 50 --max-constants 6 --c 4.0 --exploration-rate 0.2 \
  --ops "+,-,*,/,sin,cos,exp,log,R"

# MP N=0 run: edit source/mcts/mcts.cpp, set kMatchedPairN = 0, rebuild and rerun
```

## Pre-selected seed bias

MT19937 has poor single-integer seed diffusion (known defect, why NumPy replaced it in 2019). The benchmark's 100 seeds are all < 33000.

| Config | Pre-selected seeds | Random seeds |
|---|---|---|
| MT19937, no hot paths | 164k | 258k |
| MT19937, hot paths | 110k | 292k |
| PCG, no hot paths (N=0) | 228k | 187k |
| PCG, matched-pair (N=4) | 97k | 432k |

The bias affects all configurations, but asymmetrically: pre-selected seeds make matched-pair look 5.4× better than it actually is (2.35× speedup → 2.3× slowdown).

## Conclusions

1. **`a5c87f9` is not a pure optimization.** It changes search behavior on every seed tested. Labeled "optimize" with no documentation of functional changes.

2. **The improvement on pre-selected seeds does not generalize.** With random seeds, hot paths is neutral (258k → 292k). The apparent improvement (164k → 110k) is an artifact of seed selection.

3. **Hot paths is incompatible with PCG.** Mean evals 37k → 321k (8.7× worse). This blocks adoption of PCG (which itself gives 4–7× improvement over MT19937).

4. **Pre-selected benchmark seeds are biased.** The bias inflates not only upstream results but also our own: matched-pair's 2.35× speedup on pre-selected seeds reverses to a 2.3× slowdown on random seeds (Experiment 5).

5. **PCG without matched-pair is the best configuration on random seeds.** Mean 187k, 10/10 success, 20.2s total. Adding matched-pair makes it 2.3× worse. The earlier recommendation of `bdc28a1` (PCG+MP) as best config was based on biased seeds and is retracted.
