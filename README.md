# Boolformer — research fork (Ruhollah Majdoddin)

I use Boolean symbolic regression as a testbed for training a transformer with MCTS-guided
self-play, AlphaZero-style, so that search generates the model's training signal. The goal is
to carry the method to automated theorem proving
([proposal](https://docs.google.com/document/d/1JQZRALrOsStfjVx3u_9HkO01pOU6XVYJV5PTWqo6gvk/edit?usp=sharing));
Boolean formulas give a cheap, exactly verifiable reward to develop it on.

## MCTS self-play training of Boolformer (JAX)

Code: [Majdoddin/mctx, branch `boolformer-example`](https://github.com/Majdoddin/mctx/tree/boolformer-example/examples/boolformer)

- Boolformer re-implemented in JAX/Flax NNX. Gumbel MuZero search
  ([mctx](https://github.com/google-deepmind/mctx)) builds each formula token by token; the
  reward is the IoU between the formula's truth table and the target's. The policy head learns
  the search's action weights and the value head the episode's final reward, from a replay pool
  sampled by formula length.
- The whole self-play episode is JIT-compiled (about 7× faster per episode), the encoder output
  travels in the search parameters instead of being copied into every tree node, and training
  formulas are generated in parallel across CPU cores.
- Work in progress: training plateaus at about 13% exact solutions on length-4 formulas,
  against about 10% untrained. Two causes found so far: protecting rare successes in the replay
  pool filled it with stale successes and collapsed the value head to predicting 1.0; and
  weighting the policy loss by reward silenced about 70% of episodes, so the model only
  reinforced formulas it could already solve. Current approach: raw IoU against a running
  (EMA) baseline, with extra weight on episodes that beat it.

## Investigating MCTS for symbolic regression (Huang et al., NeurIPS 2025)

Code: [Majdoddin/MCTS-4-SR, branch `benchmark/seed-bias`](https://github.com/Majdoddin/MCTS-4-SR/tree/benchmark/seed-bias) ·
notes: [`papers/`](https://github.com/Majdoddin/Boolformer-arthurenard/tree/mctx/papers) ·
upstream: [merged PRs](https://github.com/PKU-CMEGroup/MCTS-4-SR/pulls?q=author%3Amajdoddin), [issue #4](https://github.com/PKU-CMEGroup/MCTS-4-SR/issues/4)

- **Fairer sibling comparisons.** The search ranks sibling subtrees by single random rollouts,
  so a good subtree can lose to bad luck. I added matched-pair re-evaluation (common random
  numbers): once all of a node's children exist, they are re-scored under the same random
  completions. It needed 2.4× fewer evaluations on 10 of the standard benchmark seeds but 2.3×
  more on fresh random seeds, so the gain came from the seed sample, not the method.
- **Seed sensitivity at scale.** To settle that, I ran 100 seeds on 34 problems (Nguyen,
  Livermore) under three setups: MT19937 with the standard SRBench seeds, MT19937 with
  quantum-random seeds (ANU QRNG), and PCG64-DXSM with quantum seeds. All three agree within
  0.3%. After publication the repo's defaults moved toward more GP mutation and less random
  exploration (gp_rate 0.2→0.5, exploration rate 0.2→0.1); under these settings the earlier
  seed sensitivity does not appear.
- **Where it still fails.** The remaining failures on these benchmarks need deeper trees than
  the depth limit allows, need a constant the benchmark's operator set omits, or are
  constant-fitting convergence misses. None is a search failure, so for this method the
  ground-truth benchmarks are saturated.
- **A search-dynamics puzzle.** Changing the condition that triggers backpropagation, which a
  monotonicity argument says cannot change results, reproducibly costs about 70% more
  evaluations on Nguyen-3. The cause is still open
  ([write-up](https://github.com/Majdoddin/Boolformer-arthurenard/blob/mctx/papers/propagate_before_backpropagate.md)).
- **Upstream regression report.** Commit `a5c87f9`, presented as a speed-up, changes search
  behavior ([issue #4](https://github.com/PKU-CMEGroup/MCTS-4-SR/issues/4)).
- **A "black-box" benchmark with a known formula (early result).** SRBench's `bodyfat` target
  is Siri's equation, `495/Density − 450`, applied to one of the input features; only 5 of 252
  rows deviate, and those are data errors. Each run's test R² is therefore set by how many of
  those 5 rows land in its test split: on all 5 seeds checked, Siri's own test R² matches the
  method's within 0.007. Injecting 1% noise into the targets during constant fitting made one
  of the 5 seeds recover the generating formula, `494/Density − 449`, where the unregularized
  run returned a 51-node expression.

## Boolformer architecture experiments (this repo)

Branches `feature/*`: an adversarial self-curriculum in which the model, in a `<gen>` mode,
proposes formulas and is rewarded when its own `<regress>` mode fails on them (WIP); full
truth-table input with a linear projection and {−1, +1} encoding; a single start token with
formula negation; Möbius-transform spectral features (WIP); dynamic encoder and decoder
padding; a Hugging Face EncoderDecoder backbone.

## About this code

This repo builds on the code release for *Boolformer: Symbolic Regression of Logic Functions
with Transformers* (d'Ascoli, Renard, Papadopoulos, Susskind, Bengio, Abbé, 2025;
[arXiv:2309.12207](https://arxiv.org/abs/2309.12207)), which was published at
github.com/arthurenard/Boolformer and is no longer online. Stéphane d'Ascoli's original 2023
implementation is at [sdascoli/boolformer](https://github.com/sdascoli/boolformer). Code not
listed above is the original authors'.

---

## Original README of the code release

This repository contains code for the paper '**Boolformer: symbolic regression of Boolean functions with transformers**'

### Installation

#### Using UV (Recommended)

1. First, install UV:

```bash
pip install uv
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
uv venv

# Activate it (Unix/MacOS)
source .venv/bin/activate
# or on Windows
.venv\Scripts\activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

### Training and evaluation

#### Training Models

To run the training of the noiseless model:

```bash
uv run python scripts/train.py -r <run_name> \
    -t config/transformer/noiseless.py \
    -f config/formula/noiseless.py \
    --nw <num_workers> \
    --bd <backup_directory> \
    --bs <batch_size> \
    --be <backup_every> \
    -d <device_id>
```

To run the training of the noisy model:

```bash
uv run python scripts/train.py -r <run_name> \
    -t config/transformer/noisy.py \
    -f config/formula/noisy.py \
    --nw <num_workers> \
    --bd <backup_directory> \
    --bs <batch_size> \
    --be <backup_every> \
    -d <device_id>
```

All hyperparameters not specified in the command arguments are contained in the configuration files, found inside `./config`.

Models are automatically saved during training in `.ckpt` format. To run the evaluation benchmark, the `.ckpt` file containing the model weights is needed.

#### Run Predictions

To run prediction on synthetic data:

```bash
uv run python scripts/predict.py \
    --model <path_to_.ckpt> \
    -f <path_to_the_config> \
    --bs <batch_size> \
    --nw <num_workers> \
    -d <cuda_device_id> \
    -s <size> \
    --output <output_folder_path> \
    --beam <beam_size>
```

#### PMLB Benchmark

To run the PMLB Benchmark:

```bash
uv run python scripts/benchmarks/pmlb/compute_results.py \
    --model <path_to_.ckpt> \
    --config config/formula/noisy.py \
    --beam <beam_size> \
    --nb_smpl <resampling_count> \
    --output <output_folder_path> \
    --device <cuda_device_id>
```

#### GRN Benchmark

To run the GRN benchmark, please build the Docker image with `docker build -t grn .`. Spin up a container with `docker run -d --gpus all -it --name grnbench grn sleep infinity`. Attach to the container.

Inside the container, you will need to run `internal_run.py`, which is located in `/reviewAndAssessment/xsolver`. For the benchmark to work, it needs access to the model saved in `.ckpt` format.

The command:

```bash
python3 internal_run.py --device <inference_device> --model_path <path_to_.ckpt>
```

will generate the formulas for the GRN using Boolformer. Results are saved in `./results/Ecoli`.

Finally, run `gen_plots.py` to generate the radial plots.

#### Generating Graphs

The repository includes several scripts for generating graphs to visualize results:

##### Noiseless Model Graphs

To generate graphs for noiseless model results:

```bash
uv run python scripts/graphs/noiseless_graphs.py \
    --input <path_to_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging> \
    --ft_size <font_size>
```

This will generate graphs showing the relationship between binary operators and metrics, formula distribution, and active variables impact.

##### Noisy Model Graphs

To generate graphs for noisy model results:

```bash
uv run python scripts/graphs/noisy_graphs.py \
    --input <path_to_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging>
```

This will create various graphs showing how different metrics (accuracy, F1 score, perfect recovery) are affected by factors like active variables, inactive variables, number of input points, and flip probability.

##### Temperature Analysis Graphs

To generate graphs analyzing the effect of temperature on model performance:

```bash
uv run python scripts/graphs/temperature_graphs.py \
    --input <path_to_temperature_results.json> \
    --output <output_directory> \
    --num_points <number_of_points_for_averaging>
```

This will create graphs showing how different metrics vary with temperature settings.

### Common Issues

If you encounter any UV-related warnings about hardlinking, you can suppress them by setting:

```bash
export UV_LINK_MODE=copy
```

or by using the `--link-mode=copy` option when running UV commands.
