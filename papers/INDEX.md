# Papers Index

Catalog of PDFs in this folder and their notes files. Each row: citation,
arXiv ID, venue, one-sentence summary, status, link to notes file (if any).

**Status key**: `noted` = notes file exists and is current; `skimmed` = PDF
read / I know what's in it but no notes yet; `unread` = in raw, not yet
touched.

## Core — Boolformer MCTS training

| Paper | arXiv | Venue | Summary | Status | Notes |
|---|---|---|---|---|---|
| d'Ascoli, Renard, Papadopoulos, Susskind, Bengio, Abbé. **Boolformer: Symbolic Regression of Logic Functions with Transformers** | [2309.12207](https://arxiv.org/abs/2309.12207) | ICML-W 2025 (v2 Jul 2025) | Transformer-based E2E symbolic regression of Boolean functions from truth tables. The paper this repo's model architecture is based on. | skimmed | — |
| Kamienny, d'Ascoli, Lample, Charton. **End-to-end Symbolic Regression with Transformers** | [2204.10532](https://arxiv.org/abs/2204.10532) | NeurIPS 2022 | Kamienny's earlier pure-seq2seq SR paper (Ch 6 of his thesis). Collapses the two-step skeleton+BFGS procedure into one transformer forward pass. Direct baseline of the MCTS paper (the `@∞` row of Table 7.3). Same architecture family as Boolformer. | noted | [kamienny_notes.md](./kamienny_notes.md) |
| Kamienny, Lample, Lamprier, Denoyer, d'Ascoli, Charton. **Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search** | [2302.11223](https://arxiv.org/abs/2302.11223) | ICML 2023 | Pretrained transformer mutation policy + AlphaZero-style MCTS over complete expressions. Closest prior art to this repo's architecture. Key ablation: mutation size 10 beats size 1 by ~20 points. | noted | [kamienny_notes.md](./kamienny_notes.md) |
| Kamienny. **Efficient adaptation of reinforcement learning agents: from model-free exploration to symbolic world models** | tel-04391194 | Sorbonne PhD thesis 2023 | Full thesis version of above. Ch 7 = DGSR+MCTS (main source of our notes), Ch 8 = symbolic dynamics for MBRL, Appendix F = recurrent formulas (older work). | noted | [kamienny_notes.md](./kamienny_notes.md) |
| Huang, Huang, Xiao, Ma, Ming, Shi, Wen. **Improving Monte Carlo Tree Search for Symbolic Regression** | [2509.15929](https://arxiv.org/abs/2509.15929) | NeurIPS 2025 | Classical token-level MCTS augmented with (1) extreme-bandit allocation with finite-time regret bounds under polynomial reward decay, (2) state-jumping actions — GP mutation/crossover applied to top-N stored trajectories per node, with bidirectional propagation. Most recent MCTS+SR SOTA. Code: github.com/PKU-CMEGroup/MCTS-4-SR. | skimmed | — |

## Adjacent / reference

| Paper | arXiv | Venue | Summary | Status | Notes |
|---|---|---|---|---|---|
| Charton. **Int2Int: a framework for mathematics with transformers** | [2502.17513](https://arxiv.org/abs/2502.17513) | arXiv preprint 2025 | Open-source PyTorch framework for transformer-based math research (number theory focus). User manual / reference implementation, not directly applicable to Boolformer. | skimmed | — |
| Charton. **Transformers know more than they can tell — Learning the Collatz sequence** | [2511.10811](https://arxiv.org/abs/2511.10811) | arXiv preprint 2025 | Interpretability case study: how transformers learn long Collatz steps, residual-class learning dynamics, base-dependence of accuracy (99.7% at base 24/32 vs 25% at base 3). Not directly applicable. | skimmed | — |

## Backlog (notes files to write)

In rough priority order for our current plateau-debugging:

1. **`huang_notes.md`** — extreme-bandit mechanics and state-jumping algorithm details. Most actionable of the unread-in-depth papers: Huang's base MCTS is token-level (like ours) so his improvements can potentially bolt on without re-architecting.
2. **`boolformer_notes.md`** — dense note on the main Boolformer paper. We build on it but have never written it up beyond a CLAUDE.md bullet. Worth doing before making non-trivial architectural changes to the encoder/decoder.
3. Cross-cutting concept notes (future): `action_granularity.md`, `value_head_collapse.md`, `reward_shaping.md` — synthesize across papers + our own experiments.

## Conventions

- **One PDF per paper** in this folder, named `<firstauthor><year>_<shortname>.pdf`.
- **One notes file per paper (or per cluster)** as `<firstauthor>_notes.md` — Kamienny's thesis + ICML paper share one notes file because they overlap.
- **Transcribe tables and key formulas** into the notes file as markdown tables / LaTeX. Cite with section or page number.
- **Extract figures that matter** to `papers/figures/<paper>/figN.png` via `pdftoppm -png -f N -l N`, reference from the notes file.
- **Update this index** whenever a PDF is added/removed or a notes file is created. The index is the canonical navigational layer.
