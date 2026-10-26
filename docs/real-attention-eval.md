# Real-Attention Validation of the Magnitude Signal

**Date:** 2026-10-26 · **Script:** `scripts/real_attention_eval.py` ·
**Environment:** torch 2.14.1+cpu, transformers 5.21.0, model
`openai-community/gpt2` (124M, 12 layers × 12 heads), eager attention,
CPU. Blocks of 16 tokens; teleport budget 12%.

## Question

Does the selector's magnitude signal — the mean L2 norm of a block's
key vectors (`selector.magnitude_outlier_scores`) — predict which KV
blocks a **real** model actually attends to, better than position
alone? The synthetic evaluator (`scripts/oracle_quality_eval.py`)
said yes, decisively — on synthetic ground truth whose needles were
constructed with amplified key magnitudes. This experiment removes
the construction: ground truth is the attention a real model really
paid.

## Method

Three prose passages (~300–350 tokens each: expository, technical,
narrative). For each: run GPT-2 with attention capture; ground truth
per block = attention mass **received** (summed over all layers,
heads, and query positions). Signals per block: magnitude (mean key
L2 norm from the model's own KV cache) and position (sink + recency,
the pre-existing cold-start signal). Metrics: Spearman rank
correlation against true block mass, and **mass recall at the 12%
budget** — the share of total attention mass captured by the blocks
each signal would teleport.

## Results

| Passage | Tokens | Blocks | ρ magnitude | ρ position | Recall@12% magnitude | Recall@12% position | Recall@12% oracle-top |
|---|---|---|---|---|---|---|---|
| 0 | 346 | 21 | −0.688 | −0.658 | 0.070 | 0.467 | 0.524 |
| 1 | 298 | 18 | −0.705 | −0.534 | 0.042 | 0.473 | 0.505 |
| 2 | 328 | 20 | −0.782 | −0.674 | 0.043 | 0.453 | 0.482 |
| **Mean** | | | **−0.725** | −0.622 | **0.052** | **0.464** | 0.504 |

Random selection at this budget recalls 0.12 by construction.

## Verdict

**The magnitude signal fails on real GPT-2 traces — it is worse than
random.** Mean mass recall 0.052 against 0.12 for chance and 0.464
for plain position. The correlation is not merely absent but
strongly *negative* (ρ ≈ −0.73): on this model, high key-norm blocks
are systematically the ones attention ignores. (Position's negative
ρ with high recall is a different artifact: received mass is
concentrated in the earliest blocks, so a recency tail hurts
whole-list rank correlation while the top of the position ranking
still captures the sink mass. Budget recall is the operational
metric for teleportation, and there position nearly matches the
oracle-top bound.)

The strong inverse correlation is recorded as an observation, not
productized: one small model does not justify shipping an inverted
signal either.

## Scope — what this does and does not establish

- Established: for GPT-2 on short prose, key-magnitude does not mark
  attention salience; the synthetic result was an artifact of how
  the synthetic needles were built.
- Not established: anything about modern architectures (RoPE, GQA,
  models trained for long context), longer contexts, or other
  domains. The signal's standing rule applies: verify per
  architecture against that model's traces before trusting it.

## Consequence in this codebase

The signal is now **opt-in**: `ContextFabric` defaults
`enable_magnitude_signal=False`, and the selector docstring carries
this result. The synthetic evaluator remains as a unit-level
instrument; its headline is superseded by this document wherever the
two disagree. This is the verification the code always said was
required before production trust — it was run, and it said no.
