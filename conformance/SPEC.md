# SINGULARITY Selection-Quality Conformance Battery — SPEC v0.1

**Status:** draft standard, open for independent runs and comment.
**Date:** 2026-10-26 · **Reference runner:** `conformance/run_battery.py`
· **Registry:** `conformance/REGISTRY.md`

## 1. Purpose

Throughput benchmarks for disaggregated KV-cache systems measure how
much data moved and how fast. They do not measure whether the *right*
data moved. A sparse-teleportation system that moves 12% of a context
is only as good as the selection policy that chose the 12%: if the
selection misses the blocks the model actually needs, the speed is
speed at losing information.

This battery measures **selection quality against known ground
truth**, so that any implementation's selection claims can be checked
by any other party, on pinned workloads, with a shared vocabulary.

It is a measurement instrument, not a certification of production
worthiness. Scope discipline is part of the specification (§7).

## 2. Workload (normative)

A workload case is fully determined by these parameters and a seed:

| Parameter | Value |
|---|---|
| context_tokens | 32768 |
| block_tokens | 256 (128 blocks) |
| sink_tokens | 32 |
| recency_tokens | 512 |
| num_needles | 20 |
| needle_mass_each | 0.8 |
| seeds | 1..10 |
| head_fractions (budgets) | 0.05, 0.12 |
| variants | `tell` (needle key-magnitude ×3.0), `null` (×1.0) |

Ground truth mixes three documented attention patterns: sink mass on
leading blocks, recency mass on trailing blocks, and needle mass on
20 blocks placed uniformly in the mid-context band (never inside the
sink or recency windows), over a 0.02 background. Mass is normalized
to sum 1. The `tell` variant additionally scales needle key vectors
by 3.0 (the published outlier-feature phenomenon); the `null`
variant gives needles no key-space tell.

The generator is normative by reference: `build_synthetic_context` in
`scripts/oracle_quality_eval.py` of this repository. Seeded RNG
(`random.Random(seed)`, `numpy.random.default_rng(seed)`) makes any
case bit-reproducible by any runner using that code at any commit
that does not change the generator; a runner MUST record the commit
it used.

## 3. Policy adapter contract

A policy under test consumes, per case: the block metadata list
(position, size), and whichever of these signals it declares —
`observed_attention` (a noisy prior observation: true mass + Gaussian
noise, σ = 0.15 × peak mass, `random.Random(99)` — identical for
every policy in a case) and/or the real KV tensors. It produces a
**hot set**: the blocks it would teleport within the budget.

A policy with no usable signal for a case (e.g., a history-only
policy on a cold start) MUST record the case with status
`no_history_signal` and null metrics. An untestable case is a
finding, not a failure, and never a zero.

## 4. Metrics (normative)

For a case with hot set H, |H| = K:

- **mass_recall** = (Σ true mass over H) / (Σ true mass over all blocks)
- **precision_vs_true_top_k** = |H ∩ T| / K, where T is the true
  top-K blocks by ground-truth mass
- **needle_recall** = |H ∩ needles| / |needles|

Aggregates are means over the 10 seeds, reported per
(variant, head_fraction). No aggregate may mix variants or budgets.

## 5. Result records

One JSON entry per (policy, mode), schema as written by the reference
runner: battery id + version, **track**, recording date, runner
(name, kind — `first-party` or `independent` — and a scope
statement), implementation (name, repo, commit), policy (id, mode,
description, declared signals), environment (OS, Python, numpy),
the pinned workload, per-seed case rows, and aggregates.

Entries are append-only. A corrected run is a new entry, never an
edit of an old one.

## 6. Tracks

- **Synthetic track** (this document): ground truth by construction.
  It can falsify a selection policy's *mechanism* — and it caught
  this repository's own magnitude signal being construction-dependent
  (see the real-track record) — but it cannot validate a policy
  against any real model.
- **Real track**: ground truth is a real model's measured attention
  (`docs/real-attention-eval.md` is the first real-track record in
  this repository). Real-track claims require the model, the corpus,
  and the capture method to be named.

An entry MUST name its track. Synthetic results MUST NOT be cited as
real-model evidence.

## 7. Scope discipline

A registry entry records that a named runner ran a named
implementation on the pinned workload in a named environment and got
these numbers. It does not record production enforcement, real-model
validity (synthetic track), or endorsement by the battery's author.
Claims beyond the entry's scope statement are the claimant's, not
the battery's.
