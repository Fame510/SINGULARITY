#!/usr/bin/env python3
"""
SINGULARITY Oracle Quality Evaluator
=====================================
The speed benchmark (slb_benchmark.py) measures how much DATA is moved.
It says nothing about whether the RIGHT data was moved. This script does.

Method
------
We generate a synthetic "true importance" distribution over blocks that
mixes three known real-world attention patterns:

  1. sink mass      -- leading tokens (real, documented: StreamingLLM)
  2. recency mass   -- trailing tokens (real, documented: H2O/heavy-hitter)
  3. "needle" mass   -- a small number of RANDOMLY PLACED mid-context blocks
                        carrying real importance that pure position heuristics
                        cannot see (e.g. a key fact stated once in the middle
                        of a long document -- the classic "needle in a
                        haystack" failure mode for position-only eviction).

We then run the UNMODIFIED production code (score_blocks / assign_tiers from
singularity.core.selector) using ONLY the signals it has access to in a real
cold-start request (attention_score defaults to 0.0 -- no prior observation),
and measure how much of the TRUE importance mass its HOT-tier selection
actually captures.

This is the honest question "26x faster" dodges: fast at moving how much of
the wrong data?

Metrics reported, at each head_fraction:
  - mass_recall:  fraction of TOTAL true importance captured by the
                  predicted HOT tier (the metric that matters -- if this is
                  low, the model loses real information no matter how fast
                  the transfer was)
  - needle_recall: of the deliberately-buried needle blocks specifically,
                  what fraction did HOT-tier selection catch? This isolates
                  the exact failure mode position-only heuristics have.
  - precision:    of the blocks selected HOT, what fraction were actually
                  in the true top-K by importance?

We also run a second condition where attention_score IS populated (warm
cache / prior-observation case) to show what the 10% "observed" term in the
production formula can recover, versus the cold-start case it cannot.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "singularity-python"))

import numpy as np

from singularity.core.blocks import BlockMeta, KVBlock, Tier  # noqa: E402
from singularity.core.selector import (  # noqa: E402
    SelectionConfig, assign_tiers, magnitude_outlier_scores,
)


def build_synthetic_context(
    total_tokens: int,
    block_tokens: int,
    sink_tokens: int,
    recency_tokens: int,
    num_needles: int,
    needle_mass_each: float,
    seed: int,
    needle_key_magnitude_mult: float = 1.0,
    num_heads: int = 8,
    head_dim: int = 128,
):
    """Build blocks with a KNOWN ground-truth importance distribution, AND
    real KV tensors -- needed to test the magnitude-outlier cold-start signal
    honestly, rather than against synthetic labels the selector never sees.

    Ground truth mixes:
      - sink blocks:    high true importance (position 0..sink_tokens)
      - recency blocks: high true importance (last recency_tokens)
      - needle blocks:  a few RANDOM mid-context blocks given deliberately
                        high true importance, simulating buried key content
                        that position heuristics structurally cannot detect
      - everything else: low background importance

    needle_key_magnitude_mult: if > 1.0, needle blocks' KEY VECTORS are
      scaled up by this factor, simulating the documented outlier-feature
      phenomenon (real salient tokens sometimes carry anomalously large key
      magnitudes). Set to 1.0 to test the NULL case -- needles with NO
      magnitude tell -- which is the harder, more honest test of whether
      the magnitude signal is doing real work or just getting lucky.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    metas: List[BlockMeta] = []
    keys: List[np.ndarray] = []
    pos = 0
    idx = 0
    while pos < total_tokens:
        n = min(block_tokens, total_tokens - pos)
        metas.append(BlockMeta(
            block_id=f"B{idx}", layer=0, start_pos=pos, end_pos=pos + n,
            num_heads=num_heads, head_dim=head_dim, dtype="float16",
        ))
        keys.append(np_rng.normal(0, 1, size=(n, num_heads, head_dim)).astype(np.float32))
        pos += n
        idx += 1

    n_blocks = len(metas)
    recency_start = max(0, total_tokens - recency_tokens)

    true_importance = [0.02] * n_blocks  # small background mass everywhere
    for i, m in enumerate(metas):
        if m.start_pos < sink_tokens:
            true_importance[i] += 1.0
        if m.end_pos > recency_start:
            true_importance[i] += 1.0

    # Bury needles in the MIDDLE band only (never in sink or recency
    # windows), so this genuinely tests detection of hidden importance,
    # not a relabeling of what position heuristics already catch.
    mid_candidates = [
        i for i, m in enumerate(metas)
        if m.start_pos >= sink_tokens and m.end_pos <= recency_start
    ]
    needle_idxs = rng.sample(mid_candidates, k=min(num_needles, len(mid_candidates)))
    for i in needle_idxs:
        true_importance[i] += needle_mass_each
        if needle_key_magnitude_mult != 1.0:
            keys[i] = keys[i] * needle_key_magnitude_mult

    total_mass = sum(true_importance)
    true_importance = [t / total_mass for t in true_importance]  # normalize to sum=1

    kv_blocks = [
        KVBlock(meta=metas[i], key=keys[i], value=keys[i].copy())
        for i in range(n_blocks)
    ]
    return metas, true_importance, set(needle_idxs), kv_blocks


def evaluate(metas, true_importance, needle_idxs, kv_blocks, config: SelectionConfig,
             total_tokens: int, mode: str):
    """Run the REAL production selector and score its HOT-tier choice
    against the ground truth we built. mode in {"cold", "warm", "cold_magnitude"}:
      - cold:           true first-request path, no signal beyond position
      - warm:           prior noisy attention_score available (repeat request)
      - cold_magnitude: first request, but NOW using real KV tensor data via
                        magnitude_outlier_scores -- the new signal, still a
                        genuine cold start (no prior request needed)
    """
    for m in metas:
        m.tier = Tier.COLD
        m.attention_score = 0.0

    magnitude_scores = None
    if mode == "warm":
        rng = random.Random(99)
        for m, t in zip(metas, true_importance):
            noise = rng.gauss(0, 0.15 * max(true_importance))
            m.attention_score = max(0.0, t + noise)
    elif mode == "cold_magnitude":
        magnitude_scores = magnitude_outlier_scores(kv_blocks)

    assign_tiers(metas, total_tokens=total_tokens, config=config,
                 magnitude_scores=magnitude_scores)

    hot_idxs = {i for i, m in enumerate(metas) if m.tier == Tier.HOT}
    n_hot = len(hot_idxs)

    # True top-K by ground truth, K = n_hot (apples-to-apples set size)
    ranked_truth = sorted(range(len(metas)), key=lambda i: true_importance[i], reverse=True)
    true_top_k = set(ranked_truth[:n_hot])

    mass_captured = sum(true_importance[i] for i in hot_idxs)
    total_mass = sum(true_importance)
    mass_recall = mass_captured / total_mass if total_mass else 0.0

    precision = len(hot_idxs & true_top_k) / n_hot if n_hot else 0.0

    needle_hit = len(hot_idxs & needle_idxs)
    needle_recall = needle_hit / len(needle_idxs) if needle_idxs else None

    return {
        "n_hot_selected": n_hot,
        "mass_recall": round(mass_recall, 4),
        "precision_vs_true_top_k": round(precision, 4),
        "needle_recall": round(needle_recall, 4) if needle_recall is not None else None,
        "needles_planted": len(needle_idxs),
        "needles_caught": needle_hit,
    }


def main():
    p = argparse.ArgumentParser(description="SINGULARITY Oracle selection-quality evaluator")
    p.add_argument("--context-tokens", type=int, default=32768)
    p.add_argument("--block-tokens", type=int, default=256)
    p.add_argument("--sink-tokens", type=int, default=32)
    p.add_argument("--recency-tokens", type=int, default=512)
    p.add_argument("--num-needles", type=int, default=5)
    p.add_argument("--needle-mass-each", type=float, default=0.8)
    p.add_argument("--needle-key-magnitude-mult", type=float, default=3.0,
                    help="1.0 = null case (no magnitude tell). >1.0 simulates "
                         "the outlier-feature phenomenon on needle blocks.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--head-fractions", type=float, nargs="+",
                    default=[0.12, 0.08, 0.0385, 0.02])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    metas, true_importance, needle_idxs, kv_blocks = build_synthetic_context(
        args.context_tokens, args.block_tokens, args.sink_tokens,
        args.recency_tokens, args.num_needles, args.needle_mass_each, args.seed,
        needle_key_magnitude_mult=args.needle_key_magnitude_mult,
    )

    results = []
    for hf in args.head_fractions:
        config = SelectionConfig(head_fraction=hf, sink_tokens=args.sink_tokens,
                                  recency_tokens=args.recency_tokens)
        cold = evaluate(metas, true_importance, needle_idxs, kv_blocks, config,
                         args.context_tokens, mode="cold")
        warm = evaluate(metas, true_importance, needle_idxs, kv_blocks, config,
                         args.context_tokens, mode="warm")
        cold_mag = evaluate(metas, true_importance, needle_idxs, kv_blocks, config,
                             args.context_tokens, mode="cold_magnitude")
        results.append({
            "head_fraction": hf, "cold_start": cold,
            "cold_start_with_magnitude_signal": cold_mag, "warm_cache": warm,
        })

    out = {
        "total_blocks": len(metas),
        "needles_planted_mid_context": len(needle_idxs),
        "note": (
            "cold_start = real production code path for a first-time request "
            "(attention_score=0.0, position signals only). warm_cache = "
            "same code path with noisy prior-observation attention_score, "
            "simulating a repeated/cached request."
        ),
        "results": results,
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print("=" * 78)
    print(" SINGULARITY Oracle Selection-Quality Evaluation")
    print(" %d blocks total | %d needles buried in mid-context (never in sink/recency)"
          % (len(metas), len(needle_idxs)))
    print("=" * 78)
    for r in results:
        hf = r["head_fraction"]
        c, m, w = r["cold_start"], r["cold_start_with_magnitude_signal"], r["warm_cache"]
        print(f"\n head_fraction = {hf}  (n_hot = {c['n_hot_selected']})")
        print(f"   {'':22}{'mass_recall':>13}{'precision':>13}{'needle_recall':>15}")
        print(f"   {'cold (old)':22}{c['mass_recall']:>13.1%}{c['precision_vs_true_top_k']:>13.1%}"
              f"{(c['needle_recall'] or 0):>15.1%}")
        print(f"   {'cold + magnitude':22}{m['mass_recall']:>13.1%}{m['precision_vs_true_top_k']:>13.1%}"
              f"{(m['needle_recall'] or 0):>15.1%}")
        print(f"   {'warm-cache':22}{w['mass_recall']:>13.1%}{w['precision_vs_true_top_k']:>13.1%}"
              f"{(w['needle_recall'] or 0):>15.1%}")
    print("\n" + "-" * 78)
    print(" cold (old)        = position-only, no prior request, no tensor signal")
    print(" cold + magnitude  = same first request, NEW: real KV tensor magnitude signal")
    print(" warm-cache        = repeat request with a noisy prior attention observation")


if __name__ == "__main__":
    main()
