"""
singularity.core.selector
==========================
Sparse KV-cache selection -- the open-source approximation of what the
closed-source "Predictive Oracle" does in the commercial tier.

Key research insight (see docs/ARCHITECTURE.md): a large fraction of next-token
attention mass concentrates on (a) the initial "attention sink" tokens and
(b) the most recent tokens. This is consistent with published work on
attention sinks (StreamingLLM) and heavy-hitter / H2O-style KV eviction.

This module ranks blocks so the fabric can *teleport the critical subset
first* and stream the rest in the background. The concrete percentages here
are configurable design parameters, not guaranteed accuracy figures.

SIGNAL AVAILABILITY AND ADAPTIVE WEIGHTING
-------------------------------------------
Earlier versions blended sink/recency/observed at fixed weights (0.45/0.45/
0.10) regardless of whether the "observed" signal actually existed. On a real
first-time request, attention_score is 0.0 for every block -- so that 10%
term was contributing nothing, and mid-context "needle" importance (a key
fact stated once, with no sink/recency correlation) was undetectable by
construction. Verified with scripts/oracle_quality_eval.py: cold-start
needle_recall = 0% at every head_fraction under the old fixed weighting.

This version weights each signal by whether it is actually informative for
this request, not by a constant:
  - observed (historical attention_score) gets weight only if at least one
    block has a nonzero score -- i.e. this is a warm/repeated request.
  - magnitude (KV tensor L2-norm outlier score) gets weight only if real
    tensor data was supplied (score_blocks_from_kv). This is grounded in
    published outlier-feature / activation-magnitude work showing certain
    architectures concentrate salience in a small number of high-magnitude
    key vectors -- it is a real, literature-supported heuristic, NOT a
    guarantee for every model. Treat it as a candidate cold-start signal to
    validate against real attention traces, not a proven fix.
  - sink/recency always contribute (cheap, always available, real signal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .blocks import BlockMeta, KVBlock, Tier


@dataclass
class SelectionConfig:
    """Parameters for sparse selection.

    head_fraction: target fraction of total blocks to place in the HOT tier.
    sink_tokens:   number of leading tokens treated as attention sinks.
    recency_tokens: size of the trailing recency window (in tokens).
    """

    head_fraction: float = 0.12
    sink_tokens: int = 32
    recency_tokens: int = 512


def magnitude_outlier_scores(blocks: Iterable[KVBlock]) -> Dict[str, float]:
    """Compute a per-block salience proxy from REAL key-vector data.

    For each block, take the mean L2 norm of its key vectors, then z-score
    that value across all blocks in this context. Blocks whose key vectors
    are magnitude outliers relative to the rest of the sequence score higher.

    This requires no prior request history -- it is available on a cold
    start, because it is computed from the actual KV tensors already present
    in this request, not from feedback about a previous one.

    Caveat (stated plainly, not buried): the correlation between key-vector
    magnitude and true attention salience is architecture-dependent and is
    NOT verified here against a real trained model -- there is no
    huggingface-class model available in this environment to check that
    against. This is a real, implementable, testable signal; whether it
    holds for any specific deployed model must be verified against that
    model's real attention traces before being trusted in production.
    """
    blocks = list(blocks)
    if not blocks:
        return {}
    norms = {}
    for b in blocks:
        norms[b.meta.block_id] = float(np.linalg.norm(b.key, axis=-1).mean())
    values = np.array(list(norms.values()))
    mean, std = values.mean(), values.std() or 1.0
    return {bid: (v - mean) / std for bid, v in norms.items()}


def score_blocks(
    metas: Iterable[BlockMeta],
    total_tokens: int,
    config: SelectionConfig,
    magnitude_scores: Optional[Dict[str, float]] = None,
) -> List[BlockMeta]:
    """Assign an importance score to each block in-place and return the list.

    Signals, each normalized into [0, 1] before blending:
      * sink:      1.0 if the block overlaps the leading sink window
      * recency:   linear ramp toward the end of the sequence
      * observed:  normalized historical attention_score, IF any block has
                   a nonzero value (i.e. this is not a true cold start)
      * magnitude: normalized KV tensor magnitude-outlier score, IF real
                   tensor data was supplied via magnitude_scores

    Weights are chosen based on which signals are actually available for
    THIS request, not fixed constants -- see module docstring.
    """
    metas = list(metas)
    if not metas:
        return metas

    has_observed = any(m.attention_score > 0.0 for m in metas)
    has_magnitude = bool(magnitude_scores)

    max_observed = max((m.attention_score for m in metas), default=0.0) or 1.0
    if has_magnitude:
        mag_vals = np.array([magnitude_scores.get(m.block_id, 0.0) for m in metas])
        mag_min, mag_max = mag_vals.min(), mag_vals.max()
        mag_range = (mag_max - mag_min) or 1.0
    recency_start = max(0, total_tokens - config.recency_tokens)

    # Adaptive weights: trust real observed feedback most when it exists;
    # otherwise fall back toward magnitude (if available) or pure position.
    if has_observed:
        w_sink, w_recency, w_observed, w_magnitude = 0.25, 0.25, 0.50, 0.0
    elif has_magnitude:
        w_sink, w_recency, w_observed, w_magnitude = 0.35, 0.35, 0.0, 0.30
    else:
        w_sink, w_recency, w_observed, w_magnitude = 0.45, 0.45, 0.10, 0.0

    for m in metas:
        sink = 1.0 if m.start_pos < config.sink_tokens else 0.0
        if m.end_pos <= recency_start:
            recency = 0.0
        else:
            overlap = m.end_pos - max(m.start_pos, recency_start)
            recency = min(1.0, overlap / max(1, config.recency_tokens))
        observed = (m.attention_score / max_observed) if has_observed else 0.0
        magnitude = 0.0
        if has_magnitude:
            raw = magnitude_scores.get(m.block_id, mag_min)
            magnitude = (raw - mag_min) / mag_range
        m.attention_score = (
            w_sink * sink + w_recency * recency
            + w_observed * observed + w_magnitude * magnitude
        )
    return metas


def assign_tiers(
    metas: Iterable[BlockMeta],
    total_tokens: int,
    config: SelectionConfig | None = None,
    magnitude_scores: Optional[Dict[str, float]] = None,
) -> List[BlockMeta]:
    """Score blocks and assign HOT/WARM/COLD tiers.

    The top `head_fraction` of blocks by score become HOT (teleported first),
    the next equal-sized band becomes WARM, the remainder COLD.
    """
    config = config or SelectionConfig()
    metas = score_blocks(metas, total_tokens, config, magnitude_scores=magnitude_scores)
    order = sorted(metas, key=lambda m: m.attention_score, reverse=True)

    n = len(order)
    hot_n = max(1, int(round(n * config.head_fraction)))
    warm_n = min(n - hot_n, hot_n * 2)

    for i, m in enumerate(order):
        if i < hot_n:
            m.tier = Tier.HOT
        elif i < hot_n + warm_n:
            m.tier = Tier.WARM
        else:
            m.tier = Tier.COLD
    return metas


def head_bytes(metas: Iterable[BlockMeta]) -> int:
    return sum(m.nbytes() for m in metas if m.tier == Tier.HOT)


def total_bytes(metas: Iterable[BlockMeta]) -> int:
    return sum(m.nbytes() for m in metas)
