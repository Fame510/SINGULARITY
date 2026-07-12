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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .blocks import BlockMeta, Tier


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


def score_blocks(
    metas: Iterable[BlockMeta],
    total_tokens: int,
    config: SelectionConfig,
) -> List[BlockMeta]:
    """Assign an importance score to each block in-place and return the list.

    Score combines three signals, each in [0, 1]:
      * sink:    1.0 if the block overlaps the leading sink window
      * recency: linear ramp toward the end of the sequence
      * observed: normalized historical attention_score (if provided)
    """
    metas = list(metas)
    if not metas:
        return metas

    max_observed = max((m.attention_score for m in metas), default=0.0) or 1.0
    recency_start = max(0, total_tokens - config.recency_tokens)

    for m in metas:
        sink = 1.0 if m.start_pos < config.sink_tokens else 0.0
        if m.end_pos <= recency_start:
            recency = 0.0
        else:
            overlap = m.end_pos - max(m.start_pos, recency_start)
            recency = min(1.0, overlap / max(1, config.recency_tokens))
        observed = m.attention_score / max_observed
        # Weighted blend; sinks and recency dominate, observed refines ties.
        m.attention_score = 0.45 * sink + 0.45 * recency + 0.10 * observed
    return metas


def assign_tiers(
    metas: Iterable[BlockMeta],
    total_tokens: int,
    config: SelectionConfig | None = None,
) -> List[BlockMeta]:
    """Score blocks and assign HOT/WARM/COLD tiers.

    The top `head_fraction` of blocks by score become HOT (teleported first),
    the next equal-sized band becomes WARM, the remainder COLD.
    """
    config = config or SelectionConfig()
    metas = score_blocks(metas, total_tokens, config)
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
