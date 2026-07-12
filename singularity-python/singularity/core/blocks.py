"""
singularity.core.blocks
=======================
Core data structures for the disaggregated KV-cache.

A *block* (context-page) is the atomic unit SINGULARITY schedules and moves
across the fabric -- not a whole pod. Each block holds the key/value tensors
for a contiguous span of token positions.

Pure-Python + NumPy so it runs anywhere (CPU laptop, CI, or GPU node). On real
hardware the payload would be device memory registered for GPU-Direct RDMA;
here it is a NumPy array so the tiering and selection logic are exercised for
real.
"""
from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class Tier(enum.IntEnum):
    """KV block storage tier, ordered hot -> cold."""

    HOT = 0   # recent tokens + attention sinks -- critical path
    WARM = 1  # mid-range context
    COLD = 2  # full history / background stream


@dataclass
class BlockMeta:
    """Metadata describing a KV block without its payload."""

    block_id: str
    layer: int
    start_pos: int
    end_pos: int
    num_heads: int
    head_dim: int
    dtype: str
    tier: Tier = Tier.COLD
    attention_score: float = 0.0
    last_access: float = field(default_factory=time.time)

    @property
    def num_tokens(self) -> int:
        return self.end_pos - self.start_pos

    def nbytes(self) -> int:
        itemsize = np.dtype(self.dtype).itemsize
        return 2 * self.num_tokens * self.num_heads * self.head_dim * itemsize


@dataclass
class KVBlock:
    """A KV block: metadata + key/value payload of shape
    (num_tokens, num_heads, head_dim)."""

    meta: BlockMeta
    key: np.ndarray
    value: np.ndarray

    def __post_init__(self) -> None:
        expected = (self.meta.num_tokens, self.meta.num_heads, self.meta.head_dim)
        if self.key.shape != expected or self.value.shape != expected:
            raise ValueError(
                f"KV shape mismatch: expected {expected}, "
                f"got key={self.key.shape} value={self.value.shape}"
            )

    def content_hash(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.key.tobytes())
        h.update(self.value.tobytes())
        return h.hexdigest()

    @staticmethod
    def random(
        block_id: str,
        layer: int,
        start_pos: int,
        num_tokens: int,
        num_heads: int = 8,
        head_dim: int = 128,
        dtype: str = "float16",
        seed: Optional[int] = None,
    ) -> "KVBlock":
        rng = np.random.default_rng(seed)
        shape = (num_tokens, num_heads, head_dim)
        key = rng.standard_normal(shape).astype(dtype)
        value = rng.standard_normal(shape).astype(dtype)
        meta = BlockMeta(
            block_id=block_id, layer=layer, start_pos=start_pos,
            end_pos=start_pos + num_tokens, num_heads=num_heads,
            head_dim=head_dim, dtype=dtype,
        )
        return KVBlock(meta=meta, key=key, value=value)
