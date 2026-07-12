"""
singularity.transport.memory
=============================
InMemoryTransport: a real, working loopback transport.

It serializes each block to a contiguous byte buffer (exactly what a real
fabric would DMA) and reconstructs it on `get`. This exercises the true cost of
serialization + copy + reconstruction, so benchmarks report *measured* numbers
on this backend -- clearly labeled as loopback, not RDMA-over-the-wire.
"""
from __future__ import annotations

import io
import time
from typing import Dict

import numpy as np

from ..core.blocks import BlockMeta, KVBlock, Tier
from .base import Transport, TransferResult


def _serialize(block: KVBlock) -> bytes:
    buf = io.BytesIO()
    np.save(buf, block.key, allow_pickle=False)
    np.save(buf, block.value, allow_pickle=False)
    return buf.getvalue()


def _deserialize(meta: BlockMeta, raw: bytes) -> KVBlock:
    buf = io.BytesIO(raw)
    key = np.load(buf, allow_pickle=False)
    value = np.load(buf, allow_pickle=False)
    return KVBlock(meta=meta, key=key, value=value)


class InMemoryTransport(Transport):
    name = "in-memory-loopback"

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self._meta: Dict[str, BlockMeta] = {}

    def put(self, block: KVBlock) -> TransferResult:
        t0 = time.perf_counter()
        raw = _serialize(block)
        self._store[block.meta.block_id] = raw
        self._meta[block.meta.block_id] = block.meta
        dt = time.perf_counter() - t0
        return TransferResult(block.meta.block_id, len(raw), dt)

    def get(self, block_id: str) -> tuple[KVBlock, TransferResult]:
        if block_id not in self._store:
            raise KeyError(f"unknown block_id: {block_id}")
        t0 = time.perf_counter()
        raw = self._store[block_id]
        block = _deserialize(self._meta[block_id], raw)
        dt = time.perf_counter() - t0
        return block, TransferResult(block_id, len(raw), dt)

    def has(self, block_id: str) -> bool:
        return block_id in self._store
