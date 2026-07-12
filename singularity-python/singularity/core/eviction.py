"""
singularity.core.eviction
==========================
Capacity management for the page table.

Real fabrics have finite memory. This module implements a tier-aware LRU
eviction policy: when the cache exceeds its byte budget, evict COLD blocks
first (least recently used within tier), then WARM, and protect HOT blocks as
long as possible. This mirrors how production KV caches protect the critical
working set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from .blocks import BlockMeta, Tier


@dataclass
class LRUEvictionPolicy:
    capacity_bytes: int
    _used: int = 0
    _last_access: Dict[str, float] = field(default_factory=dict)
    _size: Dict[str, int] = field(default_factory=dict)
    _tier: Dict[str, Tier] = field(default_factory=dict)

    def admit(self, meta: BlockMeta) -> List[str]:
        """Record a block; return ids evicted to stay within budget."""
        self._last_access[meta.block_id] = time.time()
        self._size[meta.block_id] = meta.nbytes()
        self._tier[meta.block_id] = meta.tier
        self._used += meta.nbytes()
        return self._enforce()

    def touch(self, block_id: str) -> None:
        if block_id in self._last_access:
            self._last_access[block_id] = time.time()

    def _enforce(self) -> List[str]:
        evicted: List[str] = []
        if self._used <= self.capacity_bytes:
            return evicted
        # Evict COLD, then WARM, then HOT; LRU within each tier.
        for tier in (Tier.COLD, Tier.WARM, Tier.HOT):
            candidates = sorted(
                (b for b, t in self._tier.items() if t == tier),
                key=lambda b: self._last_access.get(b, 0.0),
            )
            for b in candidates:
                if self._used <= self.capacity_bytes:
                    return evicted
                self._used -= self._size.pop(b, 0)
                self._last_access.pop(b, None)
                self._tier.pop(b, None)
                evicted.append(b)
        return evicted

    @property
    def used_bytes(self) -> int:
        return self._used
