"""
singularity.core.page_table
============================
GlobalPageTable: the fabric's routing table.

Maps block_id -> location (which transport/endpoint currently holds the bytes)
plus the block metadata. This is the structure the vLLM KVConnector consults on
`allocate_block`: local hit returns immediately, remote hit triggers a fabric
fetch. Content-hash dedup lets identical prefixes share one physical copy.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .blocks import BlockMeta, KVBlock, Tier
from ..transport.base import Transport, TransferResult


@dataclass
class Entry:
    meta: BlockMeta
    transport: str
    content_hash: Optional[str] = None


@dataclass
class FetchPlan:
    """Ordered fetch plan: HOT first (critical path), then WARM, then COLD."""

    hot: List[str] = field(default_factory=list)
    warm: List[str] = field(default_factory=list)
    cold: List[str] = field(default_factory=list)

    def ordered(self) -> List[str]:
        return self.hot + self.warm + self.cold


class GlobalPageTable:
    def __init__(self) -> None:
        self._entries: Dict[str, Entry] = {}
        self._transports: Dict[str, Transport] = {}
        self._hash_index: Dict[str, str] = {}  # content_hash -> block_id
        self._lock = threading.RLock()

    def register_transport(self, transport: Transport) -> None:
        with self._lock:
            self._transports[transport.name] = transport

    def publish(self, block: KVBlock, transport: Transport) -> TransferResult:
        """Put a block on a transport and index it. Dedups identical content."""
        with self._lock:
            chash = block.content_hash()
            if chash in self._hash_index:
                existing_id = self._hash_index[chash]
                self._entries[block.meta.block_id] = self._entries[existing_id]
                return TransferResult(block.meta.block_id, 0, 0.0)
            result = transport.put(block)
            self._entries[block.meta.block_id] = Entry(
                meta=block.meta, transport=transport.name, content_hash=chash
            )
            self._hash_index[chash] = block.meta.block_id
            return result

    def lookup(self, block_id: str) -> Optional[Entry]:
        with self._lock:
            return self._entries.get(block_id)

    def fetch(self, block_id: str) -> tuple[KVBlock, TransferResult]:
        entry = self.lookup(block_id)
        if entry is None:
            raise KeyError(f"no page-table entry for {block_id}")
        transport = self._transports[entry.transport]
        return transport.get(block_id)

    def plan_fetch(self, block_ids: List[str]) -> FetchPlan:
        plan = FetchPlan()
        for bid in block_ids:
            entry = self.lookup(bid)
            tier = entry.meta.tier if entry else Tier.COLD
            if tier == Tier.HOT:
                plan.hot.append(bid)
            elif tier == Tier.WARM:
                plan.warm.append(bid)
            else:
                plan.cold.append(bid)
        return plan

    def __len__(self) -> int:
        return len(self._entries)
