"""
singularity.sdk.fabric
======================
ContextFabric: high-level orchestration over the page table + transport.

This is the honest, working core of the "context teleportation" idea:

  1. A producer node splits a sequence's KV-cache into blocks and publishes
     them to the fabric (`publish_context`).
  2. The selector assigns HOT/WARM/COLD tiers (sparse selection).
  3. A consumer node teleports the context (`teleport_context`): it fetches HOT
     blocks first (critical path, "time-to-first-token"), then WARM, then COLD.

Everything here executes for real over the InMemoryTransport. Swapping in the
RDMA backend on real hardware requires no changes to this file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.blocks import KVBlock, Tier
from ..core.page_table import GlobalPageTable, FetchPlan
from ..core.selector import SelectionConfig, assign_tiers, head_bytes, total_bytes
from ..transport.base import Transport
from ..transport.memory import InMemoryTransport


@dataclass
class TeleportReport:
    """Measured results of a teleport. All timings are real wall-clock over the
    configured transport (loopback unless an RDMA backend is injected)."""

    num_blocks: int
    total_bytes: int
    head_bytes: int
    hot_seconds: float = 0.0
    warm_seconds: float = 0.0
    cold_seconds: float = 0.0
    blocks_fetched: List[str] = field(default_factory=list)

    @property
    def time_to_first_token_ms(self) -> float:
        """Proxy for TTFT: time to land the HOT (critical) tier."""
        return self.hot_seconds * 1000.0

    @property
    def total_ms(self) -> float:
        return (self.hot_seconds + self.warm_seconds + self.cold_seconds) * 1000.0

    @property
    def head_fraction(self) -> float:
        return self.head_bytes / self.total_bytes if self.total_bytes else 0.0


class ContextFabric:
    def __init__(
        self,
        transport: Optional[Transport] = None,
        selection: Optional[SelectionConfig] = None,
    ) -> None:
        self.transport = transport or InMemoryTransport()
        self.selection = selection or SelectionConfig()
        self.page_table = GlobalPageTable()
        self.page_table.register_transport(self.transport)

    def publish_context(self, blocks: List[KVBlock], total_tokens: int) -> None:
        assign_tiers((b.meta for b in blocks), total_tokens, self.selection)
        for b in blocks:
            self.page_table.publish(b, self.transport)

    def teleport_context(
        self, block_ids: List[str], hot_only: bool = False
    ) -> TeleportReport:
        plan: FetchPlan = self.page_table.plan_fetch(block_ids)
        report = TeleportReport(
            num_blocks=len(block_ids),
            total_bytes=0,
            head_bytes=0,
        )

        def _drain(ids: List[str]) -> float:
            elapsed = 0.0
            for bid in ids:
                block, res = self.page_table.fetch(bid)
                elapsed += res.seconds
                report.total_bytes += block.meta.nbytes()
                if block.meta.tier == Tier.HOT:
                    report.head_bytes += block.meta.nbytes()
                report.blocks_fetched.append(bid)
            return elapsed

        report.hot_seconds = _drain(plan.hot)
        if not hot_only:
            report.warm_seconds = _drain(plan.warm)
            report.cold_seconds = _drain(plan.cold)
        return report
