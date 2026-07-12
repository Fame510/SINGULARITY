"""
singularity.transport.base
===========================
Pluggable transport backend interface.

A transport moves a KV block's bytes from a source endpoint to a destination
endpoint. SINGULARITY defines one narrow interface so the *same* scheduling and
selection logic works over any fabric:

  * InMemoryTransport  -- loopback, always available (this repo, CI, laptops)
  * RdmaTransport      -- GPU-Direct RDMA / RoCE v2 (stub here; requires
                          libibverbs + registered device memory on real HW)

The interface is intentionally byte-oriented and synchronous-with-futures so a
real zero-copy RDMA implementation can satisfy it without changing callers.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from ..core.blocks import KVBlock


@dataclass
class TransferResult:
    block_id: str
    nbytes: int
    seconds: float

    @property
    def gbps(self) -> float:
        if self.seconds <= 0:
            return float("inf")
        return (self.nbytes * 8) / self.seconds / 1e9


class Transport(abc.ABC):
    """Abstract transport backend."""

    name: str = "abstract"

    @abc.abstractmethod
    def put(self, block: KVBlock) -> TransferResult:
        """Publish a block to this transport's address space."""

    @abc.abstractmethod
    def get(self, block_id: str) -> tuple[KVBlock, TransferResult]:
        """Fetch a previously-published block by id."""

    @abc.abstractmethod
    def has(self, block_id: str) -> bool:
        ...

    def close(self) -> None:  # pragma: no cover - optional hook
        pass
