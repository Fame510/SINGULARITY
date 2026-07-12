"""
singularity.transport.rdma
==========================
RDMA / RoCE v2 transport -- HARDWARE-DEPENDENT STUB.

This backend is intentionally NOT implemented in the open-source Python package
because it requires:
  * libibverbs / rdma-core and a RoCE v2 or InfiniBand NIC
  * GPU-Direct RDMA (nvidia-peermem) with device memory registered as MRs
  * a running SINGULARITY orchestrator to exchange queue-pair metadata

The production implementation lives in the Rust crate `singularity-transport`
(see repo root). This class documents the contract and fails loudly rather than
silently pretending to move data over a network.
"""
from __future__ import annotations

from ..core.blocks import KVBlock
from .base import Transport, TransferResult


class RdmaTransport(Transport):
    name = "rdma-roce-v2"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "RdmaTransport requires libibverbs + GPU-Direct RDMA hardware and "
            "the native singularity-transport crate. Use InMemoryTransport for "
            "development/CI. See docs/ARCHITECTURE.md section 1."
        )

    def put(self, block: KVBlock) -> TransferResult:  # pragma: no cover
        raise NotImplementedError

    def get(self, block_id: str):  # pragma: no cover
        raise NotImplementedError

    def has(self, block_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError
