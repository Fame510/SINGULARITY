"""
SINGULARITY Python SDK
======================
Open-source core of the Disaggregated KV-Cache Service (DKCS).

Two entry points:

  * `ContextFabric` -- the real, runnable fabric (page table + selector +
    transport). Works today on CPU/CI over a loopback transport; swap in the
    RDMA backend on real hardware without changing your code.

  * `DistributedLLM` -- a thin, vLLM-shaped facade kept for backward
    compatibility with the original SDK sketch. It orchestrates a ContextFabric
    under the hood. Text generation itself is delegated to a real engine
    (e.g. vLLM) in production; without one installed it clearly reports that
    rather than fabricating model output.

Nothing in this package fabricates performance numbers. Benchmarks measure real
serialization/transfer/reconstruction over the active transport and label the
backend used. See docs/ARCHITECTURE.md for target-vs-measured claims.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .core import BlockMeta, KVBlock, Tier, SelectionConfig
from .sdk import ContextFabric, TeleportReport
from .transport import InMemoryTransport, Transport

__version__ = "0.2.0"

__all__ = [
    "ContextFabric", "TeleportReport",
    "KVBlock", "BlockMeta", "Tier", "SelectionConfig",
    "InMemoryTransport", "Transport",
    "DistributedLLM", "FabricConfig", "OracleConfig",
    "__version__",
]


class FabricConfig:
    """Transport fabric configuration (backward-compatible)."""

    def __init__(self, mode: str = "roce", devices: Optional[List[str]] = None,
                 page_size_mb: int = 16) -> None:
        self.mode = mode
        self.devices = devices or ["mlx5_0"]
        self.page_size_mb = page_size_mb


class OracleConfig:
    """Predictive Oracle configuration (Enterprise tier).

    In open-core, oracle_enabled routes selection through the built-in sparse
    selector. The proprietary Oracle (learned attention-importance model) is a
    commercial add-on and is NOT bundled here.
    """

    def __init__(self, enabled: bool = False, head_percent: float = 0.12,
                 prefetch_aggression: float = 0.8) -> None:
        self.enabled = enabled
        self.head_percent = head_percent
        self.prefetch_aggression = prefetch_aggression


class DistributedLLM:
    """vLLM-shaped facade over a ContextFabric.

    Backward compatible with the original sketch's constructor. `generate`
    requires a real backend engine; if none is wired, it raises rather than
    returning a fake string.
    """

    def __init__(self, model: str, fabric_mode: str = "roce",
                 oracle_enabled: bool = False, tensor_parallel_size: int = 1,
                 engine=None, **kwargs) -> None:
        self.model = model
        self.fabric_config = FabricConfig(mode=fabric_mode)
        self.oracle = OracleConfig(enabled=oracle_enabled)
        self.tensor_parallel_size = tensor_parallel_size
        self._engine = engine  # inject a real generation engine here
        self.fabric = ContextFabric(
            selection=SelectionConfig(head_fraction=self.oracle.head_percent)
        )
        self._init_time = time.time()
        self._total_tokens = 0
        self._total_teleports = 0

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        if self._engine is None:
            raise NotImplementedError(
                "DistributedLLM.generate needs a real generation engine. Pass "
                "engine=<vLLM-like object with .generate(prompt, max_tokens)>. "
                "SINGULARITY handles KV-cache movement, not token sampling."
            )
        out = self._engine.generate(prompt, max_tokens=max_tokens, **kwargs)
        self._total_tokens += max_tokens
        return out

    @property
    def stats(self) -> dict:
        return {
            "model": self.model,
            "fabric_mode": self.fabric_config.mode,
            "oracle_enabled": self.oracle.enabled,
            "uptime_seconds": time.time() - self._init_time,
            "total_tokens": self._total_tokens,
            "total_teleports": self._total_teleports,
            "fabric_blocks": len(self.fabric.page_table),
        }
