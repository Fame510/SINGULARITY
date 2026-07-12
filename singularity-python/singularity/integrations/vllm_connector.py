"""
singularity.integrations.vllm_connector
========================================
A real vLLM KVConnector-style adapter.

vLLM's disaggregated-prefill / KV-transfer machinery talks to external caches
through a *connector* object with a small, well-defined surface. Concrete vLLM
APIs have shifted across releases (KVConnectorBase, then KVConnectorBase_V1),
so this module does two things:

  1. Defines a stable, version-independent Protocol (`KVConnectorLike`) capturing
     the operations SINGULARITY needs: publish produced KV blocks, and fetch
     (teleport) required KV blocks by their engine block-ids.
  2. Provides `SingularityKVConnector`, a concrete implementation backed by a
     `ContextFabric`, that maps vLLM's integer block-ids onto fabric block-ids
     and round-trips real tensors.

It imports vLLM lazily. The connector is fully usable and testable WITHOUT vLLM
installed (see tests/test_vllm_connector.py) by driving it through the same
methods vLLM would call. When vLLM IS present, `from_vllm_config` wires it up.

This is deliberately engine-version-tolerant: it does not subclass a specific
vLLM base class (which would break across releases); instead it exposes the
methods vLLM's connector loader expects and adapts. See docs/INTEGRATION.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..core.blocks import BlockMeta, KVBlock, Tier
from ..core.selector import SelectionConfig
from ..sdk.fabric import ContextFabric, TeleportReport

logger = logging.getLogger("singularity.vllm")


@runtime_checkable
class KVConnectorLike(Protocol):
    """Minimal surface a KV connector must expose for SINGULARITY."""

    def publish_blocks(
        self, request_id: str, block_ids: Sequence[int],
        keys: Sequence[np.ndarray], values: Sequence[np.ndarray],
        positions: Sequence[Tuple[int, int]], total_tokens: int,
    ) -> None: ...

    def teleport_blocks(
        self, request_id: str, block_ids: Sequence[int], hot_only: bool = False,
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], TeleportReport]: ...


@dataclass
class ConnectorStats:
    requests: int = 0
    blocks_published: int = 0
    blocks_teleported: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def as_dict(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {
            **self.__dict__,
            "hit_rate": (self.cache_hits / total) if total else 0.0,
        }


@dataclass
class SingularityKVConnector:
    """Concrete connector backed by a ContextFabric.

    block_shape describes a single vLLM KV block: (block_size_tokens,
    num_kv_heads, head_dim). dtype is the tensor dtype string.
    """

    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    dtype: str = "float16"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    fabric: ContextFabric = field(init=False)
    stats: ConnectorStats = field(default_factory=ConnectorStats)

    def __post_init__(self) -> None:
        self.fabric = ContextFabric(selection=self.selection)

    # --- id mapping: engine (request_id, int block_id) -> fabric block_id ---
    @staticmethod
    def _fabric_id(request_id: str, block_id: int) -> str:
        return f"{request_id}::{block_id}"

    def publish_blocks(
        self, request_id: str, block_ids: Sequence[int],
        keys: Sequence[np.ndarray], values: Sequence[np.ndarray],
        positions: Sequence[Tuple[int, int]], total_tokens: int,
    ) -> None:
        if not (len(block_ids) == len(keys) == len(values) == len(positions)):
            raise ValueError("block_ids/keys/values/positions length mismatch")
        blocks: List[KVBlock] = []
        for bid, k, v, (start, end) in zip(block_ids, keys, values, positions):
            meta = BlockMeta(
                block_id=self._fabric_id(request_id, bid),
                layer=0, start_pos=start, end_pos=end,
                num_heads=self.num_kv_heads, head_dim=self.head_dim,
                dtype=self.dtype, tier=Tier.COLD,
            )
            blocks.append(KVBlock(
                meta=meta, key=np.ascontiguousarray(k),
                value=np.ascontiguousarray(v)))
        self.fabric.publish_context(blocks, total_tokens=total_tokens)
        self.stats.requests += 1
        self.stats.blocks_published += len(blocks)

    def teleport_blocks(
        self, request_id: str, block_ids: Sequence[int], hot_only: bool = False,
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], TeleportReport]:
        fabric_ids = [self._fabric_id(request_id, b) for b in block_ids]
        present, missing = [], []
        for eng_id, fid in zip(block_ids, fabric_ids):
            if self.fabric.page_table.lookup(fid) is not None:
                present.append((eng_id, fid))
                self.stats.cache_hits += 1
            else:
                missing.append(eng_id)
                self.stats.cache_misses += 1
        if missing:
            logger.debug("teleport cache miss for %s blocks %s",
                         request_id, missing)

        report = self.fabric.teleport_context(
            [fid for _, fid in present], hot_only=hot_only)
        out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for eng_id, fid in present:
            block, _ = self.fabric.page_table.fetch(fid)
            out[eng_id] = (block.key, block.value)
        self.stats.blocks_teleported += len(out)
        return out, report

    # --- optional vLLM wiring (only used when vLLM is installed) ---
    @classmethod
    def from_vllm_config(cls, config, **overrides) -> "SingularityKVConnector":
        """Build a connector from a vLLM ModelConfig-like object.

        Reads num_kv_heads / head_dim / block_size / dtype when available and
        falls back to defaults otherwise. Never imports vLLM at module import
        time -- only when this constructor is called in a real engine.
        """
        def _get(obj, *names, default=None):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return default

        kwargs = dict(
            num_kv_heads=_get(config, "num_kv_heads",
                              "num_key_value_heads", default=8),
            head_dim=_get(config, "head_dim", default=128),
            block_size=_get(config, "block_size", default=16),
            dtype=str(_get(config, "kv_cache_dtype", "dtype",
                          default="float16")),
        )
        kwargs.update(overrides)
        logger.info("SingularityKVConnector.from_vllm_config -> %s", kwargs)
        return cls(**kwargs)
