"""
singularity.sdk.prefetch
========================
Asynchronous prefetch pipeline.

The whole point of tiered sparse teleportation is to overlap compute with
transfer: return the HOT tier synchronously (critical path) while WARM/COLD
blocks stream in the background so they are resident before they are needed.

This module implements a real thread-pool prefetcher over any Transport. It is
transport-agnostic: with the RDMA backend on real hardware the same code
overlaps DMA with GPU compute.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List

from ..core.page_table import GlobalPageTable


@dataclass
class PrefetchHandle:
    """Tracks an in-flight background prefetch."""

    futures: Dict[str, Future] = field(default_factory=dict)

    def wait(self, timeout: float | None = None) -> List[str]:
        done = []
        for bid, fut in self.futures.items():
            fut.result(timeout=timeout)
            done.append(bid)
        return done

    def ready(self) -> List[str]:
        return [bid for bid, f in self.futures.items() if f.done()]


class Prefetcher:
    def __init__(self, page_table: GlobalPageTable, max_workers: int = 4) -> None:
        self.page_table = page_table
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="sing-prefetch")
        self._cache: Dict[str, object] = {}
        self._lock = threading.Lock()

    def _fetch_one(self, block_id: str):
        block, res = self.page_table.fetch(block_id)
        with self._lock:
            self._cache[block_id] = block
        return res

    def prefetch(self, block_ids: List[str]) -> PrefetchHandle:
        handle = PrefetchHandle()
        for bid in block_ids:
            handle.futures[bid] = self._pool.submit(self._fetch_one, bid)
        return handle

    def get_resident(self, block_id: str):
        with self._lock:
            return self._cache.get(block_id)

    def resident_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
