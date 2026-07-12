"""
singularity.core.metrics
========================
Lightweight, dependency-free metrics for the fabric.

Exposes counters/histograms and a Prometheus-style text exposition so a real
deployment can scrape fabric health (teleport latency, hit rate, bytes moved).
No external client library required.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Metrics:
    counters: Dict[str, float] = field(default_factory=dict)
    _hist: Dict[str, List[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._hist.setdefault(name, []).append(value)

    def percentile(self, name: str, p: float) -> float:
        with self._lock:
            xs = sorted(self._hist.get(name, []))
        if not xs:
            return 0.0
        k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
        return xs[k]

    def snapshot(self) -> dict:
        with self._lock:
            out = dict(self.counters)
            for name, xs in self._hist.items():
                if xs:
                    out[f"{name}_p50"] = self.percentile(name, 50)
                    out[f"{name}_p99"] = self.percentile(name, 99)
                    out[f"{name}_count"] = float(len(xs))
        return out

    def prometheus(self) -> str:
        lines = []
        for k, v in self.snapshot().items():
            metric = "singularity_" + k.replace(".", "_").replace("-", "_")
            lines.append(f"{metric} {v}")
        return "\n".join(lines) + "\n"
