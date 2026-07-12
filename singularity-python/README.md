# singularity-dkcs (Python SDK)

Open-source core of the **SINGULARITY Disaggregated KV-Cache Service** — the
runnable implementation of *context teleportation*: split a sequence's KV-cache
into blocks, rank them by importance (sparse selection), and move the critical
subset first.

This package runs today on a laptop or in CI over a loopback transport. No GPU,
no RDMA NIC, and no fabricated numbers required.

## Install

```bash
cd singularity-python
pip install -e ".[dev]"
```

## Quickstart

```python
from singularity import ContextFabric, KVBlock, SelectionConfig

# Build a context out of KV blocks (here: synthetic tensors).
blocks = [KVBlock.random(f"B{i}", layer=0, start_pos=i*256, num_tokens=256, seed=i)
          for i in range(64)]

fabric = ContextFabric(selection=SelectionConfig(head_fraction=0.12))
fabric.publish_context(blocks, total_tokens=64 * 256)

ids = [b.meta.block_id for b in blocks]

# Sparse teleport: fetch only the HOT (critical) tier first.
report = fabric.teleport_context(ids, hot_only=True)
print(f"HOT tier: {report.head_mb if hasattr(report,'head_mb') else report.head_bytes/1e6:.2f} MB")
print(f"time-to-first-token proxy: {report.time_to_first_token_ms:.3f} ms")
```

## What is real vs. what is a target

| Component | Status in this repo |
|---|---|
| `core.blocks`, `core.page_table`, `core.selector` | **Real, tested Python** |
| `sdk.ContextFabric` teleport orchestration | **Real, tested** |
| `transport.InMemoryTransport` (loopback) | **Real** — measures serialize/copy/reconstruct |
| `transport.RdmaTransport` (RoCE v2 / GPU-Direct) | **Hardware stub** — raises `NotImplementedError`; lives in the Rust crate on real HW |
| `DistributedLLM.generate` | Requires a real engine (e.g. vLLM); refuses to fake output |
| 8.3× / 26× speedup, 320→38 ms, 85% util, $4.2M | **Design targets** — see `docs/ARCHITECTURE.md`, must be measured on H100 + RoCE |

## Benchmark

```bash
python scripts/slb_benchmark.py --context-tokens 32768 --layers 4 --json
```

The benchmark reports **measured** loopback numbers and explicitly labels that
it measures data-volume reduction, not RDMA wire latency.

## Tests

```bash
cd singularity-python && pytest
```
