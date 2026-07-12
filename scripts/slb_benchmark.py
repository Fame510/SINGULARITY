#!/usr/bin/env python3
"""
SINGULARITY Latency Benchmark (HONEST EDITION)
==============================================
Measures the *real* cost of moving a KV-cache context two ways over the active
transport:

  * FULL PULL    -- fetch every block (HOT + WARM + COLD)
  * SPARSE (HOT) -- fetch only the critical HOT tier (sparse teleportation),
                    i.e. the time-to-first-token path

IMPORTANT -- READ THIS BEFORE QUOTING NUMBERS
---------------------------------------------
By default this runs over the InMemoryTransport (loopback): it measures real
serialization + memory copy + reconstruction of real NumPy tensors. It does NOT
measure RDMA-over-the-wire latency. Loopback speedups reflect the *data-volume
reduction* of sparse selection (you move ~head_fraction of the bytes), which is
the mechanism SINGULARITY exploits -- but absolute wire latency on H100+RoCE
must be measured on real hardware. Those hardware figures are DESIGN TARGETS in
docs/ARCHITECTURE.md, not outputs of this script.

Usage:
    python slb_benchmark.py --context-tokens 32768 --layers 4 --json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "singularity-python"))

from singularity.core.blocks import KVBlock  # noqa: E402
from singularity.core.selector import SelectionConfig  # noqa: E402
from singularity.sdk.fabric import ContextFabric  # noqa: E402


def build_context(total_tokens, layers, block_tokens, num_heads, head_dim, dtype, seed):
    blocks = []
    idx = 0
    for layer in range(layers):
        pos = 0
        while pos < total_tokens:
            n = min(block_tokens, total_tokens - pos)
            blocks.append(KVBlock.random(
                block_id=f"L{layer}-B{idx}", layer=layer, start_pos=pos,
                num_tokens=n, num_heads=num_heads, head_dim=head_dim,
                dtype=dtype, seed=seed + idx,
            ))
            pos += n
            idx += 1
    return blocks


def run(args):
    blocks = build_context(
        args.context_tokens, args.layers, args.block_tokens,
        args.num_heads, args.head_dim, args.dtype, args.seed,
    )
    fabric = ContextFabric(selection=SelectionConfig(head_fraction=args.head_fraction))
    fabric.publish_context(blocks, total_tokens=args.context_tokens)
    block_ids = [b.meta.block_id for b in blocks]

    full_ms, sparse_ms = [], []
    for _ in range(args.iterations):
        full_ms.append(fabric.teleport_context(block_ids, hot_only=False).total_ms)
        sparse_ms.append(fabric.teleport_context(block_ids, hot_only=True).time_to_first_token_ms)

    sample = fabric.teleport_context(block_ids, hot_only=False)
    avg_full = statistics.mean(full_ms)
    avg_sparse = statistics.mean(sparse_ms)
    speedup = (avg_full / avg_sparse) if avg_sparse > 0 else float("inf")

    return {
        "transport": fabric.transport.name,
        "measured": True,
        "note": "Loopback backend: measures data-volume reduction, NOT RDMA wire latency.",
        "context_tokens": args.context_tokens,
        "layers": args.layers,
        "total_blocks": len(blocks),
        "total_mb": round(sample.total_bytes / 1e6, 2),
        "head_mb": round(sample.head_bytes / 1e6, 2),
        "head_fraction_measured": round(sample.head_fraction, 4),
        "iterations": args.iterations,
        "avg_full_pull_ms": round(avg_full, 4),
        "avg_sparse_hot_ms": round(avg_sparse, 4),
        "speedup_x": round(speedup, 2),
    }


def main():
    p = argparse.ArgumentParser(description="SINGULARITY honest latency benchmark")
    p.add_argument("--context-tokens", type=int, default=32768)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--block-tokens", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--head-fraction", type=float, default=0.12)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json", action="store_true", help="emit JSON only")
    args = p.parse_args()

    result = run(args)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("=" * 68)
    print(" SINGULARITY Latency Benchmark  (transport: %s)" % result["transport"])
    print("=" * 68)
    print(" Context: {ctx} tokens, {layers} layers, {blocks} blocks, {mb} MB".format(
        ctx=result["context_tokens"], layers=result["layers"],
        blocks=result["total_blocks"], mb=result["total_mb"]))
    print(" HOT (critical) tier: {hmb} MB  ({frac:.1%} of context)".format(
        hmb=result["head_mb"], frac=result["head_fraction_measured"]))
    print("-" * 68)
    print(" {:<26} {:>14} {:>10}".format("Strategy", "Avg latency", "Speedup"))
    print(" {:<26} {:>11.3f} ms {:>9}".format("Full pull (all tiers)", result["avg_full_pull_ms"], "1.0x"))
    print(" {:<26} {:>11.3f} ms {:>8.1f}x".format("Sparse teleport (HOT)", result["avg_sparse_hot_ms"], result["speedup_x"]))
    print("-" * 68)
    print(" NOTE: %s" % result["note"])


if __name__ == "__main__":
    main()
