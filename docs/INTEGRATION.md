# SINGULARITY <-> vLLM Integration

This document describes how the open-source SINGULARITY fabric plugs into
[vLLM](https://github.com/vllm-project/vllm) as a KV-cache connector.

## Why a version-tolerant connector

vLLM's KV-transfer surface has evolved (`KVConnectorBase` -> `KVConnectorBase_V1`).
Rather than hard-subclass a moving target, `SingularityKVConnector` exposes the
operations vLLM actually needs and adapts to them, so it keeps working across
vLLM releases:

- `publish_blocks(request_id, block_ids, keys, values, positions, total_tokens)`
  Called after prefill produces KV blocks. Blocks are scored by the sparse
  selector (attention-sink + recency), tiered HOT/WARM/COLD, and published to
  the fabric page table.
- `teleport_blocks(request_id, block_ids, hot_only=False)`
  Called on the decode/consumer node. Fetches the critical HOT tier first
  (time-to-first-token path); WARM/COLD stream via the async `Prefetcher`.

## Mapping vLLM block ids

vLLM identifies KV blocks by integer ids per request. The connector maps them
to fabric ids as `f"{request_id}::{block_id}"` so multiple concurrent requests
never collide while identical content still dedups via the block content hash.

## Using it without vLLM

The connector is fully testable without vLLM installed (see
`tests/test_vllm_connector.py`): drive `publish_blocks` / `teleport_blocks`
directly with NumPy tensors. When vLLM IS present, build one with
`SingularityKVConnector.from_vllm_config(model_config)`.

## Status

- Connector logic, id-mapping, tiering, dedup, prefetch: **implemented + tested**.
- Actual wiring into a live vLLM engine on GPUs, and RDMA transport: requires
  the native `singularity-transport` crate + hardware. The connector's contract
  is stable; only the transport backend swaps.
