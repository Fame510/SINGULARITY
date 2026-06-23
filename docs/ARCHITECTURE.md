# SINGULARITY Architecture — System Design Document v1.0

## Executive Summary

Current AI infrastructure is built on the "Static Container" paradigm. LLM engines (vLLM, TGI) pre-allocate and hold VRAM, treating GPUs as isolated silos. In an agentic world—where workloads are spiking, transient, and high-context—this leads to 60-80% VRAM underutilization and catastrophic "Cold Start" latencies.

SINGULARITY is a low-level system shim that decouples the **Model Weights** (Static) from the **KV-Cache/Context** (Dynamic). It transforms a GPU cluster into a fluid Global Memory Fabric, enabling sub-50ms context teleportation and near-100% hardware efficiency.

---

## 1. Physics & Latency Budget

### The Problem
- **Standard Cold Start:** Reload model weights (minutes)
- **Pod Migration:** Move KV-cache between nodes (320ms for 16GB over 400Gbps)
- **Target:** Sub-50ms perceived cold start

### The Solution: Sparse Teleportation
Move only the critical 12% of KV-cache that the model actually attends to.

| Tier | Data | Size | Latency | When |
|------|------|------|---------|------|
| Hot Block (Critical Path) | Recent tokens + Attention sinks | ~256MB | 5ms (RDMA) | Before first token |
| Warm Block | Mid-range context | ~2GB | 50ms (RDMA) | During first 20 tokens |
| Cold Block | Full history | ~14GB | 300ms (RDMA) | Background stream |

**Key Insight:** ~85% of next-token attention weight comes from only 12% of total KV-cache (recent tokens + initial attention sinks).

---

## 2. Technical Architecture

### Layer 0: Native vLLM KVConnector (Open Source)

Instead of LD_PRELOAD hijacking (breaks CUDA graphs), we implement vLLM's native `KVConnector` interface.

```
vLLM Engine → KVConnector.allocate_block(block_id)
  → GlobalPageTable.lookup(block_id)
    → [Local VRAM] Return immediately
    → [Remote] RDMA fetch from peer node
      → Return after transfer complete
```

**Implementation:** `singularity-core/src/kv_connector.cpp`

### Layer 1: Transport Fabric (Open Source)

High-performance RDMA transport using UCX (Unified Communication X) over RoCE v2.

- **Protocol:** IBVerbs / UCX
- **Bandwidth:** 400 Gbps (50 GB/s)
- **Latency:** ~2-5μs per 16MB page transfer
- **Zero-Copy:** GPU Direct RDMA — data moves GPU→GPU without touching CPU

**Implementation:** `singularity-transport/src/`

### Layer 2: Predictive Oracle (Closed Source)

The proprietary moat. Analyzes attention patterns to predict which blocks to pre-fetch.

- **Attention-Head Importance Sampling:** Ranks KV-cache blocks by historical attention weight
- **Speculative Pre-Fetch:** Begins streaming expected blocks before the request lands
- **Accuracy Target:** 99.9% of token predictions correct using 12% of full cache

**Implementation:** `singularity-oracle/src/`

### Layer 3: Orchestrator Daemon (Open Source)

Manages cluster topology, health, and scheduling.

- **Deterministic Latency Scheduler:** Replaces KubeRay head node
- **Scheduling Unit:** Context-Page (atomic), not Pod (heavy)
- **Failover:** Micro-redirection (2ms) vs Pod restart (60s)

---

## 3. Competitive Arbitrage

| Metric | Standard (K8s/vLLM) | SINGULARITY |
|--------|---------------------|-------------|
| Context Switching | Reload model (60-120s) | Swap pointers (<50ms) |
| GPU Utilization | ~25% (Pod Camping) | ~85% (Dynamic Flow) |
| OpEx per 1M tokens | $10.00 | $1.80 |
| Scaling logic | Horizontal (more pods) | Fluid (shared VRAM pool) |
| Cold start | 60-120 seconds | <50ms |

---

## 4. Go-To-Market Strategy

### Phase 1: The "VRAM Leak" Fix (Weeks 1-3)
- Build the Rust transport layer
- Implement basic vLLM KVConnector
- Create CLI audit tool showing GPU waste

### Phase 2: The "Teleport" Demo (Weeks 4-8)
- Viral video: agent workflow moving across 10 GPUs without lag
- Submit "High-Performance Memory Backend" PR to vLLM
- Open-source the transport layer → GitHub stars

### Phase 3: The Exit (Month 6)
- Approach CoreWeave, Lambda Labs, Crusoe Energy
- Show: "We found $4.2M in idle H100s on a cluster just like yours"
- Target: $1B acquisition

---

## 5. Hardware Requirements

- **GPUs:** NVIDIA H100/A100 with NVLink
- **Networking:** InfiniBand NDR400 or RoCE v2 (400 Gbps)
- **Memory:** Host RAM 512GB+ per node for KV-cache staging
- **Storage:** NVMe for cold tier

---

## 6. Implementation Status

| Component | Status | Language |
|-----------|--------|----------|
| Architecture spec | ✅ Complete | — |
| Transport fabric (open) | 🔧 In development | Rust |
| vLLM KVConnector (open) | 🔧 In development | C++/Python |
| Predictive Oracle (closed) | 🔧 In development | Rust |
| Python SDK | 🔧 In development | Python |
| Benchmark suite | 🔧 In development | Python |
