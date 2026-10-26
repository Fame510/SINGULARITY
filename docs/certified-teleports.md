# Certified Teleports — SHACKLE-Gated KV Movement

A KV teleport is an agent action: data crossing machines on an
agent's behalf. `singularity.sdk.governance` puts that action under
the same governance discipline as any tool call, so a teleport can be
**proven, not just performed**.

## How it works

`GovernedFabric` is a `ContextFabric` whose `teleport_context` is
gated. Before any byte moves:

1. The fabric builds the plan (which blocks, which tiers, how many
   bytes) from the page table.
2. The plan is evaluated by **SHACKLE's `decide()`** — the real pure
   function from the `pyshackle` package, the one the SP/1.0.1
   certification battery covers — against a byte-denominated envelope
   (budget, HITL mode, repeat caps). No local reimplementation: if
   `pyshackle` is absent, the module refuses to load.
3. The verdict and SHACKLE's verbatim reason string are written to a
   **hash-chained decision ledger** with the plan's canonical digest
   and the call's nonce.
4. **ALLOW** → the move executes; the budget is charged at the actual
   bytes moved and the nonce is consumed (a replayed nonce is a
   `DENY: policy_violation:duplicate_nonce`).
   **DENY** → `TeleportDenied`; the transport is never touched.
   **HITL** → `TeleportHitlRequired`; the exact plan is parked.

A held plan resumes only through `approve_teleport(record_id,
decision=...)`, which presents the human decision to `decide()`'s
HITL transition contract. The executor first re-derives the plan and
compares canonical digests: **if the page table changed under the
hold, execution refuses** (`executor:plan_digest_mismatch`) — what
executes must be exactly what was approved.

The ledger verifies with `verify_ledger()`; any altered, reordered,
or removed record breaks the chain.

## Usage

```python
from singularity.sdk.fabric import ContextFabric
from singularity.sdk.governance import (
    GovernedFabric, TeleportGate, TeleportDenied, TeleportHitlRequired,
)

gate = TeleportGate(budget_bytes=1 << 30, hitl_mode="on_threshold",
                    hitl_budget_threshold=0.25)
fab = GovernedFabric(gate=gate)
fab.publish_context(blocks, total_tokens)
try:
    report = fab.teleport_context([b.meta.block_id for b in blocks],
                                  idempotency_nonce="job-4815-attempt-1")
except TeleportDenied as d:
    ...  # d.record carries the verdict, reason, and plan digest
except TeleportHitlRequired as h:
    report = fab.approve_teleport(h.record_id, decision="approve")
```

Install the extra: `pip install singularity-dkcs[governance]`.

## Scope

The gate governs the fabric's Python SDK path (any transport the
fabric is configured with). It certifies *decisions about movement* —
which blocks, under what envelope, decided by which evaluator, with
what outcome. It does not by itself certify the transport's wire
behavior, and the ledger lives where the integrator keeps it
(in-memory by default; persist `gate.records` for audit).
