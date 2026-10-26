"""Certified teleports — SHACKLE-gated KV movement with decision records.

A KV teleport is an agent action: data crossing machines on an agent's
behalf. This module puts that action behind the same governance the
SHACKLE standard specifies for tool calls, so a teleport can be
*proven*, not just performed:

- Every ``teleport_context`` call on a :class:`GovernedFabric` is
  evaluated **before any byte moves** by the real SHACKLE ``decide()``
  (from the ``pyshackle`` package — no local reimplementation) against
  a byte-denominated envelope.
- Every verdict is written to a hash-chained **decision ledger**:
  the plan (blocks, tiers, bytes), its canonical digest, the nonce,
  the verdict, and SHACKLE's reason string.
- **DENY fails closed**: the exception is raised and the transport is
  never touched.
- **HITL parks the exact plan**: the held call is bound by nonce and
  canonical digest. A human decision resumes it through SHACKLE's
  HITL transition contract; if the plan has changed since the hold,
  the executor refuses (digest mismatch, fail closed).

Units: SHACKLE's envelope fields are named ``*_usd``. This gate
denominates them in **bytes** (estimated_cost := planned bytes,
budget := byte budget). The decision records themselves use honest
byte-denominated field names.

Requires the ``governance`` extra: ``pip install singularity-dkcs[governance]``.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import shackle
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "singularity.sdk.governance requires pyshackle: "
        "pip install singularity-dkcs[governance]"
    ) from e

from ..core.page_table import FetchPlan
from .fabric import ContextFabric, TeleportReport

TOOL_NAME = "kv.teleport"


def _evaluator() -> str:
    try:
        from importlib.metadata import version

        return f"pyshackle-{version('pyshackle')} shackle.decide"
    except Exception:
        return "pyshackle shackle.decide"


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# ── Exceptions ────────────────────────────────────────────────────────


class TeleportDenied(Exception):
    """The gate denied this teleport. No bytes moved."""

    def __init__(self, record: "DecisionRecord"):
        super().__init__(
            f"teleport denied ({record.reason}) — record {record.record_id}"
        )
        self.record = record


class TeleportHitlRequired(Exception):
    """The gate parked this teleport for a human decision.

    Resolve it with ``GovernedFabric.approve_teleport(record_id, ...)``.
    No bytes have moved.
    """

    def __init__(self, record: "DecisionRecord"):
        super().__init__(
            f"teleport held for human decision ({record.reason}) — "
            f"record {record.record_id}"
        )
        self.record = record
        self.record_id = record.record_id


# ── Decision record ───────────────────────────────────────────────────


@dataclass
class DecisionRecord:
    """One governed decision. Immutable once appended to the ledger:
    the hash covers every field below except ``hash`` itself."""

    seq: int
    ts: float
    record_id: str
    tool: str
    nonce: str
    plan_digest: str
    block_ids: List[str]
    tiers: Dict[str, int]
    planned_bytes: int
    hot_bytes: int
    verdict: str  # ALLOW | DENY | HITL (shackle.decide's vocabulary)
    reason: str  # shackle.decide's reason string, verbatim
    evaluator: str
    resumes: Optional[str] = None  # record_id this transition resumes
    prev_hash: str = ""
    hash: str = ""

    def body(self) -> dict:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "record_id": self.record_id,
            "tool": self.tool,
            "nonce": self.nonce,
            "plan_digest": self.plan_digest,
            "block_ids": self.block_ids,
            "tiers": self.tiers,
            "planned_bytes": self.planned_bytes,
            "hot_bytes": self.hot_bytes,
            "verdict": self.verdict,
            "reason": self.reason,
            "evaluator": self.evaluator,
            "resumes": self.resumes,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(
            (self.prev_hash + _canonical(self.body())).encode()
        ).hexdigest()


# ── The gate ──────────────────────────────────────────────────────────


class TeleportGate:
    """A SHACKLE envelope over fabric teleports, with a decision ledger.

    Parameters (the envelope):
      budget_bytes: total byte budget across all teleports on this
        gate (None = no budget rules). Consumption is charged at the
        ACTUAL bytes moved, after execution.
      hitl_mode: "never" (default), "always" (every teleport needs a
        human decision), or "on_threshold" (HITL once the remaining
        budget fraction falls to hitl_budget_threshold).
      max_repeat_calls: SHACKLE's consecutive-repeat cap (off by default;
        repeated teleports are normal fabric operation).
    """

    def __init__(
        self,
        budget_bytes: Optional[int] = None,
        hitl_mode: str = "never",
        hitl_budget_threshold: Optional[float] = None,
        max_repeat_calls: Optional[int] = None,
    ) -> None:
        if hitl_mode not in ("never", "always", "on_threshold"):
            raise ValueError(f"unknown hitl_mode: {hitl_mode}")
        self.config: Dict = {}
        if budget_bytes is not None:
            self.config["budget_usd"] = float(budget_bytes)
        if hitl_mode != "never":
            self.config["hitl_mode"] = hitl_mode
        if hitl_budget_threshold is not None:
            self.config["hitl_budget_threshold"] = float(hitl_budget_threshold)
        if max_repeat_calls is not None:
            self.config["max_repeat_calls"] = int(max_repeat_calls)

        self.state: Dict = {
            "seen_nonces": [],
            "repeat_counts": {},
            "last_tool_name": None,
            "pending_transition": None,
            "circuit_tripped": False,
        }
        if budget_bytes is not None:
            self.state["budget_remaining_usd"] = float(budget_bytes)
            self.state["budget_initial_usd"] = float(budget_bytes)

        self.records: List[DecisionRecord] = []
        self._pending: Optional[Dict] = None  # {call, record_id}

    # -- ledger ---------------------------------------------------------
    def _append(self, **kw) -> DecisionRecord:
        prev = self.records[-1].hash if self.records else "GENESIS"
        rec = DecisionRecord(
            seq=len(self.records),
            ts=time.time(),
            record_id=uuid.uuid4().hex[:16],
            prev_hash=prev,
            evaluator=_evaluator(),
            **kw,
        )
        rec.hash = rec.compute_hash()
        self.records.append(rec)
        return rec

    def verify_chain(self) -> bool:
        """Recompute the ledger's hash chain. False if any record was
        altered, reordered, or removed."""
        prev = "GENESIS"
        for i, rec in enumerate(self.records):
            if rec.seq != i or rec.prev_hash != prev:
                return False
            if rec.compute_hash() != rec.hash:
                return False
            prev = rec.hash
        return True

    # -- decision -------------------------------------------------------
    def decide_teleport(self, call: Dict, plan: Dict) -> DecisionRecord:
        """Run shackle.decide on a teleport call and record the verdict.
        On HITL the call is parked as the pending transition."""
        verdict, reason = shackle.decide(self.config, self.state, call)
        rec = self._append(
            tool=TOOL_NAME,
            nonce=call.get("nonce", ""),
            plan_digest=shackle.canonical_hash(call.get("params", {})),
            block_ids=plan["block_ids"],
            tiers=plan["tiers"],
            planned_bytes=plan["planned_bytes"],
            hot_bytes=plan["hot_bytes"],
            verdict=verdict,
            reason=reason,
        )
        if verdict == "HITL":
            self._pending = {"call": call, "record_id": rec.record_id}
            self.state["pending_transition"] = {
                "decision": None,
                "record_id": rec.record_id,
            }
        return rec

    def commit(self, record: DecisionRecord, actual_bytes: int) -> None:
        """Charge an executed ALLOW: consume the nonce, charge the
        budget at actual bytes, count the repeat. Called by the fabric
        only after the move actually happened."""
        self.state["seen_nonces"].append(record.nonce)
        if "budget_remaining_usd" in self.state:
            self.state["budget_remaining_usd"] -= float(actual_bytes)
        rc = self.state["repeat_counts"]
        rc[TOOL_NAME] = rc.get(TOOL_NAME, 0) + 1
        self.state["last_tool_name"] = TOOL_NAME

    # -- HITL transitions -------------------------------------------------
    @property
    def pending_record_id(self) -> Optional[str]:
        return self._pending["record_id"] if self._pending else None

    def pending_call(self, record_id: str) -> Dict:
        if not self._pending or self._pending["record_id"] != record_id:
            raise KeyError(f"no pending teleport for record {record_id}")
        return self._pending["call"]

    def resolve_transition(
        self, record_id: str, decision: str, plan: Dict
    ) -> DecisionRecord:
        """Present a human decision for the parked call to shackle.decide
        (its HITL transition contract) and record the outcome."""
        call = self.pending_call(record_id)
        self.state["pending_transition"] = {
            "decision": decision,
            "record_id": record_id,
        }
        verdict, reason = shackle.decide(self.config, self.state, call)
        rec = self._append(
            tool=TOOL_NAME,
            nonce=call.get("nonce", ""),
            plan_digest=shackle.canonical_hash(call.get("params", {})),
            block_ids=plan["block_ids"],
            tiers=plan["tiers"],
            planned_bytes=plan["planned_bytes"],
            hot_bytes=plan["hot_bytes"],
            verdict=verdict,
            reason=reason,
            resumes=record_id,
        )
        if verdict == "HITL":
            # defer/escalate: the hold stands, awaiting a further decision.
            self._pending = {"call": call, "record_id": rec.record_id}
            self.state["pending_transition"] = {
                "decision": None,
                "record_id": rec.record_id,
            }
        else:
            self._pending = None
            self.state["pending_transition"] = None
        return rec

    def abort_pending(self, record_id: str, reason: str, plan: Dict) -> DecisionRecord:
        """Executor-side fail-closed: the plan changed under a hold, or
        the hold is being abandoned. Records a DENY and clears the hold."""
        call = self.pending_call(record_id)
        rec = self._append(
            tool=TOOL_NAME,
            nonce=call.get("nonce", ""),
            plan_digest=shackle.canonical_hash(call.get("params", {})),
            block_ids=plan["block_ids"],
            tiers=plan["tiers"],
            planned_bytes=plan["planned_bytes"],
            hot_bytes=plan["hot_bytes"],
            verdict="DENY",
            reason=reason,
            resumes=record_id,
        )
        self._pending = None
        self.state["pending_transition"] = None
        return rec


# ── The governed fabric ───────────────────────────────────────────────


class GovernedFabric(ContextFabric):
    """A ContextFabric whose teleports are certified: gated by a
    TeleportGate (SHACKLE decide) and recorded in its ledger.

    Publishing is unchanged. Teleporting is governed.
    """

    def __init__(self, *args, gate: TeleportGate, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = gate

    # -- plan construction ------------------------------------------------
    def _plan_summary(self, block_ids: List[str], hot_only: bool):
        plan: FetchPlan = self.page_table.plan_fetch(block_ids)
        tiers_lists = {"hot": plan.hot}
        if not hot_only:
            tiers_lists.update({"warm": plan.warm, "cold": plan.cold})
        intended: List[str] = []
        tiers: Dict[str, int] = {}
        planned = 0
        hot_bytes = 0
        for name, ids in tiers_lists.items():
            tiers[name] = len(ids)
            for bid in ids:
                entry = self.page_table.lookup(bid)
                if entry is None:
                    raise KeyError(f"no page-table entry for {bid}")
                n = entry.meta.nbytes()
                planned += n
                if name == "hot":
                    hot_bytes += n
                intended.append(bid)
        summary = {
            "block_ids": sorted(intended),
            "tiers": tiers,
            "planned_bytes": planned,
            "hot_bytes": hot_bytes,
        }
        return summary

    def _call_for(self, summary: Dict, hot_only: bool, nonce: str) -> Dict:
        params = {
            "action": TOOL_NAME,
            "blocks": summary["block_ids"],
            "tiers": summary["tiers"],
            "planned_bytes": summary["planned_bytes"],
            "hot_bytes": summary["hot_bytes"],
            "hot_only": hot_only,
            "ctx": "transparent",
        }
        return {
            "tool_name": TOOL_NAME,
            "nonce": nonce,
            "estimated_cost_usd": float(summary["planned_bytes"]),
            "params": params,
        }

    # -- governed teleport --------------------------------------------------
    def teleport_context(
        self,
        block_ids: List[str],
        hot_only: bool = False,
        idempotency_nonce: Optional[str] = None,
    ) -> TeleportReport:
        pending_id = self.gate.pending_record_id
        if pending_id is not None:
            held = next(
                r for r in self.gate.records if r.record_id == pending_id
            )
            raise TeleportHitlRequired(held)
        summary = self._plan_summary(block_ids, hot_only)
        nonce = idempotency_nonce or uuid.uuid4().hex
        call = self._call_for(summary, hot_only, nonce)
        record = self.gate.decide_teleport(call, summary)
        if record.verdict == "DENY":
            raise TeleportDenied(record)
        if record.verdict == "HITL":
            raise TeleportHitlRequired(record)
        report = super().teleport_context(block_ids, hot_only=hot_only)
        self.gate.commit(record, report.total_bytes)
        return report

    def approve_teleport(
        self, record_id: str, decision: str = "approve"
    ) -> TeleportReport:
        """Resolve a held teleport with a human decision.

        approve → the held plan executes (after the executor verifies
        the plan digest is unchanged); reject → recorded denial;
        defer/escalate → the hold stands."""
        call = self.gate.pending_call(record_id)
        params = call.get("params", {})
        hot_only = bool(params.get("hot_only", False))
        block_ids = list(params.get("blocks", []))

        # Executor digest binding (SHACKLE's transition contract):
        # what executes must be exactly what was held.
        rebuilt = self._plan_summary(block_ids, hot_only)
        rebuilt_call = self._call_for(rebuilt, hot_only, call.get("nonce", ""))
        if shackle.canonical_hash(rebuilt_call["params"]) != shackle.canonical_hash(params):
            rec = self.gate.abort_pending(
                record_id, "executor:plan_digest_mismatch", rebuilt
            )
            raise TeleportDenied(rec)

        record = self.gate.resolve_transition(record_id, decision, rebuilt)
        if record.verdict == "DENY":
            raise TeleportDenied(record)
        if record.verdict == "HITL":
            raise TeleportHitlRequired(record)
        report = super().teleport_context(block_ids, hot_only=hot_only)
        self.gate.commit(record, report.total_bytes)
        return report

    # -- ledger access ------------------------------------------------------
    @property
    def ledger(self) -> List[DecisionRecord]:
        return self.gate.records

    def verify_ledger(self) -> bool:
        return self.gate.verify_chain()
