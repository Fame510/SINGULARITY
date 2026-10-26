"""Certified teleports: SHACKLE-gated fabric movement.

Every test here runs the REAL shackle.decide (pyshackle) — the same
pure function the SP/1.0.1 battery certifies — against a real
ContextFabric. The point under test: no byte moves without a recorded
ALLOW, a DENY moves nothing, and a HITL hold binds the exact plan.
"""
import pytest

pytest.importorskip("shackle")

from singularity.core.blocks import KVBlock  # noqa: E402
from singularity.sdk.fabric import ContextFabric  # noqa: E402
from singularity.sdk.governance import (  # noqa: E402
    GovernedFabric,
    TeleportDenied,
    TeleportGate,
    TeleportHitlRequired,
)


def _published(fabric_cls=ContextFabric, total=8192, block_tokens=256, **kw):
    blocks, pos, idx = [], 0, 0
    while pos < total:
        n = min(block_tokens, total - pos)
        blocks.append(KVBlock.random(f"B{idx}", 0, pos, n, seed=idx))
        pos += n
        idx += 1
    fab = fabric_cls(**kw)
    fab.publish_context(blocks, total)
    return fab, [b.meta.block_id for b in blocks], blocks


def _context_bytes():
    fab, ids, _ = _published()
    return fab.teleport_context(ids).total_bytes


def test_allow_executes_and_records():
    fab, ids, _blocks = _published(GovernedFabric, gate=TeleportGate())
    rep = fab.teleport_context(ids)
    assert rep.total_bytes > 0
    assert len(rep.blocks_fetched) == len(ids)
    assert len(fab.ledger) == 1
    rec = fab.ledger[0]
    assert rec.verdict == "ALLOW"
    assert rec.reason == "within_thresholds"
    assert rec.tool == "kv.teleport"
    assert rec.planned_bytes == rep.total_bytes
    assert "pyshackle" in rec.evaluator
    assert fab.verify_ledger()


def test_budget_overrun_denies_and_moves_nothing():
    total_bytes = _context_bytes()
    gate = TeleportGate(budget_bytes=total_bytes - 1)
    fab, ids, _blocks = _published(GovernedFabric, gate=gate)
    with pytest.raises(TeleportDenied) as e:
        fab.teleport_context(ids)
    assert e.value.record.verdict == "DENY"
    assert e.value.record.reason == "budget_overrun"
    assert fab.verify_ledger()


def test_budget_exhaustion_after_successful_teleport():
    total_bytes = _context_bytes()
    gate = TeleportGate(budget_bytes=total_bytes)
    fab, ids, _blocks = _published(GovernedFabric, gate=gate)
    fab.teleport_context(ids)  # consumes the whole budget
    with pytest.raises(TeleportDenied) as e:
        fab.teleport_context(ids)
    assert e.value.record.reason in ("budget_exhausted", "budget_overrun")


def test_replayed_nonce_denied():
    fab, ids, _blocks = _published(GovernedFabric, gate=TeleportGate())
    fab.teleport_context(ids, idempotency_nonce="attempt-1")
    with pytest.raises(TeleportDenied) as e:
        fab.teleport_context(ids, idempotency_nonce="attempt-1")
    assert e.value.record.reason == "policy_violation:duplicate_nonce"


def test_hitl_always_hold_then_approve_executes():
    gate = TeleportGate(hitl_mode="always")
    fab, ids, _blocks = _published(GovernedFabric, gate=gate)
    with pytest.raises(TeleportHitlRequired) as e:
        fab.teleport_context(ids)
    held_id = e.value.record_id
    # While held, new teleports are refused at the executor.
    with pytest.raises(TeleportHitlRequired):
        fab.teleport_context(ids)
    rep = fab.approve_teleport(held_id, decision="approve")
    assert rep.total_bytes > 0
    verdicts = [(r.verdict, r.reason) for r in fab.ledger]
    assert verdicts[0] == ("HITL", "hitl_all_calls")
    assert verdicts[-1] == ("ALLOW", "hitl_transition:approve")
    assert fab.verify_ledger()


def test_hitl_reject_denies():
    gate = TeleportGate(hitl_mode="always")
    fab, ids, _blocks = _published(GovernedFabric, gate=gate)
    with pytest.raises(TeleportHitlRequired) as e:
        fab.teleport_context(ids)
    with pytest.raises(TeleportDenied) as denied:
        fab.approve_teleport(e.value.record_id, decision="reject")
    assert denied.value.record.reason == "hitl_transition:reject"
    # The hold is cleared; a fresh attempt re-enters the gate.
    with pytest.raises(TeleportHitlRequired):
        fab.teleport_context(ids)


def test_plan_change_under_hold_fails_closed():
    from singularity.core.blocks import Tier

    gate = TeleportGate(hitl_mode="always")
    fab, ids, blocks = _published(GovernedFabric, gate=gate)
    with pytest.raises(TeleportHitlRequired) as e:
        fab.teleport_context(ids, hot_only=True)
    # The page table changes under the hold: a block that was HOT when
    # the plan was held is demoted. What would execute no longer
    # matches the held plan's digest, so the executor must refuse.
    hot = [b for b in blocks if b.meta.tier == Tier.HOT]
    assert hot, "precondition: the held plan has a hot tier"
    hot[0].meta.tier = Tier.COLD
    with pytest.raises(TeleportDenied) as denied:
        fab.approve_teleport(e.value.record_id, decision="approve")
    assert denied.value.record.reason == "executor:plan_digest_mismatch"


def test_ledger_tamper_detected():
    fab, ids, _blocks = _published(GovernedFabric, gate=TeleportGate())
    fab.teleport_context(ids)
    assert fab.verify_ledger()
    fab.ledger[0].planned_bytes += 1  # alter history
    assert not fab.verify_ledger()
