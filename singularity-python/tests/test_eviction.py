from singularity.core.blocks import KVBlock, Tier
from singularity.core.eviction import LRUEvictionPolicy


def _meta(bid, tier, tokens=16):
    b = KVBlock.random(bid, 0, 0, tokens, seed=hash(bid) % 1000)
    b.meta.tier = tier
    return b.meta


def test_cold_evicted_before_hot():
    one = _meta("cold", Tier.COLD).nbytes()
    pol = LRUEvictionPolicy(capacity_bytes=one)  # room for exactly one block
    pol.admit(_meta("hot", Tier.HOT))
    evicted = pol.admit(_meta("cold", Tier.COLD))
    # adding a second block over budget should evict the COLD one, protect HOT
    assert "cold" in evicted
    assert "hot" not in evicted


def test_used_bytes_tracks():
    pol = LRUEvictionPolicy(capacity_bytes=10**12)
    m = _meta("a", Tier.WARM)
    pol.admit(m)
    assert pol.used_bytes == m.nbytes()
