from singularity.core.blocks import KVBlock, Tier
from singularity.core.selector import SelectionConfig, assign_tiers, head_bytes, total_bytes


def _ctx(total_tokens=4096, block_tokens=256):
    blocks, pos, idx = [], 0, 0
    while pos < total_tokens:
        n = min(block_tokens, total_tokens - pos)
        blocks.append(KVBlock.random(f"B{idx}", 0, pos, n, seed=idx))
        pos += n
        idx += 1
    return blocks, total_tokens


def test_sinks_and_recency_become_hot():
    blocks, total = _ctx()
    metas = [b.meta for b in blocks]
    assign_tiers(metas, total, SelectionConfig(head_fraction=0.25,
                 sink_tokens=256, recency_tokens=512))
    tiers = {m.block_id: m.tier for m in metas}
    # first block (sink) and last block (recency) must be HOT
    assert tiers[blocks[0].meta.block_id] == Tier.HOT
    assert tiers[blocks[-1].meta.block_id] == Tier.HOT


def test_head_fraction_controls_hot_volume():
    blocks, total = _ctx()
    metas = [b.meta for b in blocks]
    assign_tiers(metas, total, SelectionConfig(head_fraction=0.12))
    frac = head_bytes(metas) / total_bytes(metas)
    # HOT volume should be in the neighbourhood of the target fraction
    assert 0.05 <= frac <= 0.30


def test_empty_is_safe():
    assert assign_tiers([], 0, SelectionConfig()) == []
