from singularity.core.blocks import KVBlock, Tier
from singularity.core.selector import SelectionConfig
from singularity.sdk.fabric import ContextFabric


def _blocks(total=8192, block_tokens=256):
    bs, pos, idx = [], 0, 0
    while pos < total:
        n = min(block_tokens, total - pos)
        bs.append(KVBlock.random(f"B{idx}", 0, pos, n, seed=idx))
        pos += n
        idx += 1
    return bs, total


def test_publish_and_teleport_full():
    blocks, total = _blocks()
    fab = ContextFabric(selection=SelectionConfig(head_fraction=0.12))
    fab.publish_context(blocks, total)
    ids = [b.meta.block_id for b in blocks]
    rep = fab.teleport_context(ids, hot_only=False)
    assert rep.num_blocks == len(ids)
    assert len(rep.blocks_fetched) == len(ids)
    assert rep.total_bytes > 0


def test_hot_only_moves_less_than_full():
    blocks, total = _blocks()
    fab = ContextFabric(selection=SelectionConfig(head_fraction=0.12))
    fab.publish_context(blocks, total)
    ids = [b.meta.block_id for b in blocks]
    hot = fab.teleport_context(ids, hot_only=True)
    full = fab.teleport_context(ids, hot_only=False)
    # HOT tier is a strict subset of the full context volume
    assert 0 < hot.head_bytes <= full.total_bytes
    assert len(hot.blocks_fetched) < len(full.blocks_fetched)


def test_dedup_identical_blocks():
    # Two identical-content blocks should dedup in the page table.
    b1 = KVBlock.random("dup-1", 0, 0, 16, seed=99)
    b2 = KVBlock.random("dup-2", 0, 0, 16, seed=99)
    fab = ContextFabric()
    fab.publish_context([b1, b2], total_tokens=16)
    assert fab.page_table.lookup("dup-1").content_hash == \
           fab.page_table.lookup("dup-2").content_hash
