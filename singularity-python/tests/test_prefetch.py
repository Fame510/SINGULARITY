from singularity.core.blocks import KVBlock
from singularity.sdk.fabric import ContextFabric
from singularity.sdk.prefetch import Prefetcher


def test_background_prefetch_makes_blocks_resident():
    blocks = [KVBlock.random(f"B{i}", 0, i * 16, 16, seed=i) for i in range(12)]
    fab = ContextFabric()
    fab.publish_context(blocks, total_tokens=12 * 16)
    ids = [b.meta.block_id for b in blocks]
    with Prefetcher(fab.page_table, max_workers=4) as pf:
        handle = pf.prefetch(ids)
        done = handle.wait(timeout=10)
        assert set(done) == set(ids)
        assert pf.resident_count() == len(ids)
        assert pf.get_resident(ids[0]) is not None
