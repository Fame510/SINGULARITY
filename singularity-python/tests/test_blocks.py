import numpy as np
import pytest

from singularity.core.blocks import BlockMeta, KVBlock, Tier


def test_block_shapes_and_nbytes():
    b = KVBlock.random("b0", layer=0, start_pos=0, num_tokens=16,
                       num_heads=8, head_dim=128, dtype="float16", seed=1)
    assert b.key.shape == (16, 8, 128)
    assert b.value.shape == (16, 8, 128)
    # 2 tensors * 16 * 8 * 128 * 2 bytes(fp16)
    assert b.meta.nbytes() == 2 * 16 * 8 * 128 * 2


def test_shape_mismatch_raises():
    meta = BlockMeta("x", 0, 0, 4, 8, 128, "float16")
    with pytest.raises(ValueError):
        KVBlock(meta=meta, key=np.zeros((3, 8, 128), "float16"),
                value=np.zeros((4, 8, 128), "float16"))


def test_content_hash_is_deterministic_and_sensitive():
    a = KVBlock.random("a", 0, 0, 8, seed=7)
    b = KVBlock.random("b", 0, 0, 8, seed=7)   # same content
    c = KVBlock.random("c", 0, 0, 8, seed=8)   # different content
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_tier_ordering():
    assert Tier.HOT < Tier.WARM < Tier.COLD
