import numpy as np

from singularity.integrations import KVConnectorLike, SingularityKVConnector


def _kv(tokens, heads=8, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    shape = (tokens, heads, dim)
    return (rng.standard_normal(shape).astype("float16"),
            rng.standard_normal(shape).astype("float16"))


def test_connector_satisfies_protocol():
    c = SingularityKVConnector(num_kv_heads=8, head_dim=128, block_size=16)
    assert isinstance(c, KVConnectorLike)


def test_publish_then_teleport_roundtrips_tensors():
    c = SingularityKVConnector(num_kv_heads=8, head_dim=128, block_size=16)
    block_ids = [0, 1, 2, 3]
    keys, values, positions = [], [], []
    for i in block_ids:
        k, v = _kv(16, seed=i)
        keys.append(k)
        values.append(v)
        positions.append((i * 16, i * 16 + 16))
    c.publish_blocks("req-1", block_ids, keys, values, positions, total_tokens=64)

    out, report = c.teleport_blocks("req-1", block_ids, hot_only=False)
    assert set(out.keys()) == set(block_ids)
    # tensors survive the round-trip byte-exact
    assert np.array_equal(out[0][0], keys[0])
    assert report.num_blocks == len(block_ids)
    assert c.stats.cache_hits == len(block_ids)


def test_teleport_missing_counts_as_miss():
    c = SingularityKVConnector()
    out, _ = c.teleport_blocks("nope", [7, 8], hot_only=False)
    assert out == {}
    assert c.stats.cache_misses == 2


def test_from_vllm_config_reads_fields():
    class Cfg:
        num_kv_heads = 16
        head_dim = 64
        block_size = 32
        kv_cache_dtype = "float16"
    c = SingularityKVConnector.from_vllm_config(Cfg())
    assert c.num_kv_heads == 16 and c.head_dim == 64 and c.block_size == 32
