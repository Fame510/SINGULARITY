import numpy as np
import pytest

from singularity.core.blocks import KVBlock
from singularity.transport.memory import InMemoryTransport


def test_roundtrip_preserves_bytes():
    t = InMemoryTransport()
    b = KVBlock.random("b0", 0, 0, 32, seed=3)
    put = t.put(b)
    assert put.nbytes > 0
    got, res = t.get("b0")
    assert np.array_equal(got.key, b.key)
    assert np.array_equal(got.value, b.value)
    assert res.nbytes == put.nbytes


def test_missing_block_raises():
    t = InMemoryTransport()
    with pytest.raises(KeyError):
        t.get("nope")


def test_rdma_backend_refuses_without_hardware():
    from singularity.transport.rdma import RdmaTransport
    with pytest.raises(NotImplementedError):
        RdmaTransport()
