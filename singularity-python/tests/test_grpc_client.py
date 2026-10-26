"""Tests for the singd gRPC client.

The unit tests exercise the client-side wire encoding against the
checked-in protobuf stubs (no server needed). The end-to-end test
drives a REAL singd binary over a real socket; it runs only when
SINGULARITY_SINGD_BIN points at a built singd (see the e2e job in
the engine repo's CI), and skips cleanly otherwise.
"""
import os
import socket
import subprocess
import time

import numpy as np
import pytest

grpc = pytest.importorskip("grpc")

from singularity.integrations.grpc_client import (  # noqa: E402
    SingdClient,
    encode_attention_chunk,
    grpc_available,
)


def test_grpc_available_and_chunk_encoding():
    if not grpc_available():
        pytest.skip("grpc extra unavailable in this environment "
                    "(needs grpcio + protobuf 7.x, Python >= 3.10)")
    attn = np.arange(8 * 64, dtype=np.float32).reshape(8, 64) / 100.0
    chunk = encode_attention_chunk("s1", 3, attn)
    assert chunk.session_id == "s1"
    assert chunk.layer == 3
    assert chunk.num_heads == 8
    assert chunk.seq_len == 64
    assert chunk.encoding == "raw_f32"
    # little-endian float32, row-major, exactly heads*seq*4 bytes
    assert len(chunk.attention_data) == 8 * 64 * 4
    back = np.frombuffer(chunk.attention_data, dtype="<f4").reshape(8, 64)
    assert np.allclose(back, attn)


def test_chunk_encoding_rejects_non_2d():
    with pytest.raises(ValueError):
        encode_attention_chunk("s1", 0, np.zeros((4,), dtype=np.float32))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.skipif(
    not os.environ.get("SINGULARITY_SINGD_BIN"),
    reason="needs a built singd binary (SINGULARITY_SINGD_BIN)",
)
def test_singd_end_to_end_over_grpc():
    singd = os.environ["SINGULARITY_SINGD_BIN"]
    port = _free_port()
    addr = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        [singd, "--serve", addr],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with SingdClient(addr, timeout=15.0) as client:
            client.wait_ready(timeout=30.0)

            status = client.register_session(
                "e2e-session",
                model_name="test-model",
                max_context=1024,
                block_size=16,
                dtype="fp16",
                oracle_enabled=True,
                target_nodes=["gpu-node-1"],
            )
            assert status.status == "active"
            assert status.pages_allocated == 64  # 1024 / 16
            assert status.oracle_active

            # Feed the oracle one real attention matrix.
            rng = np.random.default_rng(7)
            attn = rng.random((8, 64), dtype=np.float32)
            ack = client.report_attention_heads("e2e-session", 0, attn)
            assert ack.ingested

            # A freshly allocated block resolves locally.
            loc = client.resolve_page_fault("e2e-session", 3)
            assert loc.is_local
            assert loc.page_size > 0

            # Prefetch runs against the page table (nothing remote yet,
            # so it may stage zero blocks — the call itself must work).
            pre = client.prefetch_blocks(
                "e2e-session", target_tokens=1024, include_anchors=True
            )
            assert pre.blocks_prefetched >= 0

            fs = client.fabric_status()
            assert fs.active_sessions == 1
            assert fs.total_pages == 64
            assert fs.oracle_observations >= 1
            assert not fs.nerve_online  # test environment has no RDMA HW

            out = client.deregister_session("e2e-session")
            assert out.status == "deregistered"

            with pytest.raises(grpc.RpcError) as err:
                client.resolve_page_fault("e2e-session", 0)
            assert err.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Give the listener a moment to release the port.
        time.sleep(0.1)
