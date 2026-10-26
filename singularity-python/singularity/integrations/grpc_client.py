"""gRPC client for a running singd, the SINGULARITY daemon.

The daemon serves the contract in ``proto/singularity.proto``
(RegisterSession, ResolvePageFault, ReportAttentionHeads,
PrefetchBlocks, GetFabricStatus, DeregisterSession) backed by its
Global Page Table and entropy Oracle. This client is a thin wrapper:
it forwards calls over the wire and returns the daemon's responses.
It does not simulate any of them locally — if no daemon is listening,
calls fail with the gRPC error, which is the correct behavior.

Requires the ``grpc`` extra (``pip install singularity-dkcs[grpc]``).

The protobuf stubs (``singularity_pb2`` / ``singularity_pb2_grpc`` in
this package) are generated from ``proto/singularity.proto`` and are
checked in. Their gencode is 7.35 (grpcio-tools 1.85), which requires
a protobuf 7.x runtime and therefore Python >= 3.10. Regenerate with
the pinned toolchain::

    pip install "grpcio-tools==1.85.0"
    python -m grpc_tools.protoc -I proto \
        --python_out=singularity-python/singularity/integrations \
        --grpc_python_out=singularity-python/singularity/integrations \
        singularity.proto
    # then make the generated import package-relative:
    #   import singularity_pb2 -> from . import singularity_pb2
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

try:
    import grpc

    from . import singularity_pb2 as pb2
    from . import singularity_pb2_grpc as pb2_grpc

    _GRPC_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the extra
    # ImportError (no grpcio) or protobuf's VersionError (runtime older
    # than the stubs' gencode): either way the client is unavailable
    # in this environment, and SingdClient says how to fix it.
    grpc = None  # type: ignore[assignment]
    pb2 = None  # type: ignore[assignment]
    pb2_grpc = None  # type: ignore[assignment]
    _GRPC_AVAILABLE = False


def grpc_available() -> bool:
    return _GRPC_AVAILABLE


def encode_attention_chunk(
    session_id: str,
    layer: int,
    attention: np.ndarray,
) -> "pb2.AttentionChunk":
    """Encode a [num_heads, seq_len] float array as a raw_f32 chunk.

    The byte layout is little-endian float32, row-major — the exact
    layout the daemon decodes in its ReportAttentionHeads handler.
    """
    arr = np.ascontiguousarray(attention, dtype="<f4")
    if arr.ndim != 2:
        raise ValueError("attention must be a 2D [num_heads, seq_len] array")
    return pb2.AttentionChunk(
        session_id=session_id,
        layer=int(layer),
        attention_data=arr.tobytes(),
        seq_len=int(arr.shape[1]),
        num_heads=int(arr.shape[0]),
        encoding="raw_f32",
    )


class SingdClient:
    """Client for one singd endpoint. Use as a context manager or
    call :meth:`close`."""

    def __init__(self, address: str = "127.0.0.1:50051", timeout: float = 10.0):
        if not _GRPC_AVAILABLE:
            raise ImportError(
                "SingdClient requires grpcio: pip install singularity-dkcs[grpc]"
            )
        self.address = address
        self.timeout = timeout
        self._channel = grpc.insecure_channel(address)
        self._stub = pb2_grpc.SingularityServiceStub(self._channel)

    # -- lifecycle ----------------------------------------------------
    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "SingdClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def wait_ready(self, timeout: Optional[float] = None) -> None:
        """Block until the daemon accepts connections (or raise)."""
        grpc.channel_ready_future(self._channel).result(
            timeout=self.timeout if timeout is None else timeout
        )

    # -- RPCs -----------------------------------------------------------
    def register_session(
        self,
        session_id: str,
        model_name: str = "",
        max_context: int = 4096,
        block_size: int = 16,
        dtype: str = "fp16",
        oracle_enabled: bool = True,
        target_nodes: Optional[List[str]] = None,
    ):
        return self._stub.RegisterSession(
            pb2.SessionConfig(
                session_id=session_id,
                model_name=model_name,
                max_context=max_context,
                block_size=block_size,
                dtype=dtype,
                oracle_enabled=oracle_enabled,
                target_nodes=list(target_nodes or []),
            ),
            timeout=self.timeout,
        )

    def resolve_page_fault(
        self,
        session_id: str,
        block_id: int,
        layer: int = 0,
        urgent: bool = False,
    ):
        return self._stub.ResolvePageFault(
            pb2.FaultRequest(
                session_id=session_id,
                block_id=block_id,
                layer=layer,
                urgent=urgent,
            ),
            timeout=self.timeout,
        )

    def report_attention_heads(
        self,
        session_id: str,
        layer: int,
        attention: np.ndarray,
    ):
        """Send one attention matrix ([num_heads, seq_len]) to the Oracle."""
        chunk = encode_attention_chunk(session_id, layer, attention)
        return self._stub.ReportAttentionHeads(
            iter([chunk]), timeout=self.timeout
        )

    def report_attention_stream(
        self,
        chunks: Iterable["pb2.AttentionChunk"],
    ):
        return self._stub.ReportAttentionHeads(
            iter(chunks), timeout=self.timeout
        )

    def prefetch_blocks(
        self,
        session_id: str,
        target_tokens: int = 0,
        include_anchors: bool = True,
    ):
        return self._stub.PrefetchBlocks(
            pb2.PrefetchRequest(
                session_id=session_id,
                target_tokens=target_tokens,
                include_anchors=include_anchors,
            ),
            timeout=self.timeout,
        )

    def fabric_status(self, include_peers: bool = True):
        return self._stub.GetFabricStatus(
            pb2.FabricStatusRequest(include_peers=include_peers),
            timeout=self.timeout,
        )

    def deregister_session(self, session_id: str):
        return self._stub.DeregisterSession(
            pb2.SessionHandle(session_id=session_id),
            timeout=self.timeout,
        )
