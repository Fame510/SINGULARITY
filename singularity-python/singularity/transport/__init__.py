"""Pluggable transport backends for the fabric."""
from .base import Transport, TransferResult
from .memory import InMemoryTransport

__all__ = ["Transport", "TransferResult", "InMemoryTransport"]

try:  # RDMA backend is optional / hardware-dependent
    from .rdma import RdmaTransport  # noqa: F401
    __all__.append("RdmaTransport")
except Exception:  # pragma: no cover
    pass
