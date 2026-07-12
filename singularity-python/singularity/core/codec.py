"""
singularity.core.codec
======================
KV-block compression codecs.

Moving KV-cache over a fabric is bandwidth-bound, so compression directly buys
latency. This module implements two *real*, lossless-to-configurable codecs:

  * RawCodec        -- identity (baseline).
  * Int8QuantCodec  -- per-(block,head) symmetric int8 quantization. Stores one
                       fp32 scale per head and int8 payloads. ~2x smaller than
                       fp16, ~4x smaller than fp32, with bounded error. This is
                       the same family of technique used by production KV-cache
                       quantization (e.g. KV-cache int8).

Quantization error is measured and exposed so callers can trade accuracy for
bandwidth explicitly -- no hidden lossiness.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class EncodedTensor:
    codec: str
    payload: bytes
    shape: tuple
    dtype: str
    scale: np.ndarray | None = None

    def nbytes(self) -> int:
        n = len(self.payload)
        if self.scale is not None:
            n += self.scale.nbytes
        return n


class Codec(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def encode(self, arr: np.ndarray) -> EncodedTensor: ...

    @abc.abstractmethod
    def decode(self, enc: EncodedTensor) -> np.ndarray: ...


class RawCodec(Codec):
    name = "raw"

    def encode(self, arr: np.ndarray) -> EncodedTensor:
        a = np.ascontiguousarray(arr)
        return EncodedTensor(self.name, a.tobytes(), a.shape, str(a.dtype))

    def decode(self, enc: EncodedTensor) -> np.ndarray:
        return np.frombuffer(enc.payload, dtype=enc.dtype).reshape(enc.shape)


class Int8QuantCodec(Codec):
    """Per-head symmetric int8 quantization.

    Input shape is (num_tokens, num_heads, head_dim). A separate scale is
    computed per head so heads with different dynamic range keep precision.
    """

    name = "int8-perhead"

    def encode(self, arr: np.ndarray) -> EncodedTensor:
        a = np.ascontiguousarray(arr).astype(np.float32)
        # scale per head: max abs over (tokens, head_dim)
        amax = np.maximum(np.abs(a).max(axis=(0, 2)), 1e-8)  # (num_heads,)
        scale = (amax / 127.0).astype(np.float32)
        q = (np.round(a / scale[None, :, None]).clip(-127, 127)
             .astype(np.int8))
        return EncodedTensor(self.name, q.tobytes(), a.shape, "int8", scale)

    def decode(self, enc: EncodedTensor) -> np.ndarray:
        q = np.frombuffer(enc.payload, dtype=np.int8).reshape(
            enc.shape).astype(np.float32)
        deq = q * enc.scale[None, :, None]
        return deq.astype("float16")


def quantization_error(codec: Codec, arr: np.ndarray) -> float:
    """Relative L2 error introduced by a codec round-trip (0.0 for lossless)."""
    dec = codec.decode(codec.encode(arr))
    num = np.linalg.norm(dec.astype(np.float32) - arr.astype(np.float32))
    den = np.linalg.norm(arr.astype(np.float32)) + 1e-12
    return float(num / den)


CODECS = {c.name: c for c in (RawCodec(), Int8QuantCodec())}
