from singularity.core.blocks import KVBlock
from singularity.core.codec import Int8QuantCodec, RawCodec, quantization_error


def test_raw_codec_is_lossless():
    b = KVBlock.random("b", 0, 0, 16, seed=1)
    assert quantization_error(RawCodec(), b.key) == 0.0


def test_int8_codec_shrinks_and_bounds_error():
    b = KVBlock.random("b", 0, 0, 64, num_heads=8, head_dim=128, dtype="float16", seed=2)
    codec = Int8QuantCodec()
    enc = codec.encode(b.key)
    raw_bytes = b.key.astype("float16").nbytes
    # int8 payload + tiny per-head scales must be clearly smaller than fp16
    assert enc.nbytes() < raw_bytes
    # symmetric int8 quant of gaussian data: relative L2 error stays small
    err = quantization_error(codec, b.key)
    assert err < 0.05


def test_int8_roundtrip_shape_preserved():
    b = KVBlock.random("b", 0, 0, 8, seed=3)
    codec = Int8QuantCodec()
    dec = codec.decode(codec.encode(b.key))
    assert dec.shape == b.key.shape
