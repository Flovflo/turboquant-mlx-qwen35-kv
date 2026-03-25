import mlx.core as mx

from turboquant_mlx.packing import pack_sign_bits, unpack_sign_bits


def test_pack_roundtrip():
    signs = mx.array([[[[1, 0, 1, 1, 0, 0, 1, 0, 1, 1]]]], dtype=mx.uint8)
    packed = pack_sign_bits(signs)
    unpacked = unpack_sign_bits(packed, 10)
    assert unpacked.shape[-1] == 10
    assert mx.array_equal((unpacked > 0).astype(mx.uint8), signs)
