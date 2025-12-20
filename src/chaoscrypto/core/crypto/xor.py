from __future__ import annotations


def xor_bytes(data: bytes, keystream: bytes) -> bytes:
    """Byte-wise XOR between data and keystream."""
    if len(keystream) < len(data):
        raise ValueError("keystream too short")
    return bytes(d ^ k for d, k in zip(data, keystream))
