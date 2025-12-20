from __future__ import annotations

from typing import Tuple

import numpy as np

from chaoscrypto.core.chaos.lorenz import LorenzSystem
from chaoscrypto.core.constants import (
    DEFAULT_DT,
    DEFAULT_QUANT_K,
    DEFAULT_WARMUP,
    LCL_BETA,
    LCL_RHO,
    LCL_SIGMA,
)
from chaoscrypto.core.crypto.xor import xor_bytes
from chaoscrypto.core.memory.base import MemoryModel, MemoryParams
from chaoscrypto.core.memory.opensimplex import OpenSimplexMemory
from chaoscrypto.core.sampling.quantize_byte import QuantizeByteSampling
from chaoscrypto.core.seed.neighborhood3 import Neighborhood3Seed


def build_memory_field(
    token_bytes: bytes, params: MemoryParams
) -> tuple[np.ndarray, str]:
    model: MemoryModel = OpenSimplexMemory()
    field = model.generate(token_bytes, params)
    fingerprint = MemoryModel.fingerprint(field)
    return field, fingerprint


def derive_initial_state(field: np.ndarray, coord: tuple[int, int]) -> tuple[float, float, float]:
    seed_strategy = Neighborhood3Seed()
    return seed_strategy.derive_init(field, coord)


def generate_keystream(
    num_bytes: int,
    init_state: tuple[float, float, float],
    dt: float = DEFAULT_DT,
    warmup: int = DEFAULT_WARMUP,
    quant_k: float = DEFAULT_QUANT_K,
) -> bytes:
    system = LorenzSystem(
        x=init_state[0],
        y=init_state[1],
        z=init_state[2],
        dt=dt,
        sigma=LCL_SIGMA,
        rho=LCL_RHO,
        beta=LCL_BETA,
    )
    sampler = QuantizeByteSampling(k=quant_k)

    for _ in range(warmup):
        system.step()

    keystream = bytearray()
    for _ in range(num_bytes):
        state = system.step()
        keystream.append(sampler.sample(state))
    return bytes(keystream)


def encrypt_bytes(
    plaintext: bytes,
    token_bytes: bytes,
    coord: tuple[int, int],
    params: MemoryParams,
    dt: float = DEFAULT_DT,
    warmup: int = DEFAULT_WARMUP,
    quant_k: float = DEFAULT_QUANT_K,
) -> tuple[bytes, str]:
    field, field_fp = build_memory_field(token_bytes, params)
    init_state = derive_initial_state(field, coord)
    keystream = generate_keystream(len(plaintext), init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    ciphertext = xor_bytes(plaintext, keystream)
    return ciphertext, field_fp


def decrypt_bytes(
    ciphertext: bytes,
    token_bytes: bytes,
    coord: tuple[int, int],
    params: MemoryParams,
    expected_fingerprint: str | None = None,
    dt: float = DEFAULT_DT,
    warmup: int = DEFAULT_WARMUP,
    quant_k: float = DEFAULT_QUANT_K,
) -> bytes:
    field, field_fp = build_memory_field(token_bytes, params)
    if expected_fingerprint and field_fp != expected_fingerprint:
        raise ValueError("Field fingerprint mismatch – token or parameters differ.")
    init_state = derive_initial_state(field, coord)
    keystream = generate_keystream(len(ciphertext), init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    return xor_bytes(ciphertext, keystream)
