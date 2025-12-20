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
    SEED_STRATEGY,
)
from chaoscrypto.core.crypto.xor import xor_bytes
from chaoscrypto.core.memory.base import MemoryModel, MemoryParams, get_memory_model
from chaoscrypto.core.memory import opensimplex  # noqa: F401
from chaoscrypto.core.memory import perlin  # noqa: F401
from chaoscrypto.core.sampling.quantize_byte import QuantizeByteSampling
from chaoscrypto.core.seed.base import get_seed_strategy
from chaoscrypto.core.seed import strategies  # noqa: F401 (registers strategies)


def build_memory_field(
    token_bytes: bytes, params: MemoryParams
) -> tuple[np.ndarray, str]:
    model: MemoryModel = get_memory_model(params.type)
    field = model.generate(token_bytes, params)
    fingerprint = MemoryModel.fingerprint(field)
    return field, fingerprint


def derive_initial_state(field: np.ndarray, coord: tuple[int, int], seed_strategy: str) -> tuple[float, float, float]:
    strategy = get_seed_strategy(seed_strategy)
    return strategy.derive_init(field, coord)


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
    seed_strategy: str,
    dt: float = DEFAULT_DT,
    warmup: int = DEFAULT_WARMUP,
    quant_k: float = DEFAULT_QUANT_K,
) -> tuple[bytes, str]:
    field, field_fp = build_memory_field(token_bytes, params)
    init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
    keystream = generate_keystream(len(plaintext), init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    ciphertext = xor_bytes(plaintext, keystream)
    return ciphertext, field_fp


def decrypt_bytes(
    ciphertext: bytes,
    token_bytes: bytes,
    coord: tuple[int, int],
    params: MemoryParams,
    expected_fingerprint: str | None = None,
    seed_strategy: str = SEED_STRATEGY,
    dt: float = DEFAULT_DT,
    warmup: int = DEFAULT_WARMUP,
    quant_k: float = DEFAULT_QUANT_K,
) -> bytes:
    field, field_fp = build_memory_field(token_bytes, params)
    if expected_fingerprint and field_fp != expected_fingerprint:
        raise ValueError("Field fingerprint mismatch – token or parameters differ.")
    init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
    keystream = generate_keystream(len(ciphertext), init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    return xor_bytes(ciphertext, keystream)
