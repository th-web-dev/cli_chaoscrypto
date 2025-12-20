from __future__ import annotations

import hashlib
import numpy as np

from .base import MemoryParams, register_memory_model


def _xorshift64(seed: int) -> int:
    x = seed & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def _perm_table(seed: int) -> np.ndarray:
    nums = list(range(256))
    # Fisher-Yates using xorshift RNG for determinism across Python versions
    rng_state = seed or 1
    for i in range(255, 0, -1):
        rng_state = _xorshift64(rng_state)
        j = rng_state % (i + 1)
        nums[i], nums[j] = nums[j], nums[i]
    p = np.array(nums + nums, dtype=np.int32)
    return p


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(h: int, x: float, y: float) -> float:
    # 8 directions
    h = h & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _perlin_point(x: float, y: float, p: np.ndarray) -> float:
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = p[p[xi] + yi]
    ab = p[p[xi] + yi + 1]
    ba = p[p[xi + 1] + yi]
    bb = p[p[xi + 1] + yi + 1]

    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)


def _generate_perlin(token_bytes: bytes, params: MemoryParams) -> np.ndarray:
    size = params.size
    seed = int.from_bytes(hashlib.sha256(token_bytes).digest()[:8], "big")
    p = _perm_table(seed)

    field = np.empty((size, size), dtype=np.float64, order="C")
    for i in range(size):
        for j in range(size):
            field[i, j] = _perlin_point(i * params.scale, j * params.scale, p)
    return field


register_memory_model("perlin", _generate_perlin)
