from __future__ import annotations

import hashlib
import numpy as np
from opensimplex import OpenSimplex

from .base import MemoryModel, MemoryParams


class OpenSimplexMemory(MemoryModel):
    """
    Deterministic 2D OpenSimplex noise field.

    Seed is derived from token + params, so identical inputs yield identical fields.
    """

    def generate(self, token_bytes: bytes, params: MemoryParams) -> np.ndarray:
        size = params.size
        seed_material = (
            token_bytes
            + str(size).encode("utf-8")
            + repr(params.scale).encode("utf-8")
        )
        seed_int = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False
        )
        noise = OpenSimplex(seed=seed_int)

        field = np.empty((size, size), dtype=np.float64, order="C")
        for x in range(size):
            for y in range(size):
                field[x, y] = noise.noise2(x * params.scale, y * params.scale)
        return field
