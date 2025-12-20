from __future__ import annotations

import math

from .base import SamplingStrategy


class QuantizeByteSampling(SamplingStrategy):
    """Quantize the absolute x component into a byte."""

    def __init__(self, k: float):
        self.k = float(k)

    def sample(self, state: tuple[float, float, float]) -> int:
        x, _, _ = state
        return int(math.floor(abs(x) * self.k) % 256)
