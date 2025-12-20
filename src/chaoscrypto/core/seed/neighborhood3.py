from __future__ import annotations

import numpy as np

from .base import SeedStrategy


class Neighborhood3Seed(SeedStrategy):
    """
    Derive initial values from a 3-cell neighborhood with modulo indexing.

    (x0, y0, z0) = (field[x, y], field[x+1, y], field[x, y+1])
    """

    def derive_init(
        self, field: np.ndarray, coord: tuple[int, int]
    ) -> tuple[float, float, float]:
        x, y = coord
        size = field.shape[0]
        x_idx = x % size
        y_idx = y % size
        x0 = float(field[x_idx, y_idx])
        y0 = float(field[(x_idx + 1) % size, y_idx])
        z0 = float(field[x_idx, (y_idx + 1) % size])
        return x0, y0, z0
