from __future__ import annotations

import numpy as np

from .base import register_seed_strategy


def _mod_idx(idx: int, size: int) -> int:
    return idx % size


def neighborhood3(field: np.ndarray, coord: tuple[int, int]) -> tuple[float, float, float]:
    x, y = coord
    size = field.shape[0]
    x_idx = _mod_idx(x, size)
    y_idx = _mod_idx(y, size)
    x0 = float(field[x_idx, y_idx])
    y0 = float(field[_mod_idx(x_idx + 1, size), y_idx])
    z0 = float(field[x_idx, _mod_idx(y_idx + 1, size)])
    return x0, y0, z0


def window_mean(field: np.ndarray, center: tuple[int, int]) -> float:
    size = field.shape[0]
    x, y = center
    xs = [(_mod_idx(x + dx, size)) for dx in (-1, 0, 1)]
    ys = [(_mod_idx(y + dy, size)) for dy in (-1, 0, 1)]
    window = field[np.ix_(xs, ys)]
    return float(window.mean())


def window_mean_3x3(field: np.ndarray, coord: tuple[int, int]) -> tuple[float, float, float]:
    x, y = coord
    size = field.shape[0]
    x0 = window_mean(field, (x, y))
    y0 = window_mean(field, (x + 1, y))
    z0 = window_mean(field, (x, y + 1))
    return x0, y0, z0


# Registry
register_seed_strategy("neighborhood3", neighborhood3)
register_seed_strategy("window_mean_3x3", window_mean_3x3)
