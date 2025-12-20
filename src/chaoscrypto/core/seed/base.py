from __future__ import annotations

from __future__ import annotations

from typing import Callable, Dict, List
import numpy as np


class SeedStrategy:
    """Callable seed strategy wrapper."""

    def __init__(self, name: str, func: Callable[[np.ndarray, tuple[int, int]], tuple[float, float, float]]):
        self.name = name
        self.func = func

    def derive_init(self, field: np.ndarray, coord: tuple[int, int]) -> tuple[float, float, float]:
        return self.func(field, coord)


SEED_REGISTRY: Dict[str, SeedStrategy] = {}


def register_seed_strategy(name: str, func: Callable[[np.ndarray, tuple[int, int]], tuple[float, float, float]]):
    SEED_REGISTRY[name] = SeedStrategy(name=name, func=func)


def get_seed_strategy(name: str) -> SeedStrategy:
    if name not in SEED_REGISTRY:
        raise ValueError(f"Unknown seed strategy '{name}'. Available: {list_seed_strategies()}")
    return SEED_REGISTRY[name]


def list_seed_strategies() -> List[str]:
    return sorted(SEED_REGISTRY.keys())
