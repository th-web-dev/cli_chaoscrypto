from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class SeedStrategy(ABC):
    """Interface for deriving initial states from a memory field and coordinate."""

    @abstractmethod
    def derive_init(
        self, field: np.ndarray, coord: tuple[int, int]
    ) -> tuple[float, float, float]:
        ...
