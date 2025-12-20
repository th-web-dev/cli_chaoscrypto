from __future__ import annotations

from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    """Interface to extract a byte from a chaotic state."""

    @abstractmethod
    def sample(self, state: tuple[float, float, float]) -> int:
        ...
