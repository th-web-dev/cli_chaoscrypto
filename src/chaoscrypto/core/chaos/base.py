from __future__ import annotations

from abc import ABC, abstractmethod


class ChaoticSystem(ABC):
    """Base class for chaotic systems."""

    def __init__(self, x: float, y: float, z: float, dt: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.dt = float(dt)

    @abstractmethod
    def step(self) -> tuple[float, float, float]:
        """Advance one step and return the new state."""
        ...
