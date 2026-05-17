from __future__ import annotations

from .base import ChaoticSystem


class RosslerSystem(ChaoticSystem):
    """Explicit Euler integration of the Rössler system."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        dt: float,
        a: float,
        b: float,
        c: float,
    ):
        super().__init__(x, y, z, dt)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def step(self) -> tuple[float, float, float]:
        dx = -self.y - self.z
        dy = self.x + self.a * self.y
        dz = self.b + self.z * (self.x - self.c)

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        return self.x, self.y, self.z
