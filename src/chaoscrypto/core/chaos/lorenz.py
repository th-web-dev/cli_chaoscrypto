from __future__ import annotations

from .base import ChaoticSystem


class LorenzSystem(ChaoticSystem):
    """Explicit Euler integration of the Lorenz system."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        dt: float,
        sigma: float,
        rho: float,
        beta: float,
    ):
        super().__init__(x, y, z, dt)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def step(self) -> tuple[float, float, float]:
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        return self.x, self.y, self.z
