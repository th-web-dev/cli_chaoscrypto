from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class MemoryParams:
    """Parameters required to build a deterministic memory field."""

    type: str
    size: int
    scale: float


class MemoryModel(ABC):
    """Interface for deterministic memory models."""

    @abstractmethod
    def generate(self, token_bytes: bytes, params: MemoryParams) -> np.ndarray:
        """Create the deterministic memory field."""

    @staticmethod
    def fingerprint(field: np.ndarray) -> str:
        """SHA-256 fingerprint over the field bytes (row-major)."""
        return hashlib.sha256(field.tobytes(order="C")).hexdigest()
