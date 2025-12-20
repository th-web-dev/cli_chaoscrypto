from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Dict, List
import numpy as np


@dataclass(frozen=True)
class MemoryParams:
    """Parameters required to build a deterministic memory field."""

    type: str
    size: int
    scale: float


class MemoryModel:
    """Callable memory model wrapper."""

    def __init__(self, name: str, func: Callable[[bytes, MemoryParams], np.ndarray]):
        self.name = name
        self.func = func

    def generate(self, token_bytes: bytes, params: MemoryParams) -> np.ndarray:
        return self.func(token_bytes, params)

    @staticmethod
    def fingerprint(field: np.ndarray) -> str:
        """SHA-256 fingerprint over the field bytes (row-major)."""
        return hashlib.sha256(field.tobytes(order="C")).hexdigest()


MEMORY_REGISTRY: Dict[str, MemoryModel] = {}


def register_memory_model(name: str, func: Callable[[bytes, MemoryParams], np.ndarray]):
    MEMORY_REGISTRY[name] = MemoryModel(name=name, func=func)


def get_memory_model(name: str) -> MemoryModel:
    if name not in MEMORY_REGISTRY:
        raise ValueError(f"Unknown memory model '{name}'. Available: {list_memory_models()}")
    return MEMORY_REGISTRY[name]


def list_memory_models() -> List[str]:
    return sorted(MEMORY_REGISTRY.keys())
