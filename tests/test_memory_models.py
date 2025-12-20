import numpy as np

from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.core.memory import opensimplex  # noqa: F401
from chaoscrypto.core.memory import perlin  # noqa: F401
from chaoscrypto.core.memory.base import get_memory_model


def test_perlin_determinism():
    params = MemoryParams(type="perlin", size=16, scale=0.1)
    token = b"test-token"
    model = get_memory_model("perlin")
    f1 = model.generate(token, params)
    f2 = model.generate(token, params)
    assert np.array_equal(f1, f2)


def test_memory_type_differs():
    token = b"test-token"
    params = MemoryParams(type="perlin", size=16, scale=0.1)
    perlin_field = get_memory_model("perlin").generate(token, params)
    params_os = MemoryParams(type="opensimplex", size=16, scale=0.1)
    opensimplex_field = get_memory_model("opensimplex").generate(token, params_os)
    assert not np.array_equal(perlin_field, opensimplex_field)
