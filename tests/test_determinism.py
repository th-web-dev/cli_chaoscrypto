import numpy as np

from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.orchestrator.pipeline import build_memory_field


def test_memory_determinism():
    token = b"test-token"
    params = MemoryParams(
        type=constants.MEMORY_TYPE,
        size=constants.DEFAULT_MEMORY_SIZE,
        scale=constants.DEFAULT_MEMORY_SCALE,
    )

    field1, fp1 = build_memory_field(token, params)
    field2, fp2 = build_memory_field(token, params)

    assert fp1 == fp2
    assert np.array_equal(field1, field2)
