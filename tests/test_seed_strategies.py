import numpy as np

from chaoscrypto.core.seed.base import get_seed_strategy, list_seed_strategies
from chaoscrypto.core.seed import strategies  # noqa: F401 (registers)


def test_seed_strategies_deterministic():
    field = (np.arange(64, dtype=np.float64) ** 2).reshape(8, 8)
    coord = (3, 4)
    strat1 = get_seed_strategy("neighborhood3")
    strat2 = get_seed_strategy("window_mean_3x3")

    a1 = strat1.derive_init(field, coord)
    a2 = strat1.derive_init(field, coord)
    b1 = strat2.derive_init(field, coord)
    b2 = strat2.derive_init(field, coord)

    assert a1 == a2
    assert b1 == b2
    assert a1 != b1  # likely different for this field/coord


def test_seed_registry_lists_expected():
    available = list_seed_strategies()
    assert "neighborhood3" in available
    assert "window_mean_3x3" in available
