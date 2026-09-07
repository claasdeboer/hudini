import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed RNG for data generators. Fresh instance per test for isolation."""
    return np.random.default_rng(42)
