import pytest

from lamas.backends import MockBackend


@pytest.fixture
def mock():
    """A simulated backend: real ordering of events, ~50x faster than a real model."""
    return MockBackend(seed=7, time_scale=0.02, error_rate=0.0)
