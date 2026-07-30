import numpy as np
import pytest


@pytest.fixture
def float32_cells() -> list[np.ndarray]:
    """Cells of shape (n, 2) float32 covering empty, small, and larger n."""
    rng = np.random.default_rng(42)
    return [rng.random((n, 2)).astype("<f4") for n in (3, 0, 7, 1, 250, 0)]


@pytest.fixture
def uint64_cells() -> list[np.ndarray]:
    """Cells of shape (n,) uint64."""
    rng = np.random.default_rng(7)
    return [
        rng.integers(0, 2**63, size=n, dtype=np.uint64).astype("<u8") for n in (5, 0, 1, 12, 0, 3)
    ]


def as_object_array(cells: list) -> np.ndarray:
    out = np.empty(len(cells), dtype=object)
    out[:] = cells
    return out
