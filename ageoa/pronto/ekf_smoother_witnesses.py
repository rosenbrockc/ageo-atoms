from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray


def witness_stateestimatorinit() -> AbstractArray:
    """Shape witness for the initialized state vector."""
    return AbstractArray(shape=(6,), dtype="float64")
