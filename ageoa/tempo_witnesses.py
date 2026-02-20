"""Ghost witnesses for Tempo atoms."""

from __future__ import annotations

from typing import Any

from ageoa.ghost.abstract import AbstractArray, AbstractScalar


def _offset_output(seconds: Any) -> AbstractArray | AbstractScalar:
    if isinstance(seconds, AbstractArray):
        return AbstractArray(shape=seconds.shape, dtype="float64")
    if isinstance(seconds, AbstractScalar):
        return AbstractScalar(dtype="float64")
    raise ValueError("Tempo witnesses require AbstractArray or AbstractScalar input")


def witness_offset_tt2tdb(seconds: Any) -> AbstractArray | AbstractScalar:
    """Time-coordinate offset preserves structural shape while converting to float64."""
    return _offset_output(seconds)


def witness_offset_tai2tdb(seconds: Any) -> AbstractArray | AbstractScalar:
    """Composite time-coordinate offset preserves structural shape while converting to float64."""
    return _offset_output(seconds)
