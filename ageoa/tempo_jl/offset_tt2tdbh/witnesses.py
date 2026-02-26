"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations


try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_offset_tt2tdbh(seconds: AbstractScalar) -> AbstractScalar:
    """Ghost witness for offset_tt2tdbh."""
    return AbstractScalar(dtype="float64", max_val=0.0)
