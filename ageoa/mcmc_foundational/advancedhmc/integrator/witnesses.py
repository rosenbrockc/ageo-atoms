"""Auto-generated ghost witness functions for abstract simulation."""

from __future__ import annotations



try:
    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar
except ImportError:
    pass

def witness_temperingfactorcomputation(lf: AbstractArray, r: AbstractArray, step: AbstractArray, n_steps: AbstractArray) -> AbstractArray:
    """Ghost witness for TemperingFactorComputation."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",
    )
    return result

def witness_hamiltonianphasepointtransition(lf: AbstractArray, h: AbstractArray, z: AbstractArray, tempering_scale: AbstractArray) -> AbstractArray:
    """Ghost witness for HamiltonianPhasepointTransition."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",
    )
    return result
