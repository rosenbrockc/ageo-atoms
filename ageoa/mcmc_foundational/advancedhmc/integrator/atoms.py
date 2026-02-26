"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from juliacall import Main as jl  # type: ignore[import-untyped]


from .witnesses import witness_temperingfactorcomputation, witness_hamiltonianphasepointtransition
@register_atom(witness_temperingfactorcomputation)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda r: r is not None, "r cannot be None")
@icontract.require(lambda step: step is not None, "step cannot be None")
@icontract.require(lambda n_steps: n_steps is not None, "n_steps cannot be None")
@icontract.ensure(lambda result: result is not None, "TemperingFactorComputation output must not be None")
def temperingfactorcomputation(lf: np.ndarray, r: np.ndarray, step: int, n_steps: int) -> float:
    """Computes a deterministic tempering multiplier across sub-steps (with bounds checking) to scale the transition strength.

    Args:
        lf: Read-only; no persistent mutation
        r: Finite
        step: 0 <= step <= n_steps
        n_steps: Positive

    Returns:
        Deterministic function of inputs
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_hamiltonianphasepointtransition)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda tempering_scale: tempering_scale is not None, "tempering_scale cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "HamiltonianPhasepointTransition all outputs must not be None")
def hamiltonianphasepointtransition(lf: np.ndarray, h: np.ndarray, z: np.ndarray, tempering_scale: float) -> tuple[np.ndarray, bool]:
    """Execute one pure Hamiltonian transition kernel step by computing derivatives, applying step-size/tempering, and returning a new phase-point state.

    Args:
        lf: Read-only; no persistent mutation.
        h: Immutable input state
        z: Finite where required
        tempering_scale: Provided by tempering computation

    Returns:
        h_next: New immutable state object (state_out)
        is_valid: True iff finite/valid transition
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

# duplicate future import removed

from juliacall import Main as jl  # type: ignore[import-untyped]
# removed duplicate future import (already declared at top of file)

def _temperingfactorcomputation_ffi(lf: np.ndarray, r: np.ndarray, step: int, n_steps: int) -> float:
    """FFI bridge to Julia implementation of TemperingFactorComputation."""
    return jl.eval("temperingfactorcomputation(lf, r, step, n_steps)")

def _hamiltonianphasepointtransition_ffi(lf: np.ndarray, h: np.ndarray, z: np.ndarray, tempering_scale: float) -> tuple[np.ndarray, bool]:
    """FFI bridge to Julia implementation of HamiltonianPhasepointTransition."""
    return jl.eval("hamiltonianphasepointtransition(lf, h, z, tempering_scale)")