"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

from juliacall import Main as jl  # type: ignore[import-untyped]


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_utc_to_tai_leap_second_kernel)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda utc1: isinstance(utc1, (float, int, np.number)), "utc1 must be numeric")
@icontract.require(lambda utc2: isinstance(utc2, (float, int, np.number)), "utc2 must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "utc_to_tai_leap_second_kernel all outputs must not be None")
def utc_to_tai_leap_second_kernel(utc1: float, utc2: float) -> tuple[float, float]:
    """Converts a two-part UTC Julian date to TAI by resolving the applicable leap-second offset. Internally converts Julian date to calendar date (jd2cal) to locate the correct leap-second table entry, then adds the offset to produce the TAI two-part Julian date.

    Args:
        utc1: finite; utc1+utc2 must be within the supported leap-second table range
        utc2: finite; together with utc1 represents a valid UTC epoch

    Returns:
        tai1: tai1 + tai2 = utc1 + utc2 + leap_seconds/86400
        tai2: precision-preserving companion to tai1
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_tai_to_utc_inversion)  # type: ignore[name-defined, untyped-decorator]
@icontract.require(lambda tai1: isinstance(tai1, (float, int, np.number)), "tai1 must be numeric")
@icontract.require(lambda tai2: isinstance(tai2, (float, int, np.number)), "tai2 must be numeric")
@icontract.require(lambda tai_estimate: isinstance(tai_estimate, (float, int, np.number)), "tai_estimate must be numeric")
@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "tai_to_utc_inversion all outputs must not be None")
def tai_to_utc_inversion(tai1: float, tai2: float, tai_estimate: float) -> tuple[float, float, float]:
    """Entry-point atom. Inverts the UTC→TAI mapping to recover UTC from a given TAI epoch. Uses an iterative bracketing strategy: seeds a candidate UTC estimate, calls the utc_to_tai_leap_second_kernel to evaluate the forward mapping, then refines until the residual is within floating-point tolerance.

    Args:
        tai1: finite; must fall within the supported leap-second table range
        tai2: finite; precision-preserving companion to tai1
        tai_estimate: fed from utc_to_tai_leap_second_kernel output on each iteration

    Returns:
        utc1: utc1 + utc2 + leap_seconds/86400 ≈ tai1 + tai2 within floating-point epsilon
        utc2: precision-preserving companion to utc1
        candidate_utc: updated each iteration; becomes final utc1/utc2 on convergence
    """
    raise NotImplementedError("Wire to original implementation")


"""Auto-generated FFI bindings for julia implementations."""

from __future__ import annotations

from juliacall import Main as jl


def utc_to_tai_leap_second_kernel_ffi(utc1: float, utc2: float) -> object:
    """FFI bridge to Julia implementation of utc_to_tai_leap_second_kernel."""
    return jl.eval("utc_to_tai_leap_second_kernel(utc1, utc2)")

def tai_to_utc_inversion_ffi(tai1: float, tai2: float, tai_estimate: float) -> object:
    """FFI bridge to Julia implementation of tai_to_utc_inversion."""
    return jl.eval("tai_to_utc_inversion(tai1, tai2, tai_estimate)")