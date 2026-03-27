from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .inverse_schmitt_witnesses import witness_inverse_schmitt_trigger_transform

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_inverse_schmitt_trigger_transform)  # type: ignore[untyped-decorator, name-defined]
@icontract.require(lambda input_signal: input_signal is not None, "input_signal cannot be None")
@icontract.ensure(lambda result: result is not None, "inverse_schmitt_trigger_transform output must not be None")
def inverse_schmitt_trigger_transform(input_signal: np.ndarray | Mapping[str, object]) -> np.ndarray:
    """Entry-point pure transform for inverse Schmitt trigger behavior. No sub-methods, mutable attributes, or config-gated branches were provided, so this is modeled as a single stateless atom.

    Args:
        input_signal: Exact signature not provided; use implementation-defined type/shape.

    Returns:
        Matches implementation-defined output type/shape.
    """
    import numpy as np
    # Inverse Schmitt trigger: output goes high when input drops below
    # low_threshold, goes low when input rises above high_threshold
    if isinstance(input_signal, Mapping):
        signal = np.asarray(input_signal.get("signal", []), dtype=np.float64)
        low_thresh = float(input_signal.get("low_threshold", 0.3))
        high_thresh = float(input_signal.get("high_threshold", 0.7))
        prev_output = bool(input_signal.get("prev_output", False))
    else:
        signal = np.atleast_1d(np.asarray(input_signal, dtype=np.float64))
        low_thresh, high_thresh = 0.3, 0.7
        prev_output = False

    output = np.empty(signal.shape, dtype=np.float64)
    state = prev_output
    for i in range(len(signal)):
        if signal[i] <= low_thresh:
            state = True
        elif signal[i] >= high_thresh:
            state = False
        output[i] = float(state)
    return output


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _inverse_schmitt_trigger_transform_ffi(input_signal: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of inverse schmitt trigger transform. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./inverse_schmitt_trigger_transform.so")
    _func_name = 'inverse_schmitt_trigger_transform_prime'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return ctypes.c_void_p(_func(input_signal))
