from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_dm_candidate_filter

@register_atom(witness_dm_candidate_filter)  # type: ignore[misc]
@icontract.require(lambda sens: isinstance(sens, (float, int, np.number)), "sens must be numeric")
@icontract.require(lambda DM_base: isinstance(DM_base, (float, int, np.number)), "DM_base must be numeric")
@icontract.require(lambda fchan: isinstance(fchan, (float, int, np.number)), "fchan must be numeric")
@icontract.require(lambda width: isinstance(width, (float, int, np.number)), "width must be numeric")
@icontract.require(lambda tsamp: isinstance(tsamp, (float, int, np.number)), "tsamp must be numeric")
@icontract.ensure(lambda result, **kwargs: result is not None, "dm_candidate_filter output must not be None")
def dm_candidate_filter(data: np.ndarray, data_base: np.ndarray, sens: float, DM_base: float, candidates: np.ndarray, fchan: float, width: float, tsamp: float) -> np.ndarray:
    """Filters Dispersion Measure (DM) candidates for pulsar detection. Compares observed data against a base DM model using sensitivity and channel parameters to keep only viable candidates.

    Args:
        data: Input data.
        data_base: Input data.
        sens: sens > 0
        DM_base: DM_base >= 0
        candidates: Input data.
        fchan: fchan > 0
        width: width > 0
        tsamp: tsamp > 0

    Returns:
        subset of input candidates
    """
    raise NotImplementedError("Wire to original implementation")
