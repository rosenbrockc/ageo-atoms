from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

from typing import Any

import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_dm_candidate_filter

def witness_dm_candidate_filter(*args, **kwargs): pass  # replaced at runtime by the generated witnesses module

@register_atom(witness_dm_candidate_filter)  # type: ignore[misc]
@icontract.require(lambda sens: isinstance(sens, (float, int, np.number)), "sens must be numeric")
@icontract.require(lambda DM_base: isinstance(DM_base, (float, int, np.number)), "DM_base must be numeric")
@icontract.require(lambda fchan: isinstance(fchan, (float, int, np.number)), "fchan must be numeric")
@icontract.require(lambda width: isinstance(width, (float, int, np.number)), "width must be numeric")
@icontract.require(lambda tsamp: isinstance(tsamp, (float, int, np.number)), "tsamp must be numeric")
def dm_candidate_filter(data: Any, data_base: Any, sens: float, DM_base: float, candidates: Any, fchan: Any, width: float, tsamp: float) -> Any:
    """Evaluates and filters dispersion measure (DM) candidates by comparing observed data against a base DM model, using sensitivity, channel frequency, bandwidth, and time sampling parameters to identify viable DM candidates.

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
