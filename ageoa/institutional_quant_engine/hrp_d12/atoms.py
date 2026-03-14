from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np
import pandas as pd

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_hrppipelinerun


@register_atom(witness_hrppipelinerun)
@icontract.require(lambda data: isinstance(data, pd.DataFrame), "data must be a pandas DataFrame")
@icontract.require(lambda data: data.shape[1] >= 2, "data must have at least 2 asset columns")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "hrppipelinerun must return a numpy array")
def hrppipelinerun(data: pd.DataFrame) -> np.ndarray:
    """Executes the full Hierarchical Risk Parity pipeline: ingests asset return data, constructs a hierarchical clustering structure via a correlation/distance matrix, applies recursive bisection to allocate risk, and emits final portfolio weights.

    Args:
        data: asset returns DataFrame; no NaN values; N >= 2 columns; T > N rows recommended for stable covariance estimation

    Returns:
        Linkage matrix or portfolio weight array; all weights in [0, 1]; sum == 1.0
    """
    raise NotImplementedError("Wire to original implementation")
