from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_templatefeaturecomputation
from ppg_sqa import template_matching_features

# Witness functions should be imported from the generated witnesses module
def witness_templatefeaturecomputation(*args, **kwargs): pass
@register_atom(witness_templatefeaturecomputation)  # type: ignore[untyped-decorator]
@icontract.require(lambda hc: hc is not None, "hc cannot be None")
@icontract.ensure(lambda result: result is not None, "TemplateFeatureComputation output must not be None")
def templatefeaturecomputation(hc: object) -> object:
    """Computes template-matching features from the provided input without persistent state mutation.

    Args:
        hc: Required input context for feature computation.

    Returns:
        Derived deterministically from hc.
    """
    return template_matching_features(hc=hc)
