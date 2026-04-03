from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .template_matching_witnesses import witness_templatefeaturecomputation

_E2E_PPG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "E2E-PPG"


def _load_ppg_sqa_module():
    if str(_E2E_PPG_ROOT) not in sys.path:
        sys.path.insert(0, str(_E2E_PPG_ROOT))
    return importlib.import_module("ppg_sqa")

@register_atom(witness_templatefeaturecomputation)  # type: ignore[untyped-decorator]
@icontract.require(lambda hc: hc is not None, "hc cannot be None")
@icontract.ensure(lambda result: result is not None, "TemplateFeatureComputation output must not be None")
def templatefeaturecomputation(hc: Sequence[Sequence[float]] | np.ndarray) -> tuple[float, float]:
    """Computes template-matching features from the provided input without persistent state mutation.

    Args:
        hc: Required input context for feature computation.

    Returns:
        Derived deterministically from hc.
    """
    normalized_hc = [np.asarray(beat, dtype=float) for beat in hc]
    tm_ave_eu, tm_ave_corr = _load_ppg_sqa_module().template_matching_features(hc=normalized_hc)
    return (float(tm_ave_eu), float(tm_ave_corr))
