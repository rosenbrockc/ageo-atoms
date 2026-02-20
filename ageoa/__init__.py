"""Top-level package exports for ageoa.

Core modules are imported eagerly. Domain-specific stacks with heavy optional
dependencies are imported lazily so `import ageoa` stays usable in minimal
environments.
"""

from __future__ import annotations

import importlib

from . import biosppy
from . import ghost
from . import numpy
from . import scipy


def _maybe_import(submodule: str) -> None:
    try:
        globals()[submodule] = importlib.import_module(f"{__name__}.{submodule}")
    except Exception:
        # Optional module: missing dependency or environment constraint.
        pass


for _name in ("pasqal", "pulsar", "mint", "alphafold", "e2e_ppg", "quant_engine"):
    _maybe_import(_name)


__all__ = [
    "numpy",
    "scipy",
    "ghost",
    "biosppy",
    "pasqal",
    "pulsar",
    "mint",
    "alphafold",
    "e2e_ppg",
    "quant_engine",
]
