"""Top-level package exports for ageoa.

Core modules are imported eagerly. Domain-specific stacks with heavy optional
dependencies are imported lazily so `import ageoa` stays usable in minimal
environments.
"""

from __future__ import annotations

from ageoa_julia_runtime import configure_juliacall_env

configure_juliacall_env()

try:
    import juliacall
except ImportError:
    pass

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
        globals()[submodule] = None


for _name in (
    "pasqal",
    "pulsar",
    "pulsar_folding",
    "mint",
    "alphafold",
    "e2e_ppg",
    "quant_engine",
    "rust_robotics",
    "tempo_jl",
    "quantfin",
    "datadriven",
    "pronto",
    "institutional_quant_engine",
    "molecular_docking",
    "mcmc_foundational",
    "advancedvi",
    "jax_advi",
    "kalman_filters",
    "conjugate_priors",
    "particle_filters",
    "bayes_rs",
    "belief_propagation",
    "neurokit2",
    "hftbacktest",
    "jFOF",
    "astroflow",
    "skyfield",
    "hPDB",
    "heartpy",
    "sklearn",
):
    _maybe_import(_name)


__all__ = [
    "numpy",
    "scipy",
    "ghost",
    "biosppy",
    "pasqal",
    "pulsar",
    "pulsar_folding",
    "mint",
    "alphafold",
    "e2e_ppg",
    "quant_engine",
    "rust_robotics",
    "tempo_jl",
    "quantfin",
    "datadriven",
    "pronto",
    "institutional_quant_engine",
    "molecular_docking",
    "mcmc_foundational",
    "advancedvi",
    "jax_advi",
    "kalman_filters",
    "conjugate_priors",
    "particle_filters",
    "bayes_rs",
    "belief_propagation",
    "neurokit2",
    "hftbacktest",
    "jFOF",
    "astroflow",
    "skyfield",
    "hPDB",
    "heartpy",
    "sklearn",
]
