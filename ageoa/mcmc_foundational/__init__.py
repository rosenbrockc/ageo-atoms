from __future__ import annotations

import importlib

def _maybe_import(submodule: str) -> None:
    try:
        globals()[submodule] = importlib.import_module(f"{__name__}.{submodule}")
    except Exception:
        globals()[submodule] = None

for _name in (
    "advancedhmc",
    "kthohr_mcmc",
    "mini_mcmc",
):
    _maybe_import(_name)

__all__: list[str] = []
