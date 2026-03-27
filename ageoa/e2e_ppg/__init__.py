from .atoms import kazemi_peak_detection, ppg_reconstruction, ppg_sqa

from importlib import import_module

__all__ = [
    "kazemi_peak_detection",
    "ppg_reconstruction",
    "ppg_sqa",
]


def _maybe_export(module_name: str, names: list[str]) -> None:
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        return
    for name in names:
        globals()[name] = getattr(module, name)
    __all__.extend(names)


_maybe_export("ageoa.e2e_ppg.gan_reconstruction", ["generatereconstructedppg", "gan_reconstruction"])
_maybe_export("ageoa.e2e_ppg.heart_cycle", ["detect_heart_cycles", "heart_cycle_detection"])
_maybe_export("ageoa.e2e_ppg.kazemi_wrapper.atoms", ["wrapperpredictionsignalcomputation", "signalarraynormalization"])
_maybe_export("ageoa.e2e_ppg.reconstruction.atoms", ["gan_patch_reconstruction", "windowed_signal_reconstruction"])
_maybe_export("ageoa.e2e_ppg.template_matching", ["templatefeaturecomputation"])
