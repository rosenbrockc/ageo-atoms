from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_computebeatagreementsqi(
    detector_1: AbstractArray,
    detector_2: AbstractArray,
    fs: AbstractScalar,
    mode: AbstractScalar,
    search_window: AbstractScalar,
) -> AbstractScalar:
    """Shape-and-type check for compute beat agreement sqi."""
    return AbstractScalar(dtype="float64")


def witness_computefrequencysqi(
    ecg_signal: AbstractSignal,
    fs: AbstractScalar,
    nseg: AbstractScalar,
    num_spectrum: AbstractArray,
    dem_spectrum: AbstractArray | None,
    mode: AbstractScalar,
) -> AbstractScalar:
    """Shape-and-type check for compute frequency sqi."""
    return AbstractScalar(dtype="float64")


def witness_computekurtosissqi(signal: AbstractArray, fisher: AbstractScalar) -> AbstractScalar:
    """Shape-and-type check for compute kurtosis sqi."""
    return AbstractScalar(dtype="float64")


def witness_assemblezz2018sqi(
    b_sqi: AbstractScalar,
    f_sqi: AbstractScalar,
    k_sqi: AbstractScalar,
) -> dict[str, AbstractScalar]:
    """Shape-and-type check for assemble zz2018 sqi."""
    return {
        "b_sqi": AbstractScalar(dtype="float64"),
        "f_sqi": AbstractScalar(dtype="float64"),
        "k_sqi": AbstractScalar(dtype="float64"),
    }
