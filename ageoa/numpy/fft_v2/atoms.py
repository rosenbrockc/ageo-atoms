"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations
from typing import Any

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import (
    witness_forwardmultidimensionalfft,
    witness_inversemultidimensionalfft,
    witness_hermitianspectraltransform,
)

@register_atom(witness_forwardmultidimensionalfft)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda norm: norm is not None, "norm cannot be None")
@icontract.ensure(lambda result: result is not None, "ForwardMultidimensionalFFT output must not be None")
def forwardmultidimensionalfft(a: np.ndarray, s: list[int] | None, axes: list[int] | None, norm: str | None) -> np.ndarray:
    """Computes the forward N-dimensional Fast Fourier Transform (FFT) over specified axes. Returns a complex-valued frequency-domain array representing the full spectrum of the input signal.

    Args:
        a: spatial or time-domain input array
        s: output shape along each transformed axis; None preserves input size
        axes: axes over which the FFT is computed; None applies to all axes
        norm: normalization mode - None (backward), 'ortho', or 'forward'

    Returns:
        full complex frequency-domain representation
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_inversemultidimensionalfft)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda s: s is not None, "s cannot be None")
@icontract.require(lambda norm: norm is not None, "norm cannot be None")
@icontract.ensure(lambda result: result is not None, "InverseMultidimensionalFFT output must not be None")
def inversemultidimensionalfft(a: np.ndarray, s: list[int] | None, axes: list[int] | None, norm: str | None) -> np.ndarray:
    """Computes the inverse N-dimensional Fast Fourier Transform (FFT), reconstructing a spatial or time-domain signal from its frequency spectrum.

    Args:
        a: frequency-domain input spectrum
        s: output shape along each transformed axis; None preserves input size
        axes: axes over which the inverse FFT is computed; None applies to all axes
        norm: normalization mode - None (backward), 'ortho', or 'forward'

    Returns:
        reconstructed spatial or time-domain array; imaginary part negligible for real-valued originals
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_hermitianspectraltransform)
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda n: n is not None, "n cannot be None")
@icontract.require(lambda norm: norm is not None, "norm cannot be None")
@icontract.ensure(lambda result: result is not None, "HermitianSpectralTransform output must not be None")
def hermitianspectraltransform(a: np.ndarray, n: int | None, axis: int, norm: str | None) -> np.ndarray:
    """Handles FFTs that exploit or enforce Hermitian symmetry on a single axis. hfft assumes a Hermitian-symmetric complex input (conjugate-even spectrum) and produces a real-valued output array of configurable length. ihfft is its inverse: it takes a real-valued array and returns the complex Hermitian-symmetric result (equivalent to the conjugate of rfft output), suitable for round-trip pipelines involving real signals.

    Args:
        a: Hermitian-symmetric complex signal for hfft; real-valued spectrum for ihfft
        n: desired output length along the transformed axis; None infers from input
        axis: single axis over which the transform is applied; default -1
        norm: normalization mode - None (backward), 'ortho', or 'forward'

    Returns:
        real-valued output spectrum for hfft; Hermitian-symmetric complex array for ihfft
    """
    raise NotImplementedError("Wire to original implementation")
