import os

import numpy as np
import icontract
from typing import Sequence, Union

from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import witness_fft, witness_ifft, witness_rfft, witness_irfft

# Types
ArrayLike = Union[np.ndarray, list, tuple]

_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"


def _roundtrip_close(original: np.ndarray, reconstructed: np.ndarray, atol: float = 1e-10) -> bool:
    """Check that a round-trip reconstruction is close to the original."""
    return bool(np.allclose(original, reconstructed, atol=atol))


@register_atom(witness_fft)
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result, a, n: result.shape[-1] == (n if n is not None else np.asarray(a).shape[-1]), "Result shape must match n or input shape")
@icontract.ensure(lambda result: np.iscomplexobj(result), "FFT output must be complex-valued")
@icontract.ensure(
    lambda result, a, n, axis, norm: _roundtrip_close(
        np.asarray(a) if n is None else np.asarray(a),
        np.fft.ifft(result, n=n, axis=axis, norm=norm),
    ),
    "Round-trip IFFT(FFT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def fft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional n-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [CT].

    Args:
        a: Input array, can be complex.
        n: Length of the transformed axis of the output. If n is smaller
            than the length of the input, the input is cropped. If it
            is larger, the input is padded with zeros. If n is not
            given, the length of the input along the axis specified by
            axis is used.
        axis: Axis over which to compute the FFT. If not given, the
            last axis is used.
        norm: Normalization mode. Default is None, meaning no
            normalization. Can be "ortho", "forward", or "backward".

    Returns:
        The truncated or zero-padded input, transformed along the
        axis indicated by axis, or the last one if axis is not
        specified.
    
    """
    return np.fft.fft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_ifft)
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(lambda result, a, n: result.shape[-1] == (n if n is not None else np.asarray(a).shape[-1]), "Result shape must match n or input shape")
@icontract.ensure(
    lambda result, a, n, axis, norm: _roundtrip_close(
        np.asarray(a),
        np.fft.fft(result, n=n, axis=axis, norm=norm),
    ),
    "Round-trip FFT(IFFT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def ifft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional n-point
    discrete Fourier Transform computed by fft.

    Args:
        a: Input array, can be complex.
        n: Length of the transformed axis of the output. If n is smaller
            than the length of the input, the input is cropped. If it
            is larger, the input is padded with zeros. If n is not
            given, the length of the input along the axis specified by
            axis is used.
        axis: Axis over which to compute the inverse FFT. If not given,
            the last axis is used.
        norm: Normalization mode. Default is None.

    Returns:
        The truncated or zero-padded input, transformed along the
        axis indicated by axis, or the last one if axis is not
        specified.
    
    """
    return np.fft.ifft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_rfft)
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n: result.shape[-1] == (n // 2 + 1 if n is not None else np.asarray(a).shape[-1] // 2 + 1),
    "Result shape must match n//2+1 or input_shape//2+1",
)
@icontract.ensure(lambda result: np.iscomplexobj(result), "RFFT output must be complex-valued")
def rfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the one-dimensional discrete Fourier Transform for real input.

    This function computes the one-dimensional n-point discrete Fourier
    Transform of a real-valued array using the FFT algorithm. Since the
    input is real, the output is Hermitian-symmetric and only the
    positive-frequency half is returned.

    Args:
        a: Input array, must be real-valued.
        n: Number of points in the transformed axis. If n is smaller
            than the length of the input, the input is cropped. If
            larger, padded with zeros.
        axis: Axis over which to compute the FFT. Default is -1.
        norm: Normalization mode. Default is None.

    Returns:
        The positive-frequency terms of the Fourier transform of the
        real input, with shape n//2+1 along the transformed axis.
    
    """
    return np.fft.rfft(a, n=n, axis=axis, norm=norm)


@register_atom(witness_irfft)
@icontract.require(lambda a: a is not None, "Input array must not be None")
@icontract.require(lambda a: np.asarray(a).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, a, n: result.shape[-1] == (n if n is not None else 2 * (np.asarray(a).shape[-1] - 1)),
    "Result shape must match n or 2*(input_shape-1)",
)
@icontract.ensure(lambda result: np.isrealobj(result), "IRFFT output must be real-valued")
def irfft(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> np.ndarray:
    """Compute the inverse FFT for rfft output.

    This function computes the inverse of the one-dimensional n-point
    discrete Fourier Transform of real input computed by rfft.

    Args:
        a: Input array, the output of rfft (complex, Hermitian-symmetric).
        n: Length of the transformed axis of the output. For n output
            points, n//2+1 input points are necessary. If n is not
            given, it is determined from the length of the input.
        axis: Axis over which to compute the inverse FFT. Default is -1.
        norm: Normalization mode. Default is None.

    Returns:
        The real-valued inverse FFT result.
    
    """
    return np.fft.irfft(a, n=n, axis=axis, norm=norm)


@icontract.require(lambda n: n > 0, "n must be positive")
@icontract.require(lambda d: d > 0, "d must be positive")
@icontract.ensure(lambda result, n: result.shape == (n,), "Result shape must match n")
def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Return the Discrete Fourier Transform sample frequencies.

    The returned float array f contains the frequency bin centers in
    cycles per unit of the sample spacing (with zero at the start).

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate). Default is 1.

    Returns:
        Array of length n containing the sample frequencies.
    
    """
    return np.fft.fftfreq(n, d=d)


@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Result shape must match input shape")
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> np.ndarray:
    """Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that y[0] is the Python convention for y[0,0,0...] and not
    necessarily the corner of the multidimensional array.

    Args:
        x: Input array.
        axes: Axes over which to shift. Default is None, which shifts
            all axes.

    Returns:
        The shifted array.
    
    """
    return np.fft.fftshift(x, axes=axes)
