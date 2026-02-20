import os

import numpy as np
import scipy.signal
import icontract
from typing import Union

from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import (
    witness_butter, witness_cheby1, witness_cheby2, witness_firwin,
    witness_sosfilt, witness_lfilter, witness_freqz,
)

ArrayLike = Union[np.ndarray, list, tuple]

_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"


def _poles_inside_unit_circle(a: np.ndarray) -> bool:
    """Check that all poles (roots of denominator polynomial) lie inside the unit circle."""
    roots = np.roots(a)
    return bool(np.all(np.abs(roots) < 1.0))


def _is_valid_filter_order(n: int) -> bool:
    """Check that filter order is a positive integer."""
    return isinstance(n, (int, np.integer)) and n > 0


def _is_valid_critical_freq(wn: Union[float, ArrayLike], fs: Union[float, None]) -> bool:
    """Check that critical frequency is in valid range."""
    wn_arr = np.atleast_1d(np.asarray(wn, dtype=float))
    if np.any(wn_arr <= 0):
        return False
    if fs is not None:
        nyquist = fs / 2.0
        if np.any(wn_arr >= nyquist):
            return False
    return True


# ---------------------------------------------------------------------------
# Filter design atoms
# ---------------------------------------------------------------------------

@register_atom(witness_butter)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def butter(
    N: int,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital or analog Butterworth filter.

    Returns filter coefficients in transfer function (b, a) form.

    Args:
        N: The order of the filter. Must be a positive integer.
        Wn: Critical frequency or frequencies. For digital filters,
            Wn is normalized to [0, 1] where 1 is the Nyquist
            frequency (unless fs is specified).
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Maximally-Flat Rational Transfer Function Generator",
        "conceptual_transform": "Computes the coefficients of a rational function (ratio of polynomials) that exhibits a maximally flat response within a specified range. It maps a set of constraint parameters (complexity degree and transition thresholds) to a pair of vectors representing the numerator and denominator of a linear transformation.",
        "abstract_inputs": [
            {
                "name": "N",
                "description": "An integer specifying the degree of the transformation (model complexity)."
            },
            {
                "name": "Wn",
                "description": "A scalar or vector defining the transition threshold(s) relative to a reference scale."
            },
            {
                "name": "btype",
                "description": "A categorical identifier for the transformation mode (e.g., lower-bound, upper-bound, or range-bound attenuation)."
            },
            {
                "name": "fs",
                "description": "An optional scale factor to normalize the threshold parameters."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple of two 1D arrays of floats representing the numerator and denominator coefficients of the generated rational function."
            }
        ],
        "algorithmic_properties": [
            "deterministic",
            "parameterized",
            "stability-constrained",
            "linear-system-primitive"
        ],
        "cross_disciplinary_applications": [
            "Isolating low-frequency trends in econometric time-series data.",
            "Preprocessing high-frequency sensor telemetry to remove measurement noise in mechanical systems.",
            "Regulating spectral energy distribution in acoustic signal processing.",
            "Smoothing control-loop feedback signals in autonomous navigation systems."
        ]
    }
    <!-- /conceptual_profile -->
    """
    b, a = scipy.signal.butter(N, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


@register_atom(witness_cheby1)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda rp: rp > 0, "Passband ripple must be positive")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def cheby1(
    N: int,
    rp: float,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital Chebyshev Type I filter.

    Returns filter coefficients with rp decibels of passband ripple.

    Args:
        N: The order of the filter. Must be a positive integer.
        rp: The maximum ripple allowed in the passband, in decibels.
        Wn: Critical frequency or frequencies.
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Equiripple Passband Rational Transfer Function Generator",
        "conceptual_transform": "Computes the coefficients of a rational function that minimizes the maximum error in the passband (equiripple) while achieving a specified roll-off. It maps structural constraints (complexity, ripple tolerance, threshold) to a numerator/denominator pair.",
        "abstract_inputs": [
            {
                "name": "N",
                "description": "An integer specifying the model complexity."
            },
            {
                "name": "rp",
                "description": "A scalar defining the maximum allowable error margin (ripple) within the constrained region."
            },
            {
                "name": "Wn",
                "description": "A scalar or vector defining the transition threshold(s)."
            },
            {
                "name": "btype",
                "description": "A categorical identifier for the transformation mode."
            },
            {
                "name": "fs",
                "description": "An optional scale factor to normalize the threshold parameters."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple of two 1D arrays of floats representing the numerator and denominator coefficients."
            }
        ],
        "algorithmic_properties": [
            "deterministic",
            "parameterized",
            "stability-constrained",
            "minimax-optimization"
        ],
        "cross_disciplinary_applications": [
            "Minimizing passband distortion in communication channel equalization.",
            "Designing precise anti-aliasing pre-filters for analog-to-digital converters.",
            "Smoothing biomedical sensor data while preserving low-amplitude features."
        ]
    }
    <!-- /conceptual_profile -->
    """
    b, a = scipy.signal.cheby1(N, rp, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


@register_atom(witness_cheby2)
@icontract.require(lambda N: _is_valid_filter_order(N), "Filter order must be a positive integer")
@icontract.require(lambda rs: rs > 0, "Stopband attenuation must be positive")
@icontract.require(lambda Wn, fs: _is_valid_critical_freq(Wn, fs), "Critical frequency must be positive (and < Nyquist if fs given)")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (b, a) tuple")
@icontract.ensure(lambda result: result[0].ndim == 1 and result[1].ndim == 1, "b and a must be 1D arrays")
@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def cheby2(
    N: int,
    rs: float,
    Wn: Union[float, ArrayLike],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Design an Nth-order digital Chebyshev Type II filter.

    Returns filter coefficients with rs decibels of stopband attenuation.

    Args:
        N: The order of the filter. Must be a positive integer.
        rs: The minimum attenuation required in the stop band, in dB.
        Wn: Critical frequency or frequencies.
        btype: Type of filter: 'low', 'high', 'band', or 'stop'.
        analog: If True, return an analog filter.
        output: Type of output: 'ba' for transfer function coefficients.
        fs: The sampling frequency of the digital system.

    Returns:
        Tuple of (b, a) numerator and denominator polynomials.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Equiripple Stopband Rational Transfer Function Generator",
        "conceptual_transform": "Computes the coefficients of a rational function that minimizes the maximum error in the stopband (equiripple) while achieving a specified roll-off. It maps structural constraints (complexity, attenuation requirement, threshold) to a numerator/denominator pair.",
        "abstract_inputs": [
            {
                "name": "N",
                "description": "An integer specifying the model complexity."
            },
            {
                "name": "rs",
                "description": "A scalar defining the minimum required suppression margin within the excluded region."
            },
            {
                "name": "Wn",
                "description": "A scalar or vector defining the transition threshold(s)."
            },
            {
                "name": "btype",
                "description": "A categorical identifier for the transformation mode."
            },
            {
                "name": "fs",
                "description": "An optional scale factor to normalize the threshold parameters."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple of two 1D arrays of floats representing the numerator and denominator coefficients."
            }
        ],
        "algorithmic_properties": [
            "deterministic",
            "parameterized",
            "stability-constrained",
            "minimax-optimization"
        ],
        "cross_disciplinary_applications": [
            "Suppressing specific narrow-band interference in radar signal processing.",
            "Isolating high-frequency structural resonances in civil engineering telemetry.",
            "Removing powerline noise from geophysical seismic recordings."
        ]
    }
    <!-- /conceptual_profile -->
    """
    b, a = scipy.signal.cheby2(N, rs, Wn, btype=btype, analog=analog, output=output, fs=fs)
    return b, a


# ---------------------------------------------------------------------------
# FIR design
# ---------------------------------------------------------------------------

@register_atom(witness_firwin)
@icontract.require(lambda numtaps: isinstance(numtaps, (int, np.integer)) and numtaps > 0, "numtaps must be a positive integer")
@icontract.ensure(lambda result, numtaps: result.shape == (numtaps,), "Output shape must equal (numtaps,)")
@icontract.ensure(lambda result: np.isrealobj(result), "FIR coefficients must be real-valued")
def firwin(
    numtaps: int,
    cutoff: Union[float, ArrayLike],
    width: float | None = None,
    window: str = "hamming",
    pass_zero: Union[bool, str] = True,
    scale: bool = True,
    fs: float | None = None,
) -> np.ndarray:
    """Design an FIR filter using the window method.

    Compute the coefficients of a finite impulse response filter using
    the window method.

    Args:
        numtaps: Length of the filter (number of coefficients). Must
            be odd for Types I and II FIR filters.
        cutoff: Cutoff frequency or frequencies.
        width: Approximate width of transition region.
        window: Window function to use. Default is 'hamming'.
        pass_zero: If True, the DC gain is 1. If False, DC gain is 0.
        scale: If True, scale coefficients so the frequency response
            is exactly unity at certain frequencies.
        fs: The sampling frequency of the signal.

    Returns:
        1D array of FIR filter coefficients with length numtaps.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Windowed Finite Convolution Kernel Generator",
        "conceptual_transform": "Computes the coefficients of a finite-length sequence that approximates a desired ideal frequency-domain response using a truncation and tapering (windowing) technique. It maps frequency constraints and window characteristics to a 1D convolution kernel.",
        "abstract_inputs": [
            {
                "name": "numtaps",
                "description": "An integer specifying the length of the generated sequence (kernel size)."
            },
            {
                "name": "cutoff",
                "description": "A scalar or vector defining the transition threshold(s)."
            },
            {
                "name": "width",
                "description": "An optional scalar defining the transition smoothness."
            },
            {
                "name": "window",
                "description": "A string identifying the tapering function used to minimize truncation artifacts."
            },
            {
                "name": "pass_zero",
                "description": "A boolean indicating whether the zero-frequency (DC) component is preserved or suppressed."
            },
            {
                "name": "scale",
                "description": "A boolean indicating whether to normalize the resulting kernel."
            },
            {
                "name": "fs",
                "description": "An optional scale factor to normalize the threshold parameters."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D array of floats representing the generated convolution kernel."
            }
        ],
        "algorithmic_properties": [
            "deterministic",
            "parameterized",
            "finite-support",
            "linear-phase-capable"
        ],
        "cross_disciplinary_applications": [
            "Extracting specific spatial frequency bands in 1D image scanlines.",
            "Computing weighted moving averages for sequential resource-flow monitoring.",
            "Implementing matched filters for pulse detection in digital communications."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.signal.firwin(
        numtaps, cutoff, width=width, window=window,
        pass_zero=pass_zero, scale=scale, fs=fs,
    )


# ---------------------------------------------------------------------------
# Filter application atoms
# ---------------------------------------------------------------------------

@register_atom(witness_sosfilt)
@icontract.require(lambda sos: np.asarray(sos).ndim == 2 and np.asarray(sos).shape[1] == 6, "sos must have shape (n_sections, 6)")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input signal must not be empty")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Output shape must match input shape")
def sosfilt(
    sos: np.ndarray,
    x: ArrayLike,
    axis: int = -1,
    zi: np.ndarray | None = None,
) -> np.ndarray:
    """Filter data along one dimension using cascaded second-order sections.

    Apply a digital filter in second-order sections (SOS) format to the
    input signal.

    Args:
        sos: Array of second-order filter coefficients with shape
            (n_sections, 6). Each row is [b0, b1, b2, a0, a1, a2].
        x: Input signal array.
        axis: The axis of x to which the filter is applied.
        zi: Initial conditions for the filter delays.

    Returns:
        The filtered output with the same shape as x.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Cascaded Biquadratic Sequential Transformer",
        "conceptual_transform": "Applies a series of second-order linear difference equations to a sequential input. By cascading multiple low-order stages, it achieves high-order transformations while minimizing numerical instability and quantization noise.",
        "abstract_inputs": [
            {
                "name": "sos",
                "description": "A 2D array of shape (N, 6) defining the coefficients for N cascaded second-order stages."
            },
            {
                "name": "x",
                "description": "An N-dimensional tensor representing the input sequence."
            },
            {
                "name": "axis",
                "description": "An integer specifying the dimension along which to apply the transformation."
            },
            {
                "name": "zi",
                "description": "An optional array defining the initial internal state of the transformation."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An N-dimensional tensor of the same shape as the input representing the transformed sequence."
            }
        ],
        "algorithmic_properties": [
            "sequential",
            "linear",
            "stateful",
            "numerically-stabilized"
        ],
        "cross_disciplinary_applications": [
            "Applying high-order dynamic models to robotic control systems.",
            "Real-time processing of high-fidelity audio streams.",
            "Simulating multi-stage physical absorption processes in chemistry."
        ]
    }
    <!-- /conceptual_profile -->
    """
    result = scipy.signal.sosfilt(sos, x, axis=axis, zi=zi)
    if zi is not None:
        return result[0]
    return result


@register_atom(witness_lfilter)
@icontract.require(lambda b: bool(np.asarray(b).ndim == 1), "Numerator b must be 1D")
@icontract.require(lambda a: bool(np.asarray(a).ndim == 1), "Denominator a must be 1D")
@icontract.require(lambda a: np.asarray(a).ndim != 1 or float(np.asarray(a).flat[0]) != 0.0, "Leading denominator coefficient a[0] must not be zero")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input signal must not be empty")
@icontract.ensure(lambda result, x: result.shape == np.asarray(x).shape, "Output shape must match input shape")
def lfilter(
    b: ArrayLike,
    a: ArrayLike,
    x: ArrayLike,
    axis: int = -1,
    zi: np.ndarray | None = None,
) -> np.ndarray:
    """Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence x using a digital filter described by the
    numerator and denominator coefficient vectors b and a.

    Args:
        b: Numerator coefficient vector of the filter (1D).
        a: Denominator coefficient vector of the filter (1D).
            a[0] must be nonzero.
        x: Input signal array.
        axis: The axis of x to apply the filter.
        zi: Initial conditions for the filter delays.

    Returns:
        The filtered output with the same shape as x.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Linear Difference Equation Solver",
        "conceptual_transform": "Computes the output of a discrete-time linear time-invariant system given its input and rational transfer function coefficients. It iteratively solves a linear difference equation along a specified dimension.",
        "abstract_inputs": [
            {
                "name": "b",
                "description": "A 1D array representing the feedforward coefficients."
            },
            {
                "name": "a",
                "description": "A 1D array representing the feedback coefficients."
            },
            {
                "name": "x",
                "description": "An N-dimensional tensor representing the input sequence."
            },
            {
                "name": "axis",
                "description": "An integer specifying the dimension along which to solve the equation."
            },
            {
                "name": "zi",
                "description": "An optional array defining the initial internal state of the solver."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An N-dimensional tensor of the same shape as the input representing the solution."
            }
        ],
        "algorithmic_properties": [
            "sequential",
            "linear",
            "stateful",
            "recursive"
        ],
        "cross_disciplinary_applications": [
            "Calculating auto-regressive moving-average (ARMA) models in statistics.",
            "Simulating simple damped harmonic oscillators in physics.",
            "Tracking cumulative resource levels in a sequential allocation process."
        ]
    }
    <!-- /conceptual_profile -->
    """
    result = scipy.signal.lfilter(b, a, x, axis=axis, zi=zi)
    if zi is not None:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------

@register_atom(witness_freqz)
@icontract.require(lambda b: np.asarray(b).size > 0, "Numerator b must be non-empty")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (w, h) tuple")
@icontract.ensure(
    lambda result, worN: (
        result[0].shape[0] == (worN if isinstance(worN, int) else len(worN))
        if worN is not None else True
    ),
    "Frequency array length must match worN",
)
def freqz(
    b: ArrayLike,
    a: ArrayLike = 1,
    worN: Union[int, ArrayLike, None] = 512,
    whole: bool = False,
    fs: float = 2 * np.pi,
    include_nyquist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the frequency response of a digital filter.

    Compute the frequency response of a digital filter given the
    numerator and denominator coefficients.

    Args:
        b: Numerator of the transfer function.
        a: Denominator of the transfer function. Default is 1 (FIR).
        worN: Number of frequencies to compute, or array of
            frequencies. Default is 512.
        whole: If True, compute frequencies from 0 to 2*pi.
        fs: The sampling frequency of the digital system.
            Default is 2*pi (angular frequency).
        include_nyquist: If True, include the Nyquist frequency.

    Returns:
        Tuple of (w, h) where w is the frequency array and h is the
        complex frequency response.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Discrete Transfer Function Evaluator",
        "conceptual_transform": "Evaluates the complex frequency response of a linear system defined by a rational transfer function. It maps system coefficients to a set of complex values describing amplitude and phase shifts across a spectrum.",
        "abstract_inputs": [
            {
                "name": "b",
                "description": "A 1D array representing the numerator coefficients."
            },
            {
                "name": "a",
                "description": "A 1D array representing the denominator coefficients."
            },
            {
                "name": "worN",
                "description": "An integer specifying the number of evaluation points or an array of specific points."
            },
            {
                "name": "whole",
                "description": "A boolean indicating whether to evaluate over the entire periodic domain."
            },
            {
                "name": "fs",
                "description": "A scalar used to scale the evaluation domain."
            },
            {
                "name": "include_nyquist",
                "description": "A boolean indicating whether to include the upper boundary of the symmetric domain."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tuple containing a 1D array of evaluation points and a 1D array of complex responses."
            }
        ],
        "algorithmic_properties": [
            "deterministic",
            "spectral",
            "complex-valued"
        ],
        "cross_disciplinary_applications": [
            "Analyzing the stability margins of feedback control loops.",
            "Evaluating the dispersion characteristics of photonic metamaterials.",
            "Characterizing the resonant modes of mechanical vibrations in structures."
        ]
    }
    <!-- /conceptual_profile -->
    """
    w, h = scipy.signal.freqz(
        b, a=a, worN=worN, whole=whole, fs=fs,
        include_nyquist=include_nyquist,
    )
    return w, h
