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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Complex Orthogonal Basis Projector",
        "conceptual_transform": "Transforms a sequence from its native coordinate system (typically time or space) into a representation based on complex sinusoidal basis functions. It maps a set of samples to a set of complex amplitudes and phases, revealing the fundamental periodicities of the input.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D or ND tensor containing complex or real-valued samples."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the number of basis functions to use (projection resolution)."
            },
            {
                "name": "axis",
                "description": "The dimension along which to perform the projection."
            },
            {
                "name": "norm",
                "description": "An optional string defining the scaling of the basis functions."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A complex-valued tensor representing the amplitudes and phases of the sinusoidal components."
            }
        ],
        "algorithmic_properties": [
            "spectral",
            "complex-valued",
            "invertible",
            "orthogonal-projection"
        ],
        "cross_disciplinary_applications": [
            "Analyzing the vibration frequencies of an aircraft wing from sensor data.",
            "Decomposing a complex audio waveform into its constituent musical notes.",
            "Filtering out specific periodic noise in digital image processing."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Complex Orthogonal Basis Synthesizer",
        "conceptual_transform": "Reconstructs a sequence in its native coordinate system from its representation as a sum of complex sinusoidal basis functions. It is the perfect inverse of the spectral projection.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A complex-valued tensor of spectral amplitudes and phases."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the number of synthesis points."
            },
            {
                "name": "axis",
                "description": "The dimension along which to perform the synthesis."
            },
            {
                "name": "norm",
                "description": "An optional string defining the scaling."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A complex-valued tensor representing the synthesized spatial or temporal signal."
            }
        ],
        "algorithmic_properties": [
            "spectral-synthesis",
            "complex-valued",
            "invertible"
        ],
        "cross_disciplinary_applications": [
            "Reconstructing a physical waveform from a filtered frequency-domain representation.",
            "Generating acoustic signals from specified harmonic content.",
            "Synthesizing spatial fields from their spectral domain solutions in physics."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Symmetric Real-to-Complex Projector",
        "conceptual_transform": "Projects a real-valued sequence into a compact spectral representation by exploiting the Hermitian symmetry of the resulting complex coefficients. It provides the same information as a full spectral projection but in a more memory-efficient form.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A 1D or ND real-valued tensor."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the number of projection points."
            },
            {
                "name": "axis",
                "description": "The dimension along which to perform the projection."
            },
            {
                "name": "norm",
                "description": "An optional scaling mode."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A complex-valued tensor representing the non-redundant half of the spectral amplitudes."
            }
        ],
        "algorithmic_properties": [
            "spectral",
            "symmetry-exploiting",
            "memory-efficient",
            "real-to-complex"
        ],
        "cross_disciplinary_applications": [
            "Efficiently analyzing the frequency content of real-world sensor streams.",
            "Preprocessing real-valued audio data for spectral feature extraction.",
            "Compressed representation of real physical fields in the frequency domain."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Symmetric Complex-to-Real Synthesizer",
        "conceptual_transform": "Reconstructs a real-valued sequence from its non-redundant complex spectral representation by assuming Hermitian symmetry. It maps a halved spectral tensor back into a full-sized real-valued sequence.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "A complex-valued tensor of non-redundant spectral amplitudes."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the length of the reconstructed sequence."
            },
            {
                "name": "axis",
                "description": "The dimension along which to perform the synthesis."
            },
            {
                "name": "norm",
                "description": "An optional scaling mode."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A real-valued tensor representing the synthesized spatial or temporal signal."
            }
        ],
        "algorithmic_properties": [
            "spectral-synthesis",
            "symmetry-exploiting",
            "complex-to-real"
        ],
        "cross_disciplinary_applications": [
            "Synthesizing real-valued time-series from manipulated frequency spectra.",
            "Reconstructing real-world signals after frequency-domain filtering.",
            "Efficiently generating real physical fields from spectral data."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Spectral Domain Coordinate Generator",
        "conceptual_transform": "Generates the discrete coordinate system (bin centers) for the spectral domain corresponding to a given spatial or temporal sampling configuration. It maps sampling parameters to an absolute physical scale.",
        "abstract_inputs": [
            {
                "name": "n",
                "description": "An integer specifying the number of points in the transform window."
            },
            {
                "name": "d",
                "description": "A scalar representing the interval between samples in the native coordinate system."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A 1D tensor of coordinates for each spectral bin."
            }
        ],
        "algorithmic_properties": [
            "coordinate-mapping",
            "deterministic",
            "domain-resolution"
        ],
        "cross_disciplinary_applications": [
            "Labeling the frequency axes in a vibration analysis plot.",
            "Determining the physical wavelength bins in hyperspectral imaging.",
            "Calibrating the Doppler shift bins in radar signal processing."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Periodic Domain Centering Shifter",
        "conceptual_transform": "Reorganizes a periodic sequence by swapping its halves, typically moving the zero-frequency (DC) component from the start to the center for intuitive visualization and processing. It performs a circular shift of exactly half the domain length.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "A tensor representing a periodically sampled signal or its transform."
            },
            {
                "name": "axes",
                "description": "An optional integer or sequence specifying which dimensions to shift."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The tensor with reorganized periodic halves."
            }
        ],
        "algorithmic_properties": [
            "circular-shift",
            "permutation",
            "re-centering"
        ],
        "cross_disciplinary_applications": [
            "Centering the zero-frequency component in a 2D optical diffraction pattern.",
            "Preprocessing spectral data for standard visualization in signal analyzers.",
            "Aligning periodic boundary conditions in spatial simulations."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.fft.fftshift(x, axes=axes)
