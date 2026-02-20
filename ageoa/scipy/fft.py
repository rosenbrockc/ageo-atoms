import os

import numpy as np
import scipy.fft
import icontract
from typing import Union

from ageoa.ghost.registry import register_atom
from ageoa.ghost.witnesses import witness_dct, witness_idct

ArrayLike = Union[np.ndarray, list, tuple]

_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"


def _roundtrip_close(original: np.ndarray, reconstructed: np.ndarray, atol: float = 1e-10) -> bool:
    """Check that a round-trip reconstruction is close to the original."""
    return bool(np.allclose(original, reconstructed, atol=atol))


@register_atom(witness_dct)
@icontract.require(lambda x: x is not None, "Input array must not be None")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, x, n: result.shape == (np.asarray(x).shape if n is None else
        tuple(n if i == (np.asarray(x).ndim - 1) else s for i, s in enumerate(np.asarray(x).shape))),
    "Output shape must be preserved (or match n along axis)",
)
@icontract.ensure(lambda result: np.isrealobj(result), "DCT output must be real-valued")
@icontract.ensure(
    lambda result, x, type, n, axis, norm: _roundtrip_close(
        np.asarray(x),
        scipy.fft.idct(result, type=type, n=n, axis=axis, norm=norm),
    ),
    "Round-trip IDCT(DCT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def dct(
    x: ArrayLike,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    """Compute the Discrete Cosine Transform.

    Computes the DCT of the input array along the specified axis.
    The DCT is a real-valued transform related to the DFT.

    Args:
        x: Input array, must be real-valued.
        type: Type of DCT (1, 2, 3, or 4). Default is 2.
        n: Length of the transform. If n is smaller than the input,
            the input is cropped. If larger, padded with zeros.
        axis: Axis over which to compute the DCT. Default is -1.
        norm: Normalization mode. None or "ortho".
        overwrite_x: If True, the contents of x may be destroyed.

    Returns:
        The DCT of the input array, real-valued, with shape preserved.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Real-Valued Spectral Decomposer",
        "conceptual_transform": "Projects an N-dimensional real-valued spatial or temporal signal onto a basis of real orthogonal cosine functions, summarizing its frequency content without complex arithmetic.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "An N-dimensional tensor containing real-valued measurements."
            },
            {
                "name": "type",
                "description": "An integer specifying the boundary conditions and symmetry of the basis functions."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the length of the projection window."
            },
            {
                "name": "axis",
                "description": "An integer specifying the dimension to project."
            },
            {
                "name": "norm",
                "description": "An optional string indicating whether the basis functions are normalized."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An N-dimensional tensor representing the amplitudes of the real-valued frequency components."
            }
        ],
        "algorithmic_properties": [
            "spectral",
            "real-valued",
            "orthogonal-projection",
            "invertible"
        ],
        "cross_disciplinary_applications": [
            "Compressing high-resolution 2D image tiles (JPEG compression).",
            "Extracting acoustic features (MFCCs) for pattern recognition models operating on sequential feature vectors.",
            "Solving boundary value problems in computational fluid dynamics."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.fft.dct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)


@register_atom(witness_idct)
@icontract.require(lambda x: x is not None, "Input array must not be None")
@icontract.require(lambda x: np.asarray(x).size > 0, "Input array must not be empty")
@icontract.ensure(
    lambda result, x, n: result.shape == (np.asarray(x).shape if n is None else
        tuple(n if i == (np.asarray(x).ndim - 1) else s for i, s in enumerate(np.asarray(x).shape))),
    "Output shape must be preserved (or match n along axis)",
)
@icontract.ensure(lambda result: np.isrealobj(result), "IDCT output must be real-valued")
@icontract.ensure(
    lambda result, x, type, n, axis, norm: _roundtrip_close(
        np.asarray(x),
        scipy.fft.dct(result, type=type, n=n, axis=axis, norm=norm),
    ),
    "Round-trip DCT(IDCT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def idct(
    x: ArrayLike,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    """Compute the Inverse Discrete Cosine Transform.

    Computes the IDCT of the input array along the specified axis.

    Args:
        x: Input array, must be real-valued.
        type: Type of DCT (1, 2, 3, or 4). Default is 2.
        n: Length of the transform. If n is smaller than the input,
            the input is cropped. If larger, padded with zeros.
        axis: Axis over which to compute the IDCT. Default is -1.
        norm: Normalization mode. None or "ortho".
        overwrite_x: If True, the contents of x may be destroyed.

    Returns:
        The IDCT of the input array, real-valued, with shape preserved.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Real-Valued Spatial Synthesizer",
        "conceptual_transform": "Reconstructs an N-dimensional real-valued spatial or temporal signal from its projection coefficients on a basis of real orthogonal cosine functions.",
        "abstract_inputs": [
            {
                "name": "x",
                "description": "An N-dimensional tensor containing real-valued spectral amplitudes."
            },
            {
                "name": "type",
                "description": "An integer specifying the boundary conditions and symmetry of the basis functions."
            },
            {
                "name": "n",
                "description": "An optional integer specifying the length of the reconstruction window."
            },
            {
                "name": "axis",
                "description": "An integer specifying the dimension to reconstruct."
            },
            {
                "name": "norm",
                "description": "An optional string indicating whether the basis functions are normalized."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "An N-dimensional tensor representing the reconstructed real-valued spatial or temporal signal."
            }
        ],
        "algorithmic_properties": [
            "spectral-synthesis",
            "real-valued",
            "orthogonal-projection",
            "invertible"
        ],
        "cross_disciplinary_applications": [
            "Reconstructing spatial data blocks from compressed spectral coefficients.",
            "Re-synthesizing temporal waveforms from compressed latent representations.",
            "Reconstructing physical fields from spectral domain solutions."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return scipy.fft.idct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
