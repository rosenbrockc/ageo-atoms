"""Auto-generated verified atom wrapper for Photoplethysmography (PPG) processing."""

import numpy as np
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_kazemi_peak_detection, witness_ppg_reconstruction, witness_ppg_sqa




@register_atom(witness_kazemi_peak_detection)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def kazemi_peak_detection(data: np.ndarray) -> np.ndarray:
    """Extracts local maxima from a wandering 1D scalar signal array.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(data)
    return peaks.astype(np.intp)

@register_atom(witness_ppg_reconstruction)
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def ppg_reconstruction(data: np.ndarray) -> np.ndarray:
    """Reconstructs corrupted segments of a 1D scalar sequence.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    # Simple signal reconstruction via interpolation of NaN/corrupted segments
    reconstructed = data.copy()
    nan_mask = ~np.isfinite(reconstructed)
    if nan_mask.any():
        valid = np.where(~nan_mask)[0]
        invalid = np.where(nan_mask)[0]
        if len(valid) >= 2:
            reconstructed[invalid] = np.interp(invalid, valid, reconstructed[valid])
    return reconstructed

@register_atom(witness_ppg_sqa)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
@icontract.require(lambda data: data.ndim >= 1, "data must have at least one dimension")
@icontract.require(lambda data: data is not None, "data must not be None")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be a numpy array")
@icontract.ensure(lambda result: result is not None, "result must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be a numpy array")
@icontract.ensure(lambda result: result.ndim >= 1, "result must have at least one dimension")
def ppg_sqa(data: np.ndarray) -> np.ndarray:
    """Performs Signal Quality Assessment (SQA), quantifying the reliability and signal-to-noise ratio of a 1D scalar array.

    Args:
        data: Input N-dimensional tensor or 1D scalar array.

    Returns:
        Processed output array.
    """
    # SQA: compute signal quality metrics (SNR, kurtosis, skewness)
    from scipy.stats import kurtosis, skew
    snr = np.mean(data ** 2) / (np.var(data) + 1e-15)
    k = kurtosis(data)
    s = skew(data)
    return np.array([snr, k, s])
