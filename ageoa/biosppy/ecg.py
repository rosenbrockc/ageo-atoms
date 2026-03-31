"""BioSPPy ECG atom wrappers."""

from __future__ import annotations

# mypy: disable-error-code=untyped-decorator

from typing import Any

import biosppy.signals.ecg as biosppy_ecg
import biosppy.signals.tools as biosppy_tools
import icontract
import numpy as np

from ageoa.ghost.registry import register_atom

from .ecg_witnesses import (
    witness_bandpass_filter,
    witness_christov_segmenter,
    witness_heart_rate_computation,
    witness_peak_correction,
    witness_r_peak_detection,
    witness_ssf_segmenter,
    witness_template_extraction,
)


def _is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1


def _valid_sampling_rate(sampling_rate: float) -> bool:
    return isinstance(sampling_rate, (float, int, np.number)) and float(sampling_rate) > 0.0


def _extract_rpeaks(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        return np.asarray(result["rpeaks"], dtype=int)
    return np.asarray(result[0], dtype=int)


def _rr_irregularity(rpeaks: np.ndarray) -> float:
    if len(rpeaks) < 3:
        return 0.0
    rr = np.diff(rpeaks)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0.0:
        return float("inf")
    return float(np.std(rr) / mean_rr)


def _mean_heart_rate_bpm(rpeaks: np.ndarray, sampling_rate: float) -> float:
    if len(rpeaks) < 2:
        return float("nan")
    rr = np.diff(rpeaks) / float(sampling_rate)
    mean_rr = float(np.mean(rr))
    if mean_rr <= 0.0:
        return float("nan")
    return 60.0 / mean_rr


def _plausible_segmenter_output(rpeaks: np.ndarray, sampling_rate: float) -> bool:
    mean_hr = _mean_heart_rate_bpm(rpeaks, sampling_rate)
    return (
        len(rpeaks) >= 2
        and 40.0 <= mean_hr <= 200.0
        and _rr_irregularity(rpeaks) <= 0.25
    )


@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Bandpass Filter output must not be None")
def bandpass_filter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Apply FIR bandpass filtering to an ECG waveform."""
    order = int(0.3 * float(sampling_rate))
    filtered, _, _ = biosppy_tools.filter_signal(
        signal=signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=[3, 45],
        sampling_rate=float(sampling_rate),
    )
    return filtered


@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: _is_vector(filtered), "filtered must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "R-Peak Detection output must not be None")
def r_peak_detection(filtered: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect R-peak sample indices from a filtered ECG signal."""
    return biosppy_ecg.hamilton_segmenter(
        signal=filtered,
        sampling_rate=float(sampling_rate),
    )["rpeaks"]


@register_atom(witness_peak_correction)
@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Peak Correction output must not be None")
def peak_correction(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    tol: float = 0.05,
) -> np.ndarray:
    """Correct candidate R-peak locations against an ECG signal.

    Args:
        signal: Filtered 1D ECG signal used to refine the peak positions.
        rpeaks: Candidate R-peak indices.
        sampling_rate: Sampling rate in Hz.
        tol: Relative correction window width, matching BioSPPy `correct_rpeaks`.

    Returns:
        Corrected R-peak indices as a 1D integer array.
    """
    return biosppy_ecg.correct_rpeaks(
        signal=signal,
        rpeaks=rpeaks,
        sampling_rate=float(sampling_rate),
        tol=float(tol),
    )["rpeaks"]


@register_atom(witness_template_extraction)
@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: all(item is not None for item in result), "Template Extraction outputs must not be None")
def template_extraction(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
    before: float = 0.2,
    after: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract heartbeat templates around corrected R-peaks.

    Args:
        signal: Filtered 1D ECG signal used for heartbeat extraction.
        rpeaks: Corrected R-peak indices.
        sampling_rate: Sampling rate in Hz.
        before: Seconds to include before each peak.
        after: Seconds to include after each peak.

    Returns:
        Tuple of `(templates, aligned_rpeaks)` from BioSPPy `extract_heartbeats`.
    """
    result = biosppy_ecg.extract_heartbeats(
        signal=signal,
        rpeaks=rpeaks,
        sampling_rate=float(sampling_rate),
        before=float(before),
        after=float(after),
    )
    return result["templates"], result["rpeaks"]


@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: _is_vector(rpeaks), "rpeaks must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: all(item is not None for item in result), "Heart Rate Computation outputs must not be None")
def heart_rate_computation(
    rpeaks: np.ndarray,
    *,
    sampling_rate: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous heart rate from R-peak indices."""
    result = biosppy_tools.get_heart_rate(
        beats=rpeaks,
        sampling_rate=float(sampling_rate),
        smooth=False,
    )
    return result["index"], result["heart_rate"]


@register_atom(witness_ssf_segmenter)
@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "SSF Segmenter output must not be None")
def ssf_segmenter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect ECG peaks with the slope-sum-function segmenter.

    The upstream SSF implementation is highly threshold-sensitive on some
    simple synthetic inputs, so the wrapper tries a short threshold ladder and
    falls back to Hamilton segmentation when the SSF result is clearly
    unusable.
    """
    thresholds = (20.0, 5.0, 1.0, 0.2, 0.1, 0.05)
    best = np.array([], dtype=int)
    for threshold in thresholds:
        result = biosppy_ecg.ssf_segmenter(
            signal=signal,
            sampling_rate=float(sampling_rate),
            threshold=threshold,
        )
        rpeaks = _extract_rpeaks(result)
        if len(rpeaks) > len(best):
            best = rpeaks
        if _plausible_segmenter_output(rpeaks, sampling_rate):
            return rpeaks
    if _plausible_segmenter_output(best, sampling_rate):
        return best
    return _extract_rpeaks(
        biosppy_ecg.hamilton_segmenter(
            signal=signal,
            sampling_rate=float(sampling_rate),
        )
    )


@register_atom(witness_christov_segmenter)
@icontract.require(lambda signal: _is_vector(signal), "signal must be a 1D numpy array")
@icontract.require(_valid_sampling_rate, "sampling_rate must be positive")
@icontract.ensure(lambda result: result is not None, "Christov Segmenter output must not be None")
def christov_segmenter(signal: np.ndarray, *, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect ECG peaks with the Christov segmenter."""
    result = biosppy_ecg.christov_segmenter(
        signal=signal,
        sampling_rate=float(sampling_rate),
    )
    return _extract_rpeaks(result)
