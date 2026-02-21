"""ECG atoms ingested via the Smart Ingester.

These atoms are deterministic wrappers with explicit contracts and ghost witness
bindings so they can participate cleanly in matcher/synthesizer rounds.
"""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.biosppy.ecg_witnesses import (
    witness_bandpass_filter,
    witness_r_peak_detection,
    witness_peak_correction,
    witness_template_extraction,
    witness_heart_rate_computation,
    witness_ssf_segmenter,
    witness_christov_segmenter,
)
from ageoa.ghost.registry import register_atom


def _peak_indices(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    min_distance_sec: float,
    height_scale: float,
    prominence_scale: float,
) -> np.ndarray:
    """Return monotonically-increasing peak indices using deterministic heuristics."""
    rectified = np.abs(signal)
    if rectified.size == 0:
        return np.empty(0, dtype=np.int64)

    peak_level = float(np.max(rectified))
    if peak_level <= 0.0:
        return np.empty(0, dtype=np.int64)

    distance = max(1, int(round(min_distance_sec * sampling_rate)))
    height = max(1e-12, height_scale * peak_level)
    prominence = max(1e-12, prominence_scale * peak_level)

    peaks, _ = scipy.signal.find_peaks(
        rectified,
        distance=distance,
        height=height,
        prominence=prominence,
    )
    return np.asarray(peaks, dtype=np.int64)


@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.require(lambda lowcut, highcut, sampling_rate: 0 < lowcut < highcut < (sampling_rate / 2.0), "Cutoffs must satisfy 0 < lowcut < highcut < Nyquist")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output shape must match input")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "Filtered output must be finite")
def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float = 1000.0,
    lowcut: float = 3.0,
    highcut: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a finite impulse response bandpass filter to isolate a target frequency band from a uniformly sampled 1D signal.
    """
    x = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * sampling_rate
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")

    # filtfilt is preferred for zero phase; for short signals fallback to lfilter.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.size <= padlen:
        return scipy.signal.lfilter(b, a, x)
    return scipy.signal.filtfilt(b, a, x)


@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda filtered: filtered.size > 0, "Filtered signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "R-peaks must be a 1D array")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "R-peaks must be integer indices")
@icontract.ensure(lambda result: result.size == 0 or bool(np.all(np.diff(result) > 0)), "R-peaks must be strictly increasing")
def r_peak_detection(filtered: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect dominant periodic peaks in a filtered 1D signal using adaptive thresholding.
    """
    x = np.asarray(filtered, dtype=np.float64)
    return _peak_indices(
        x,
        sampling_rate,
        min_distance_sec=0.4,
        height_scale=0.3,
        prominence_scale=0.1,
    )


@register_atom(witness_peak_correction)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result, rpeaks: result.shape == rpeaks.shape, "Corrected peaks must preserve count")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "Corrected peaks must be integer indices")
def peak_correction(
    filtered: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate: float = 1000.0,
    tolerance_sec: float = 0.05,
) -> np.ndarray:
    """Refine detected peak locations to the nearest local maximum within a tolerance window.
    """
    x = np.asarray(filtered, dtype=np.float64)
    n = x.size
    peaks = np.asarray(rpeaks, dtype=np.int64)

    if peaks.size == 0:
        return peaks

    tol = max(1, int(round(tolerance_sec * sampling_rate)))
    corrected = np.empty_like(peaks)

    for i, p in enumerate(peaks):
        center = int(np.clip(p, 0, max(0, n - 1)))
        lo = max(0, center - tol)
        hi = min(n, center + tol + 1)
        corrected[i] = lo + int(np.argmax(np.abs(x[lo:hi])))

    return corrected


@register_atom(witness_template_extraction)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.require(lambda before_sec: before_sec > 0, "before_sec must be positive")
@icontract.require(lambda after_sec: after_sec > 0, "after_sec must be positive")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (templates, rpeaks_final)")
@icontract.ensure(
    lambda result, before_sec, after_sec, sampling_rate: result[0].ndim == 2
    and result[0].shape[1] == max(1, int(round((before_sec + after_sec) * sampling_rate))),
    "Template width must match the extraction window",
)
@icontract.ensure(lambda result: result[1].ndim == 1, "Final R-peaks must be 1D")
def template_extraction(
    filtered: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate: float = 1000.0,
    before_sec: float = 0.2,
    after_sec: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract windowed waveform segments centered on detected peak locations.
    """
    x = np.asarray(filtered, dtype=np.float64)
    peaks = np.asarray(rpeaks, dtype=np.int64)

    before = max(1, int(round(before_sec * sampling_rate)))
    after = max(1, int(round(after_sec * sampling_rate)))
    width = before + after

    templates: list[np.ndarray] = []
    kept_peaks: list[int] = []

    for peak in peaks:
        start = int(peak) - before
        end = int(peak) + after
        if start < 0 or end > x.size:
            continue
        templates.append(x[start:end])
        kept_peaks.append(int(peak))

    if not templates:
        return np.empty((0, width), dtype=np.float64), np.empty((0,), dtype=np.int64)

    return np.vstack(templates), np.asarray(kept_peaks, dtype=np.int64)


@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (hr_idx, heart_rate)")
@icontract.ensure(lambda result: result[0].shape == result[1].shape, "hr_idx and heart_rate must have the same shape")
@icontract.ensure(lambda result, rpeaks: result[0].size == max(0, rpeaks.size - 1), "Heart-rate count must match RR interval count")
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])), "Heart-rate values must be finite")
def heart_rate_computation(
    rpeaks: np.ndarray,
    sampling_rate: float = 1000.0,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous event rate from inter-event intervals with optional smoothing.
    """
    peaks = np.asarray(rpeaks, dtype=np.int64)
    if peaks.size < 2:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)

    rr_sec = np.diff(peaks).astype(np.float64) / sampling_rate
    heart_rate = np.zeros_like(rr_sec, dtype=np.float64)
    valid = rr_sec > 0
    heart_rate[valid] = 60.0 / rr_sec[valid]

    if smooth and heart_rate.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        heart_rate = np.convolve(heart_rate, kernel, mode="same")

    return peaks[1:], heart_rate


@register_atom(witness_ssf_segmenter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "peaks must be 1D")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "peaks must be integer indices")
def ssf_segmenter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect periodic spikes in a 1D time-series scalar array using slope-sum thresholding."""
    filtered = bandpass_filter(signal, sampling_rate=sampling_rate)
    slope = np.diff(filtered, prepend=filtered[0])
    ssf = np.square(np.maximum(slope, 0.0))

    window = max(1, int(round(0.08 * sampling_rate)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    envelope = np.convolve(ssf, kernel, mode="same")

    peak_level = float(np.max(envelope))
    if peak_level <= 0.0:
        return np.empty((0,), dtype=np.int64)

    peaks, _ = scipy.signal.find_peaks(
        envelope,
        distance=max(1, int(round(0.35 * sampling_rate))),
        height=max(1e-12, 0.25 * peak_level),
        prominence=max(1e-12, 0.10 * peak_level),
    )
    return np.asarray(peaks, dtype=np.int64)


@register_atom(witness_christov_segmenter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "peaks must be 1D")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "peaks must be integer indices")
def christov_segmenter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Isolate discrete events in a 1D time-series scalar array via adaptive envelope thresholding."""
    filtered = bandpass_filter(
        signal,
        sampling_rate=sampling_rate,
        lowcut=2.0,
        highcut=min(40.0, 0.49 * sampling_rate),
        order=2,
    )

    derivative = np.abs(np.diff(filtered, prepend=filtered[0]))
    window = max(1, int(round(0.12 * sampling_rate)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    envelope = np.convolve(derivative, kernel, mode="same")

    peak_level = float(np.max(envelope))
    if peak_level <= 0.0:
        return np.empty((0,), dtype=np.int64)

    adaptive_height = float(np.median(envelope) + 0.6 * np.std(envelope))
    peaks, _ = scipy.signal.find_peaks(
        envelope,
        distance=max(1, int(round(0.30 * sampling_rate))),
        height=max(1e-12, adaptive_height),
        prominence=max(1e-12, 0.08 * peak_level),
    )

    if peaks.size == 0:
        return _peak_indices(
            filtered,
            sampling_rate,
            min_distance_sec=0.35,
            height_scale=0.20,
            prominence_scale=0.08,
        )

    return np.asarray(peaks, dtype=np.int64)


@register_atom(witness_ssf_segmenter)  # Assuming same witness structure for simplicity
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "peaks must be 1D")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "peaks must be integer indices")
def hamilton_segmenter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect periodic spikes in a 1D time-series scalar array using bandpass filtering and moving averages."""
    # Dummy skeleton implementation, as requested for manual ingestion without original code
    raise NotImplementedError("Skeleton for future ingestion.")
