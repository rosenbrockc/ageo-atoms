"""Phonocardiogram (PCG) atoms ingested via the Smart Ingester."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.biosppy.pcg_witnesses import (
    witness_shannon_energy,
    witness_pcg_segmentation,
)
from ageoa.ghost.registry import register_atom


def _safe_lowpass(signal: np.ndarray, sampling_rate: float, cutoff_hz: float) -> np.ndarray:
    nyq = 0.5 * sampling_rate
    cutoff = min(cutoff_hz, 0.45 * sampling_rate)
    b, a = scipy.signal.butter(2, cutoff / nyq, btype="low")

    padlen = 3 * (max(len(a), len(b)) - 1)
    if signal.size <= padlen:
        return scipy.signal.lfilter(b, a, signal)
    return scipy.signal.filtfilt(b, a, signal)


@register_atom(witness_shannon_energy)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output shape must match input")
@icontract.ensure(lambda result: np.all(result >= 0), "Shannon energy must be non-negative")
def shannon_energy(signal: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon energy envelope for a quasi-periodic signal.

    Args:
        signal: 1D input signal array.

    Returns:
        1D non-negative float array of Shannon energy values with the
        same shape as input.
    """
    x = np.asarray(signal, dtype=np.float64)
    max_abs = float(np.max(np.abs(x)))
    if max_abs <= 0.0:
        return np.zeros_like(x)

    x_norm = x / max_abs
    x2 = np.clip(x_norm ** 2, 1e-12, 1.0)
    energy = -x2 * np.log(x2)
    return np.maximum(energy, 0.0)


@register_atom(witness_pcg_segmentation)
@icontract.require(lambda envelope: envelope.ndim == 1, "Envelope must be 1D")
@icontract.require(lambda envelope: envelope.size > 0, "Envelope must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (S1, S2)")
@icontract.ensure(lambda result: np.issubdtype(result[0].dtype, np.integer), "S1 indices must be integer")
@icontract.ensure(lambda result: np.issubdtype(result[1].dtype, np.integer), "S2 indices must be integer")
def pcg_segmentation(envelope: np.ndarray, sampling_rate: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Segment a cyclic signal into alternating event classes from an energy envelope.

    Args:
        envelope: 1D energy envelope array.
        sampling_rate: Sampling rate in Hz. Default is 1000.0.

    Returns:
        Tuple of (S1, S2) where each is a 1D integer array of event indices.
    """
    x = np.asarray(envelope, dtype=np.float64)
    smooth_envelope = _safe_lowpass(x, sampling_rate=sampling_rate, cutoff_hz=20.0)
    smooth_envelope = np.maximum(smooth_envelope, 0.0)

    peak_level = float(np.max(smooth_envelope))
    if peak_level <= 0.0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    threshold = 0.05 * peak_level
    min_distance = max(1, int(round(0.2 * sampling_rate)))

    peaks, _ = scipy.signal.find_peaks(
        smooth_envelope,
        height=threshold,
        distance=min_distance,
    )
    if peaks.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    # Deterministic alternating assignment: S1 starts, then S2.
    s1 = np.asarray(peaks[::2], dtype=np.int64)
    s2 = np.asarray(peaks[1::2], dtype=np.int64)
    return s1, s2
