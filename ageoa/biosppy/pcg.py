"""PCG atoms ingested via the Smart Ingester."""

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

<!-- conceptual_profile -->
{
    "abstract_name": "Nonlinear Log-Entropy Envelope Transformer",
    "conceptual_transform": "Computes a non-negative energy envelope by weighting squared signal components with their self-information (negative log-probability). It effectively amplifies low-magnitude structural variations while compressing high-magnitude peaks, making it ideal for detecting subtle transients in noisy sequences.",
    "abstract_inputs": [
        {
            "name": "signal",
            "description": "A 1D tensor representing a physical measurement sequence."
        }
    ],
    "abstract_outputs": [
        {
            "name": "result",
            "description": "A 1D non-negative tensor representing the entropy-weighted energy envelope."
        }
    ],
    "algorithmic_properties": [
        "nonlinear",
        "entropy-weighted",
        "envelope-extraction",
        "magnitude-invariant-scaling"
    ],
    "cross_disciplinary_applications": [
        "Detecting micro-cracks in ultrasonic material testing signals.",
        "Isolating subtle speech onsets in high-noise environments.",
        "Identifying low-energy seismic precursors in geomechanical monitoring."
    ]
}
<!-- /conceptual_profile -->
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

<!-- conceptual_profile -->
{
    "abstract_name": "Alternating Bimodal Event Partitioner",
    "conceptual_transform": "Partitions a sequence of detected event indices into two distinct alternating categories based on a cyclic prior. It transforms a single stream of event timestamps into a structured multi-class event sequence.",
    "abstract_inputs": [
        {
            "name": "envelope",
            "description": "A 1D non-negative tensor representing the signal's energy profile."
        },
        {
            "name": "sampling_rate",
            "description": "A scalar representing temporal resolution."
        }
    ],
    "abstract_outputs": [
        {
            "name": "s1",
            "description": "A 1D tensor of indices for the first event class in the cycle."
        },
        {
            "name": "s2",
            "description": "A 1D tensor of indices for the second event class in the cycle."
        }
    ],
    "algorithmic_properties": [
        "cyclic-partitioning",
        "bimodal",
        "event-classification",
        "deterministic-alternation"
    ],
    "cross_disciplinary_applications": [
        "Distinguishing between intake and exhaust strokes in a simplified internal combustion engine cycle.",
        "Separating loading and unloading phases in a rhythmic mechanical industrial process.",
        "Classifying alternating 'tick' and 'tock' events in an acoustic clockwork mechanism."
    ]
}
<!-- /conceptual_profile -->
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
