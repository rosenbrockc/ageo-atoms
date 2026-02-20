"""EDA atoms ingested via the Smart Ingester."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.biosppy.eda_witnesses import (
    witness_gamboa_segmenter,
    witness_eda_feature_extraction,
)
from ageoa.ghost.registry import register_atom


def _safe_lowpass(signal: np.ndarray, sampling_rate: float, cutoff_hz: float) -> np.ndarray:
    """Low-pass filter with deterministic fallback for short inputs."""
    nyq = 0.5 * sampling_rate
    cutoff = min(cutoff_hz, 0.45 * sampling_rate)
    b, a = scipy.signal.butter(2, cutoff / nyq, btype="low")

    padlen = 3 * (max(len(a), len(b)) - 1)
    if signal.size <= padlen:
        return scipy.signal.lfilter(b, a, signal)
    return scipy.signal.filtfilt(b, a, signal)


@register_atom(witness_gamboa_segmenter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "Onset indices must be 1D")
@icontract.ensure(lambda result: np.issubdtype(result.dtype, np.integer), "Onset indices must be integer")
@icontract.ensure(lambda result: result.size == 0 or bool(np.all(np.diff(result) > 0)), "Onset indices must be strictly increasing")
def gamboa_segmenter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect transient onset events in a low-frequency signal via derivative peak analysis.

<!-- conceptual_profile -->
{
    "abstract_name": "Phasic Rise Onset Detector",
    "conceptual_transform": "Identifies the starting points of significant upward transitions in a low-frequency signal by analyzing its derivative peaks. It maps a continuous slow-moving sequence to a discrete set of indices where a new 'rise' event begins.",
    "abstract_inputs": [
        {
            "name": "signal",
            "description": "A 1D tensor representing a continuous physical measurement with slow baseline changes and superimposed fast transients."
        },
        {
            "name": "sampling_rate",
            "description": "A scalar representing the temporal resolution."
        }
    ],
    "abstract_outputs": [
        {
            "name": "result",
            "description": "A 1D tensor of integer indices representing the onsets of detected transient rises."
        }
    ],
    "algorithmic_properties": [
        "derivative-based",
        "thresholding",
        "peak-finding",
        "event-delimiting"
    ],
    "cross_disciplinary_applications": [
        "Detecting the onset of thermal expansion events in a temperature monitoring system.",
        "Identifying the beginning of sudden pressure increases in a chemical reactor.",
        "Locating the start of slow-onset structural shifts in geomechanical sensors."
    ]
}
<!-- /conceptual_profile -->
    """
    x = np.asarray(signal, dtype=np.float64)

    filtered = _safe_lowpass(x, sampling_rate=sampling_rate, cutoff_hz=2.0)
    diff = np.diff(filtered)
    if diff.size == 0:
        return np.empty((0,), dtype=np.int64)

    positive_diff = np.maximum(diff, 0.0)
    max_rise = float(np.max(positive_diff))
    if max_rise <= 0.0:
        return np.empty((0,), dtype=np.int64)

    threshold = 0.5 * max_rise
    min_distance = max(1, int(round(5.0 * sampling_rate)))
    peaks, _ = scipy.signal.find_peaks(
        positive_diff,
        height=threshold,
        distance=min_distance,
    )

    if peaks.size == 0:
        return np.empty((0,), dtype=np.int64)

    epsilon = max(1e-12, 0.01 * max_rise)
    onsets: list[int] = []
    for peak in peaks:
        onset = int(peak)
        while onset > 0 and positive_diff[onset] > epsilon:
            onset -= 1
        onsets.append(onset)

    return np.unique(np.asarray(onsets, dtype=np.int64))


@register_atom(witness_eda_feature_extraction)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda signal: signal.size > 0, "Signal must not be empty")
@icontract.require(lambda onsets: onsets.ndim == 1, "Onsets must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 3, "Output must be (amplitudes, rise_times, decay_times)")
@icontract.ensure(lambda result, onsets: result[0].shape == onsets.shape, "Amplitude shape must match onsets")
@icontract.ensure(lambda result, onsets: result[1].shape == onsets.shape, "Rise-time shape must match onsets")
@icontract.ensure(lambda result, onsets: result[2].shape == onsets.shape, "Decay-time shape must match onsets")
def eda_feature_extraction(
    signal: np.ndarray,
    onsets: np.ndarray,
    sampling_rate: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract amplitude, rise-time, and half-recovery decay-time for each onset.

<!-- conceptual_profile -->
{
    "abstract_name": "Transient Event Morphology Characterizer",
    "conceptual_transform": "Quantifies the structural properties (magnitude, growth rate, and decay rate) of localized transient events following identified onsets. It transforms event indices and the raw signal into a structured representation of event dynamics.",
    "abstract_inputs": [
        {
            "name": "signal",
            "description": "A 1D tensor representing the source signal."
        },
        {
            "name": "onsets",
            "description": "A 1D tensor of event starting indices."
        },
        {
            "name": "sampling_rate",
            "description": "A scalar representing temporal resolution."
        }
    ],
    "abstract_outputs": [
        {
            "name": "amplitudes",
            "description": "A 1D tensor representing the peak magnitude of each event relative to its onset baseline."
        },
        {
            "name": "rise_times",
            "description": "A 1D tensor representing the duration from onset to peak magnitude."
        },
        {
            "name": "decay_times",
            "description": "A 1D tensor representing the duration for the event to return to a half-magnitude state."
        }
    ],
    "algorithmic_properties": [
        "feature-extraction",
        "temporal-analysis",
        "event-characterization"
    ],
    "cross_disciplinary_applications": [
        "Characterizing the heating and cooling profiles of individual pulses in a laser material process.",
        "Analyzing the rise and fall times of individual power surge events in an electrical grid.",
        "Measuring the kinetics of transient chemical concentration spikes in a microfluidic channel."
    ]
}
<!-- /conceptual_profile -->
    """
    x = np.asarray(signal, dtype=np.float64)
    onset_idx = np.asarray(onsets, dtype=np.int64)

    amplitudes = np.zeros(onset_idx.shape, dtype=np.float64)
    rise_times = np.zeros(onset_idx.shape, dtype=np.float64)
    decay_times = np.zeros(onset_idx.shape, dtype=np.float64)

    search_window = max(1, int(round(5.0 * sampling_rate)))

    for i, onset in enumerate(onset_idx):
        if onset < 0 or onset >= x.size:
            continue

        end = min(x.size, int(onset) + search_window)
        if end <= onset + 1:
            continue

        peak_local = int(np.argmax(x[onset:end]))
        peak_idx = int(onset) + peak_local

        baseline = float(x[onset])
        peak_value = float(x[peak_idx])
        amp = max(0.0, peak_value - baseline)

        amplitudes[i] = amp
        rise_times[i] = max(0.0, (peak_idx - int(onset)) / sampling_rate)

        if amp <= 0.0:
            decay_times[i] = 0.0
            continue

        half_level = baseline + 0.5 * amp
        tail = x[peak_idx:end]
        below = np.where(tail <= half_level)[0]

        if below.size > 0:
            decay_times[i] = max(1.0 / sampling_rate, float(below[0]) / sampling_rate)
        else:
            decay_times[i] = max(1.0 / sampling_rate, float(end - peak_idx - 1) / sampling_rate)

    return amplitudes, rise_times, decay_times
