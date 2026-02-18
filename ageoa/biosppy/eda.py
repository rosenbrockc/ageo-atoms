"""EDA atoms ingested via the Smart Ingester."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.ghost.registry import register_atom
from ageoa.biosppy.eda_witnesses import (
    witness_gamboa_segmenter,
    witness_eda_feature_extraction,
)

@register_atom(witness_gamboa_segmenter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "Onset indices must be 1D")
def gamboa_segmenter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect phasic EDA responses using Gamboa's segmenter (SCR detection).
    
    Logic: Uses the first derivative to find peaks of sweat responses.
    """
    # 1. Low-pass filter (2Hz)
    nyq = 0.5 * sampling_rate
    b, a = scipy.signal.butter(2, 2.0 / nyq, btype='low')
    filtered = scipy.signal.filtfilt(b, a, signal)
    
    # 2. Derivative
    diff = np.diff(filtered)
    
    # 3. Peak detection on derivative (finding the fastest rise)
    threshold = 0.5 * np.max(diff)
    peaks, _ = scipy.signal.find_peaks(diff, height=threshold, distance=int(5.0 * sampling_rate))
    
    # 4. Search backward for onset (where diff was close to 0)
    onsets = []
    for p in peaks:
        onset = p
        while onset > 0 and diff[onset] > 0.001:
            onset -= 1
        onsets.append(onset)
                    
    return np.unique(np.array(onsets, dtype=np.int64))

@register_atom(witness_eda_feature_extraction)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda onsets: onsets.ndim == 1, "Onsets must be 1D")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 3, "Output must be (amplitudes, rise_times, decay_times)")
def eda_feature_extraction(signal: np.ndarray, onsets: np.ndarray, sampling_rate: float = 1000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract amplitudes, rise times, and decay times for detected EDA phasic responses."""
    amplitudes = []
    rise_times = []
    decay_times = []
    
    for o in onsets:
        # 1. Find peak (next local max)
        search_window = int(5.0 * sampling_rate)
        end = min(len(signal), o + search_window)
        if o < end:
            peak_idx = o + np.argmax(signal[o:end])
            amp = signal[peak_idx] - signal[o]
            amplitudes.append(amp)
            rise_times.append((peak_idx - o) / sampling_rate)
            decay_times.append(1.0) # Placeholder
        else:
            amplitudes.append(0.0)
            rise_times.append(0.0)
            decay_times.append(0.0)
            
    return np.array(amplitudes), np.array(rise_times), np.array(decay_times)
