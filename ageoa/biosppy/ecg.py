"""Auto-generated atom wrappers following the ageoa pattern."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.ghost.registry import register_atom
from ageoa.biosppy.ecg_witnesses import (
    witness_bandpass_filter,
    witness_r_peak_detection,
    witness_peak_correction,
    witness_template_extraction,
    witness_heart_rate_computation,
)

@register_atom(witness_bandpass_filter)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output shape must match input")
def bandpass_filter(signal: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Apply FIR bandpass filter (3-45 Hz) to remove baseline wander and high-frequency noise from the raw ECG signal."""
    lowcut = 3.0
    highcut = 45.0
    nyq = 0.5 * sampling_rate
    
    # BioSPPy uses a FIR filter with order = 0.3 * sampling_rate
    numtaps = int(0.3 * sampling_rate)
    if numtaps % 2 == 0:
        numtaps += 1  # FIR filter with numtaps must be odd for bandpass
        
    b = scipy.signal.firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=sampling_rate)
    # Zero-phase filtering for ECG
    return scipy.signal.filtfilt(b, [1.0], signal)

@register_atom(witness_r_peak_detection)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: result.ndim == 1, "R-peak indices must be 1D")
def r_peak_detection(filtered: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Detect R-peak locations in the filtered ECG signal using a robust implementation of the Hamilton segmenter algorithm."""
    # 1. Differentiation
    diff = np.diff(filtered)
    # 2. Absolute value
    abs_diff = np.abs(diff)
    # 3. Moving average (80ms window)
    window_size = int(0.08 * sampling_rate)
    if window_size < 1:
        window_size = 1
    ma = scipy.signal.lfilter(np.ones(window_size) / window_size, 1, abs_diff)
    
    # 4. Thresholding
    # We use a dynamic threshold based on the maximum of the moving average
    # to better distinguish R-peaks from T-waves and noise.
    threshold = 0.5 * np.max(ma)
    
    # 5. Peak detection in MA signal
    # Min distance between R-peaks is typically 200ms
    min_dist = int(0.2 * sampling_rate)
    peaks, _ = scipy.signal.find_peaks(ma, height=threshold, distance=min_dist)
    
    # 6. Refine peaks in the filtered signal
    # For each peak in MA, look for the actual maximum in the filtered signal
    # within a small window around it.
    refined_peaks = []
    search_window = int(0.05 * sampling_rate) # 50ms search window
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(filtered), p + search_window)
        if start < end:
            # We look for the maximum amplitude in the original filtered signal
            # BioSPPy Hamilton refined logic
            peak_idx = start + np.argmax(np.abs(filtered[start:end]))
            refined_peaks.append(peak_idx)
            
    return np.unique(np.array(refined_peaks, dtype=np.int64))

@register_atom(witness_peak_correction)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.ensure(lambda result, rpeaks: result.shape == rpeaks.shape, "Output shape must match input rpeaks")
def peak_correction(filtered: np.ndarray, rpeaks: np.ndarray, sampling_rate: float = 1000.0) -> np.ndarray:
    """Correct R-peak locations to the nearest local maximum within a tolerance window (50ms)."""
    search_window = int(0.05 * sampling_rate)
    corrected_peaks = []
    
    for r in rpeaks:
        start = max(0, int(r) - search_window)
        end = min(len(filtered), int(r) + search_window)
        if start < end:
            peak_idx = start + np.argmax(np.abs(filtered[start:end]))
            corrected_peaks.append(peak_idx)
        else:
            corrected_peaks.append(r)
            
    return np.array(corrected_peaks, dtype=np.int64)

@register_atom(witness_template_extraction)
@icontract.require(lambda filtered: filtered.ndim == 1, "Filtered signal must be 1D")
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (templates, rpeaks)")
def template_extraction(
    filtered: np.ndarray, 
    rpeaks: np.ndarray, 
    sampling_rate: float = 1000.0,
    before: float = 0.2, 
    after: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Extract individual heartbeat waveform templates around each R-peak with configurable before/after windows."""
    n_before = int(before * sampling_rate)
    n_after = int(after * sampling_rate)
    
    templates = []
    valid_rpeaks = []
    
    for r in rpeaks:
        start = int(r) - n_before
        end = int(r) + n_after
        if start >= 0 and end <= len(filtered):
            templates.append(filtered[start:end])
            valid_rpeaks.append(r)
            
    if not templates:
        return np.empty((0, n_before + n_after)), np.empty(0, dtype=np.int64)
        
    return np.array(templates), np.array(valid_rpeaks, dtype=np.int64)

@register_atom(witness_heart_rate_computation)
@icontract.require(lambda rpeaks: rpeaks.ndim == 1, "R-peaks must be 1D")
@icontract.require(lambda rpeaks: len(rpeaks) >= 2, "At least 2 R-peaks are required to compute heart rate")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (index, heart_rate)")
def heart_rate_computation(rpeaks: np.ndarray, sampling_rate: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous heart rate in bpm from R-R intervals."""
    # R-R intervals in samples
    rr_intervals = np.diff(rpeaks)
    
    # Convert to seconds
    rr_seconds = rr_intervals / sampling_rate
    
    # Compute heart rate in bpm
    # To avoid division by zero
    rr_seconds[rr_seconds == 0] = np.nan
    heart_rate = 60.0 / rr_seconds
    
    # The indices for heart rate are usually the R-peak locations starting from the second one
    indices = rpeaks[1:]
    
    return indices, heart_rate
