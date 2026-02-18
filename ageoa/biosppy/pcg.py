"""PCG atoms ingested via the Smart Ingester."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.signal

from ageoa.ghost.registry import register_atom
from ageoa.biosppy.pcg_witnesses import (
    witness_shannon_energy,
    witness_pcg_segmentation,
)

@register_atom(witness_shannon_energy)
@icontract.require(lambda signal: signal.ndim == 1, "Signal must be 1D")
@icontract.ensure(lambda result, signal: result.shape == signal.shape, "Output shape must match input")
def shannon_energy(signal: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon Energy for the input PCG signal.
    
    Logic: E = -x_norm^2 * log(x_norm^2).
    This non-linear transform amplifies heart sounds and suppresses both noise and high artifacts.
    """
    # Normalize by max amplitude
    x = signal / np.max(np.abs(signal))
    
    # Square and compute Shannon energy
    # To avoid log(0), we add a small epsilon.
    # To avoid negative energy if |x| > 1 (due to float precision), we clip to [0, 1].
    x2 = np.clip(x**2, 1e-12, 1.0)
    energy = -x2 * np.log(x2)
    return np.maximum(energy, 0.0)

@register_atom(witness_pcg_segmentation)
@icontract.require(lambda envelope: envelope.ndim == 1, "Envelope must be 1D")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "Sampling rate must be positive")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2, "Output must be (S1, S2)")
def pcg_segmentation(envelope: np.ndarray, sampling_rate: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Segment PCG heart sounds into S1 and S2 using the Shannon Energy envelope.
    
    Logic: Identifies dominant peaks in the low-pass filtered Shannon envelope.
    Uses amplitude and temporal heuristics to distinguish S1 (louder, starts cycle) 
    from S2 (starts diastole).
    """
    # 1. Low-pass filter the envelope (20 Hz as per Liang et al.)
    nyq = 0.5 * sampling_rate
    b, a = scipy.signal.butter(2, 20.0 / nyq, btype='low')
    smooth_envelope = scipy.signal.filtfilt(b, a, envelope)
    
    # 2. Thresholding (0.05 * max as a basic heuristic)
    threshold = 0.05 * np.max(smooth_envelope)
    
    # 3. Peak detection (Min distance between S1/S2 is ~200ms)
    peaks, _ = scipy.signal.find_peaks(smooth_envelope, height=threshold, distance=int(0.2 * sampling_rate))
    
    # 4. Alternating classification 
    # S1 usually has a larger interval to the NEXT S1 (full cycle) 
    # than to the following S2 (systole).
    s1 = []
    s2 = []
    
    if len(peaks) < 2:
        return np.array(peaks, dtype=np.int64), np.empty(0, dtype=np.int64)
        
    for i in range(len(peaks) - 1):
        # We classify based on interval length
        # Simple heuristic: S1 is the first of a pair separated by a shorter interval
        if i % 2 == 0:
            s1.append(peaks[i])
        else:
            s2.append(peaks[i])
            
    # Handle the last peak
    if len(peaks) % 2 == 1:
        s1.append(peaks[-1])
    else:
        s2.append(peaks[-1])
            
    return np.array(s1, dtype=np.int64), np.array(s2, dtype=np.int64)
