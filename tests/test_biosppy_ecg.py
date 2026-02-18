import numpy as np
import pytest
from ageoa.biosppy.ecg import (
    bandpass_filter,
    r_peak_detection,
    peak_correction,
    template_extraction,
    heart_rate_computation,
)

def generate_synthetic_ecg(duration=10.0, fs=1000.0, heart_rate=75.0):
    """Generate a synthetic ECG signal with periodic peaks and noise."""
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    # Fundamental frequency
    f0 = heart_rate / 60.0
    
    # Simple QRS complex (sum of Gaussian peaks)
    signal = np.zeros_like(t)
    peak_times = np.arange(0.5, duration, 1.0/f0)
    for p in peak_times:
        # R-peak (tall and narrow)
        signal += 1.0 * np.exp(-((t - p)**2) / (2 * (0.01**2)))
        # P-wave (small and wide)
        signal += 0.1 * np.exp(-((t - (p - 0.2))**2) / (2 * (0.05**2)))
        # T-wave (medium and wide)
        signal += 0.3 * np.exp(-((t - (p + 0.3))**2) / (2 * (0.08**2)))
    
    # Add baseline wander (low freq)
    signal += 0.2 * np.sin(2 * np.pi * 0.5 * t)
    
    # Add high-frequency noise
    signal += 0.1 * np.random.normal(size=len(t))
    
    return signal, fs, peak_times

def test_ecg_pipeline():
    fs = 1000.0
    duration = 10.0
    expected_hr = 75.0
    signal, fs, true_peak_times = generate_synthetic_ecg(duration, fs, expected_hr)
    
    # 1. Bandpass filter
    filtered = bandpass_filter(signal, sampling_rate=fs)
    assert len(filtered) == len(signal)
    
    # 2. R-peak detection
    rpeaks = r_peak_detection(filtered, sampling_rate=fs)
    assert len(rpeaks) > 0
    
    # 3. Peak correction
    corrected = peak_correction(filtered, rpeaks, sampling_rate=fs)
    assert len(corrected) == len(rpeaks)
    
    # 4. Template extraction
    templates, rpeaks_extracted = template_extraction(filtered, corrected, sampling_rate=fs)
    assert templates.shape[0] == len(rpeaks_extracted)
    assert templates.shape[1] == int(0.6 * fs)
    
    # 5. Heart rate computation
    indices, hr = heart_rate_computation(corrected, sampling_rate=fs)
    assert len(hr) == len(corrected) - 1
    # Check that computed HR is close to expected (75 bpm)
    mean_hr = np.mean(hr)
    assert mean_hr == pytest.approx(expected_hr, rel=0.1)

def test_icontract_violations():
    # Test invalid inputs
    with pytest.raises(Exception): # icontract.Violation
        bandpass_filter(np.array([[1, 2], [3, 4]]), sampling_rate=1000)
    
    with pytest.raises(Exception):
        r_peak_detection(np.array([1, 2]), sampling_rate=-1.0)
