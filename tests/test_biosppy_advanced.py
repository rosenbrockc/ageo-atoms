import numpy as np
import pytest
from ageoa.biosppy.ecg import ssf_segmenter, christov_segmenter
from ageoa.biosppy.pcg import shannon_energy, pcg_segmentation
from ageoa.biosppy.eda import gamboa_segmenter, eda_feature_extraction


def generate_synthetic_ecg(duration=10.0, fs=1000.0, heart_rate=75.0):
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    f0 = heart_rate / 60.0
    signal = np.zeros_like(t)
    peak_times = np.arange(0.5, duration, 1.0/f0)
    for p in peak_times:
        # Stronger R-peak
        signal += 1.5 * np.exp(-((t - p)**2) / (2 * (0.005**2)))
    # Add some lower amplitude noise
    signal += 0.05 * np.random.normal(size=len(t))
    return signal, fs


class TestSSFSegmenter:
    def test_detects_peaks(self):
        signal, fs = generate_synthetic_ecg()
        rpeaks = ssf_segmenter(signal, sampling_rate=fs)
        assert len(rpeaks) > 0
        # Average HR should be ~75 bpm
        rr_intervals = np.diff(rpeaks) / fs
        mean_hr = 60.0 / np.mean(rr_intervals)
        assert mean_hr == pytest.approx(75.0, rel=0.1)


class TestChristovSegmenter:
    def test_detects_peaks(self):
        signal, fs = generate_synthetic_ecg()
        rpeaks = christov_segmenter(signal, sampling_rate=fs)
        assert len(rpeaks) > 0
        rr_intervals = np.diff(rpeaks) / fs
        mean_hr = 60.0 / np.mean(rr_intervals)
        assert mean_hr == pytest.approx(75.0, rel=0.3) # Christov is more sensitive


class TestPCGPipeline:
    def test_pcg_segmentation(self):
        # Generate synthetic PCG (S1/S2 bursts)
        fs = 1000.0
        t = np.linspace(0, 10.0, 10000)
        signal = np.zeros_like(t)
        for p in np.arange(0.5, 10.0, 0.8): # 0.8s heart cycle
            # S1 (loud, 50-100Hz)
            signal += 1.0 * np.exp(-((t - p)**2) / (2 * (0.03**2))) * np.sin(2 * np.pi * 70 * t)
            # S2 (less loud, 100-150Hz)
            signal += 0.7 * np.exp(-((t - (p + 0.3))**2) / (2 * (0.03**2))) * np.sin(2 * np.pi * 120 * t)

        signal += 0.1 * np.random.normal(size=len(t))

        envelope = shannon_energy(signal)
        assert len(envelope) == len(signal)
        assert np.all(envelope >= 0)

        s1, s2 = pcg_segmentation(envelope, sampling_rate=fs)
        assert len(s1) > 0
        assert len(s2) > 0
        # S1 and S2 should be roughly equal in count
        assert abs(len(s1) - len(s2)) <= 1


class TestEDAPipeline:
    def test_eda_feature_extraction(self):
        # Generate synthetic EDA (Slow baseline + SCRs)
        fs = 100.0
        t = np.linspace(0, 30.0, 3000)
        signal = 5.0 + 0.05 * t # Slow drift
        # Add SCRs (classic rise and slow decay)
        scr_onsets = [5.0, 15.0, 25.0]
        for onset in scr_onsets:
            mask = t >= onset
            dt = t[mask] - onset
            scr = 1.0 * (dt / 2.0) * np.exp(-dt / 2.0)
            signal[mask] += scr

        signal += 0.01 * np.random.normal(size=len(t))

        onsets = gamboa_segmenter(signal, sampling_rate=fs)
        assert len(onsets) == 3

        amps, rise, decay = eda_feature_extraction(signal, onsets, sampling_rate=fs)
        assert len(amps) == 3
        assert np.all(amps > 0.1) # SCRs should be detected
        assert np.all(rise > 0)
        assert np.all(decay > 0)
