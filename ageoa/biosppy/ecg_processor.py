
import numpy as np
import biosppy.signals.ecg as ecg
import biosppy.signals.tools as st

class ECGProcessor:
    def __init__(self, sampling_rate=1000.0):
        self.sampling_rate = sampling_rate
        self.filtered = None
        self.rpeaks = None
        self.rpeaks_corrected = None
        self.templates = None
        self.rpeaks_final = None
        self.hr_idx = None
        self.heart_rate = None

    def filter_signal(self, signal):
        order = int(0.3 * self.sampling_rate)
        filtered, _, _ = st.filter_signal(
            signal=signal,
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=[3, 45],
            sampling_rate=self.sampling_rate,
        )
        self.filtered = filtered

    def detect_rpeaks(self, filtered):
        (rpeaks,) = ecg.hamilton_segmenter(signal=filtered, sampling_rate=self.sampling_rate)
        self.rpeaks = rpeaks

    def correct_peaks(self, filtered, rpeaks):
        (rpeaks_corrected,) = ecg.correct_rpeaks(
            signal=filtered, rpeaks=rpeaks, sampling_rate=self.sampling_rate, tol=0.05
        )
        self.rpeaks_corrected = rpeaks_corrected
        self.rpeaks = rpeaks_corrected

    def extract_templates(self, filtered, rpeaks):
        templates, rpeaks_final = ecg.extract_heartbeats(
            signal=filtered, rpeaks=rpeaks, sampling_rate=self.sampling_rate
        )
        self.templates = templates
        self.rpeaks_final = rpeaks_final
        self.rpeaks = rpeaks_final

    def compute_heart_rate(self, rpeaks):
        ts, heart_rate = ecg.extract_heart_rate(rpeaks=rpeaks, sampling_rate=self.sampling_rate)
        self.hr_idx = ts
        self.heart_rate = heart_rate
