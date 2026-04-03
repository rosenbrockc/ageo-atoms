import neurokit2 as nk
import numpy as np

from ageoa.neurokit2.atoms import averageqrstemplate, zhao2018hrvanalysis


def _synthetic_signal_and_peaks() -> tuple[np.ndarray, np.ndarray]:
    signal = nk.ecg_simulate(duration=20, sampling_rate=1000, heart_rate=70, random_state=42)
    rpeaks, _ = nk.ecg_peaks(signal, sampling_rate=1000)
    return signal, np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)


def test_averageqrstemplate_uses_upstream_defaults() -> None:
    signal, peaks = _synthetic_signal_and_peaks()
    result = averageqrstemplate(signal, peaks)
    assert isinstance(result, np.ndarray)
    assert result.shape == signal.shape


def test_zhao2018hrvanalysis_uses_upstream_defaults() -> None:
    signal, peaks = _synthetic_signal_and_peaks()
    assert zhao2018hrvanalysis(signal, peaks) == "Barely acceptable"
