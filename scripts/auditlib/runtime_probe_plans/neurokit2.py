"""Runtime probe plans for the NeuroKit2 family."""

from __future__ import annotations

import neurokit2 as nk
import numpy as np


def get_probe_plans() -> dict[str, object]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_shape = rt._assert_shape
    _assert_value = rt._assert_value

    signal = nk.ecg_simulate(duration=20, sampling_rate=1000, heart_rate=70, random_state=42)
    rpeaks, _ = nk.ecg_peaks(signal, sampling_rate=1000)
    peaks = np.asarray(rpeaks["ECG_R_Peaks"], dtype=int)
    return {
        "ageoa.neurokit2.averageqrstemplate": ProbePlan(
            positive=ProbeCase(
                "average QRS template returns a signal-length waveform",
                lambda func: func(signal, peaks),
                _assert_shape(signal.shape),
            ),
            negative=ProbeCase(
                "average QRS template rejects a non-numeric sampling rate",
                lambda func: func(signal, peaks, sampling_rate="bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.neurokit2.zhao2018hrvanalysis": ProbePlan(
            positive=ProbeCase(
                "Zhao 2018 quality analysis returns the expected quality label",
                lambda func: func(signal, peaks),
                _assert_value("Barely acceptable"),
            ),
            negative=ProbeCase(
                "Zhao 2018 quality analysis rejects a non-string mode",
                lambda func: func(signal, peaks, mode=None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
