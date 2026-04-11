"""Runtime probe plans for numpy.fft_v2 families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_shape = rt._assert_shape

    signal = np.arange(8, dtype=float).reshape(2, 4)
    spectrum = np.fft.fftn(signal, s=(2, 4), axes=(0, 1), norm="backward")
    return {
        "ageoa.numpy.fft_v2.forwardmultidimensionalfft": ProbePlan(
            positive=ProbeCase(
                "forward N-D FFT over a small 2x4 signal",
                lambda func: func(signal, [2, 4], [0, 1], "backward"),
                _assert_array(spectrum),
            ),
            negative=ProbeCase(
                "reject a missing input array",
                lambda func: func(None, [2, 4], [0, 1], "backward"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft_v2.inversemultidimensionalfft": ProbePlan(
            positive=ProbeCase(
                "inverse N-D FFT reconstructs the original signal",
                lambda func: func(spectrum, [2, 4], [0, 1], "backward"),
                _assert_array(signal.astype(complex)),
            ),
            negative=ProbeCase(
                "reject a missing spectrum input",
                lambda func: func(None, [2, 4], [0, 1], "backward"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft_v2.hermitianspectraltransform": ProbePlan(
            positive=ProbeCase(
                "Hermitian FFT over a symmetric complex spectrum",
                lambda func: func(
                    np.array([1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j]),
                    4,
                    -1,
                    "backward",
                ),
                _assert_shape((4,)),
            ),
            negative=ProbeCase(
                "reject a missing Hermitian input",
                lambda func: func(None, 4, -1, "backward"),
                expect_exception=True,
            ),
        ),
    }
