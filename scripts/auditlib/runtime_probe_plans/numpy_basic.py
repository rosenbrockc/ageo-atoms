"""Runtime probe plans for core NumPy families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_scalar = rt._assert_scalar

    plans = {
        "ageoa.numpy.arrays.array": ProbePlan(
            positive=ProbeCase(
                "numpy.array over a short Python list",
                lambda func: func([1, 2, 3]),
                _assert_array(np.array([1, 2, 3])),
            ),
        ),
        "ageoa.numpy.arrays.zeros": ProbePlan(
            positive=ProbeCase(
                "numpy.zeros over a tiny shape",
                lambda func: func((2, 2)),
                _assert_array(np.zeros((2, 2))),
            ),
            negative=ProbeCase(
                "numpy.zeros rejects an invalid shape type",
                lambda func: func("bad-shape"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.dot": ProbePlan(
            positive=ProbeCase(
                "numpy.dot over short vectors",
                lambda func: func(np.array([1, 2]), np.array([3, 4])),
                _assert_scalar(11),
            ),
            negative=ProbeCase(
                "numpy.dot rejects incompatible dimensions",
                lambda func: func(np.array([1, 2]), np.array([1, 2, 3])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.vstack": ProbePlan(
            positive=ProbeCase(
                "numpy.vstack over two rows",
                lambda func: func([np.array([1, 2]), np.array([3, 4])]),
                _assert_array(np.array([[1, 2], [3, 4]])),
            ),
            negative=ProbeCase(
                "numpy.vstack rejects an empty tuple",
                lambda func: func([]),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.reshape": ProbePlan(
            positive=ProbeCase(
                "numpy.reshape over a 1D vector",
                lambda func: func(np.arange(6), (2, 3)),
                _assert_array(np.arange(6).reshape(2, 3)),
            ),
            negative=ProbeCase(
                "numpy.reshape rejects a missing array",
                lambda func: func(None, (2, 1)),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.emath.sqrt": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.sqrt over positive inputs",
                lambda func: func(np.array([1.0, 4.0, 9.0])),
                _assert_array(np.array([1.0, 2.0, 3.0])),
            ),
        ),
        "ageoa.numpy.emath.log": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.log over positive inputs",
                lambda func: func(np.array([1.0, np.e, np.e**2])),
                _assert_array(np.array([0.0, 1.0, 2.0])),
            ),
        ),
        "ageoa.numpy.emath.log10": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.log10 over powers of ten",
                lambda func: func(np.array([1.0, 10.0, 100.0])),
                _assert_array(np.array([0.0, 1.0, 2.0])),
            ),
        ),
        "ageoa.numpy.emath.power": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.power over a small vector",
                lambda func: func(np.array([2.0, 3.0]), np.array([3.0, 2.0])),
                _assert_array(np.array([8.0, 9.0])),
            ),
        ),
        "ageoa.numpy.fft.fft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fft over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0])),
                _assert_array(np.fft.fft(np.array([1.0, 2.0, 3.0]))),
            ),
            negative=ProbeCase(
                "numpy.fft.fft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.ifft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.ifft over a short complex spectrum",
                lambda func: func(np.fft.fft(np.array([1.0, 2.0, 3.0]))),
                _assert_array(np.array([1.0, 2.0, 3.0]) + 0j),
            ),
            negative=ProbeCase(
                "numpy.fft.ifft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.rfft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.rfft over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0, 4.0])),
                _assert_array(np.fft.rfft(np.array([1.0, 2.0, 3.0, 4.0]))),
            ),
            negative=ProbeCase(
                "numpy.fft.rfft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.irfft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.irfft over a short Hermitian spectrum",
                lambda func: func(np.fft.rfft(np.array([1.0, 2.0, 3.0, 4.0]))),
                _assert_array(np.array([1.0, 2.0, 3.0, 4.0])),
            ),
            negative=ProbeCase(
                "numpy.fft.irfft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.fftfreq": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fftfreq over a short window",
                lambda func: func(4, d=0.5),
                _assert_array(np.array([0.0, 0.5, -1.0, -0.5])),
            ),
            negative=ProbeCase(
                "numpy.fft.fftfreq rejects non-positive n",
                lambda func: func(0),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.fftshift": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fftshift over a short vector",
                lambda func: func(np.array([0, 1, 2, 3])),
                _assert_array(np.array([2, 3, 0, 1])),
            ),
            negative=ProbeCase(
                "numpy.fft.fftshift rejects None",
                lambda func: func(None),
                expect_exception=True,
            ),
        ),
    }
    return plans
