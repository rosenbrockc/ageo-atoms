"""Runtime probe plans for core SciPy families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_scalar = rt._assert_scalar

    matrix = np.array([[4.0, 2.0], [1.0, 3.0]])
    vector = np.array([1.0, 2.0])
    lu = np.array([[4.0, 2.0], [0.25, 2.5]])
    piv = np.array([0, 1], dtype=np.int32)

    def _linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.asarray(x, dtype=float) + b

    return {
        "ageoa.scipy.fft.dct": ProbePlan(
            positive=ProbeCase(
                "scipy.fft.dct over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0]), norm="ortho"),
                _assert_array(np.array([3.46410162, -1.41421356, 0.0])),
            ),
            negative=ProbeCase(
                "scipy.fft.dct rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.fft.idct": ProbePlan(
            positive=ProbeCase(
                "scipy.fft.idct over a short real vector",
                lambda func: func(np.array([3.46410162, -1.41421356, 0.0]), norm="ortho"),
                _assert_array(np.array([1.0, 2.0, 3.0])),
            ),
            negative=ProbeCase(
                "scipy.fft.idct rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.solve": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.solve over a tiny system",
                lambda func: func(matrix, vector),
                _assert_array(np.array([-0.1, 0.7])),
            ),
            negative=ProbeCase(
                "scipy.linalg.solve rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]]), np.array([1.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.inv": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.inv over a tiny matrix",
                lambda func: func(matrix),
                _assert_array(np.array([[0.3, -0.2], [-0.1, 0.4]])),
            ),
            negative=ProbeCase(
                "scipy.linalg.inv rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.det": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.det over a tiny matrix",
                lambda func: func(matrix),
                _assert_scalar(10.0),
            ),
        ),
        "ageoa.scipy.linalg.lu_factor": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.lu_factor over a tiny matrix",
                lambda func: func(matrix),
                lambda result: (
                    np.testing.assert_allclose(np.asarray(result[0]), lu),
                    np.testing.assert_array_equal(np.asarray(result[1]), piv),
                ),
            ),
            negative=ProbeCase(
                "scipy.linalg.lu_factor rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.lu_solve": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.lu_solve over a tiny factored system",
                lambda func: func((lu, piv), vector),
                _assert_array(np.array([-0.1, 0.7])),
            ),
            negative=ProbeCase(
                "scipy.linalg.lu_solve rejects incompatible RHS",
                lambda func: func((lu, piv), np.array([1.0, 2.0, 3.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.optimize.curve_fit": ProbePlan(
            positive=ProbeCase(
                "scipy.optimize.curve_fit recovers a simple linear model",
                lambda func: func(
                    _linear_model,
                    np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
                    np.array([1.0, 3.0, 5.0, 7.0], dtype=float),
                ),
                lambda result: (
                    np.testing.assert_allclose(
                        np.asarray(result[0]), np.array([2.0, 1.0]), atol=1e-6
                    ),
                    np.testing.assert_equal(np.asarray(result[1]).shape, (2, 2)),
                ),
            ),
            negative=ProbeCase(
                "scipy.optimize.curve_fit rejects mismatched input lengths",
                lambda func: func(
                    _linear_model,
                    np.array([0.0, 1.0], dtype=float),
                    np.array([1.0], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
