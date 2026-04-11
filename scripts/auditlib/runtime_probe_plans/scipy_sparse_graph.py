"""Runtime probe plans for scipy.sparse_graph families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array

    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    signal = np.array([1.0, 0.0])
    eigenvectors = np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]])
    x_hat = np.array([-0.70710678, -0.70710678])
    return {
        "ageoa.scipy.sparse_graph.graph_laplacian": ProbePlan(
            positive=ProbeCase(
                "graph Laplacian over a symmetric 2-node graph",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(weights)).toarray(),
                _assert_array(laplacian),
            ),
            negative=ProbeCase(
                "graph Laplacian rejects asymmetric weights",
                lambda func: func(
                    __import__("scipy.sparse").sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.graph_fourier_transform": ProbePlan(
            positive=ProbeCase(
                "graph Fourier transform on a 2-node Laplacian",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal),
                lambda result: (
                    np.testing.assert_allclose(np.asarray(result[0]), x_hat, atol=1e-6),
                    np.testing.assert_allclose(np.asarray(result[1]), np.array([0.0, 2.0]), atol=1e-6),
                    np.testing.assert_allclose(np.abs(np.asarray(result[2])), np.abs(eigenvectors), atol=1e-6),
                ),
            ),
            negative=ProbeCase(
                "graph Fourier transform rejects mismatched signal length",
                lambda func: func(
                    __import__("scipy.sparse").sparse.csr_matrix(laplacian),
                    np.array([1.0, 0.0, 2.0]),
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.inverse_graph_fourier_transform": ProbePlan(
            positive=ProbeCase(
                "inverse graph Fourier transform on a 2-node basis",
                lambda func: func(x_hat, eigenvectors),
                _assert_array(signal, atol=1e-6),
            ),
            negative=ProbeCase(
                "inverse graph Fourier transform rejects mismatched coefficient count",
                lambda func: func(np.array([1.0]), eigenvectors),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.heat_kernel_diffusion": ProbePlan(
            positive=ProbeCase(
                "heat kernel diffusion smooths a 2-node signal",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal, 0.5),
                _assert_array(np.array([0.68393972, 0.31606028]), atol=1e-6),
            ),
            negative=ProbeCase(
                "heat kernel diffusion rejects negative diffusion time",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal, -0.5),
                expect_exception=True,
            ),
        ),
    }
