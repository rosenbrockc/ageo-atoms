import numpy as np

from ageoa.pronto.dynamic_stance_estimator.atoms import (
    initializefilter,
    predictstep,
    querystance,
    updatestep,
)


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    x0 = np.array([[0.0], [1.0]], dtype=float)
    p0 = np.eye(2, dtype=float)
    A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = 0.1 * np.eye(2, dtype=float)
    R = np.array([[0.5]], dtype=float)
    return initializefilter(x0, p0, A, H, Q, R)


def test_predictstep_propagates_state_and_covariance() -> None:
    state, params = _fixture()

    predicted = predictstep(state, params, 0.1)

    np.testing.assert_allclose(predicted["x"], np.array([[1.0], [1.0]], dtype=float))
    np.testing.assert_allclose(predicted["P"], np.array([[2.1, 1.0], [1.0, 1.1]], dtype=float))


def test_updatestep_and_querystance_produce_expected_scalar() -> None:
    _, params = _fixture()
    predicted = {
        "x": np.array([[1.0], [1.0]], dtype=float),
        "P": np.array([[2.1, 1.0], [1.0, 1.1]], dtype=float),
    }

    updated = updatestep(predicted, params, np.array([[1.2]], dtype=float))

    np.testing.assert_allclose(updated["x"], np.array([[1.1615384615384614], [1.0769230769230769]], dtype=float), atol=1e-9)
    assert querystance(updated) == 1.1615384615384614
