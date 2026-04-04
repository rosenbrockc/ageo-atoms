from __future__ import annotations

import numpy as np

from ageoa.scipy.interpolate_v2.atoms import cubicsplinefit, rbfinterpolatorfit


def test_cubicsplinefit_preserves_optional_defaults() -> None:
    spline = cubicsplinefit(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )

    values = np.asarray(spline(np.array([0.5, 1.5], dtype=float)))
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def test_rbfinterpolatorfit_accepts_default_optional_arguments() -> None:
    interpolator = rbfinterpolatorfit(
        np.array([[0.0], [1.0], [2.0]], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )

    values = np.asarray(interpolator(np.array([[0.5], [1.5]], dtype=float)))
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
