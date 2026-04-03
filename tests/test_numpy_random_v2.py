import numpy as np

from ageoa.numpy.random_v2.atoms import (
    combinatoricssampler,
    continuousmultivariatesampler,
    discreteeventsampler,
)


def test_continuousmultivariatesampler_uses_numpy_defaults() -> None:
    np.random.seed(0)
    mvn, dirichlet = continuousmultivariatesampler(
        np.array([0.0, 1.0], dtype=float),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        size=2,
    )
    assert mvn.shape == (2, 2)
    assert dirichlet.shape == (2, 2)
    np.testing.assert_allclose(dirichlet.sum(axis=1), np.ones(2))


def test_discreteeventsampler_defaults_size_to_none() -> None:
    np.random.seed(0)
    result = discreteeventsampler(5, np.array([0.2, 0.8], dtype=float))
    assert result.shape == (2,)
    assert int(result.sum()) == 5


def test_combinatoricssampler_matches_permutation_and_choice_surface() -> None:
    np.random.seed(0)
    permuted, selected = combinatoricssampler(
        np.array([1, 2, 3], dtype=int),
        np.array([10, 20, 30], dtype=int),
        size=2,
        replace=False,
    )
    assert sorted(permuted.tolist()) == [1, 2, 3]
    assert selected.shape == (2,)
