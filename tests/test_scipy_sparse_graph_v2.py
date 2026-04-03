import numpy as np

from ageoa.scipy.sparse_graph_v2.atoms import (
    allpairsshortestpath,
    minimumspanningtree,
    singlesourceshortestpath,
)


def _graph() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 2.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=float,
    )


def test_singlesourceshortestpath_matches_scipy_defaults() -> None:
    result = singlesourceshortestpath(_graph(), indices=0)
    np.testing.assert_allclose(result, np.array([0.0, 1.0, 3.0], dtype=float))


def test_allpairsshortestpath_supports_overwrite_defaulted_parameter() -> None:
    result = allpairsshortestpath(_graph())
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 2.0],
                [3.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_minimumspanningtree_defaults_overwrite_to_false() -> None:
    result = minimumspanningtree(_graph()).toarray()
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
