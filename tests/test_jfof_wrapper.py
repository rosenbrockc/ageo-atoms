import numpy as np

from ageoa.jFOF.atoms import find_fof_clusters


def test_find_fof_clusters_uses_upstream_optional_defaults() -> None:
    result = find_fof_clusters(
        np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        ),
        0.2,
        2.0,
    )
    assert result.shape == (3,)
    assert result.dtype.kind in {"i", "u"}
    assert result[0] == result[1]
