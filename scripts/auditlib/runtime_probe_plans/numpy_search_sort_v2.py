"""Runtime probe plans for numpy.search_sort_v2 families."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array

    def _assert_partition_result(expected_kth_value: int) -> Callable[[Any], None]:
        def _validator(result: Any) -> None:
            assert isinstance(result, tuple)
            assert len(result) == 2
            partitioned = np.asarray(result[0])
            partition_indices = np.asarray(result[1])
            assert partitioned.shape == (4,)
            assert partition_indices.shape == (4,)
            np.testing.assert_equal(partitioned[2], expected_kth_value)

        return _validator

    return {
        "ageoa.numpy.search_sort_v2.binarysearchinsertion": ProbePlan(
            positive=ProbeCase(
                "searchsorted returns deterministic insertion points for a sorted array",
                lambda func: func(np.array([1, 3, 5]), np.array([0, 3, 4, 6]), side="left"),
                _assert_array(np.array([0, 1, 2, 3])),
            ),
            negative=ProbeCase(
                "searchsorted rejects a missing sorted array",
                lambda func: func(None, np.array([1])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.numpy.search_sort_v2.lexicographicindirectsort": ProbePlan(
            positive=ProbeCase(
                "lexsort returns a deterministic indirect ordering for two key arrays",
                lambda func: func((np.array([2, 1, 2]), np.array([1, 2, 0]))),
                _assert_array(np.array([2, 0, 1])),
            ),
            negative=ProbeCase(
                "lexsort rejects a missing key sequence",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.numpy.search_sort_v2.partialsortpartition": ProbePlan(
            positive=ProbeCase(
                "partition and argpartition agree on the median pivot location",
                lambda func: func(np.array([4, 1, 3, 2]), 2),
                _assert_partition_result(3),
            ),
            negative=ProbeCase(
                "partition rejects a missing kth selector",
                lambda func: func(np.array([1, 2, 3]), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
