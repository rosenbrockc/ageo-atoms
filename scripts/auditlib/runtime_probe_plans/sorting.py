"""Runtime probe plans for sorting families."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_sorted_array = rt._assert_sorted_array

    base = np.array([4, 1, 3, 2], dtype=np.int64)
    return {
        "ageoa.algorithms.sorting.merge_sort": ProbePlan(
            positive=ProbeCase(
                "merge sort over a short integer vector",
                lambda func: func(base),
                _assert_sorted_array(np.array([1, 2, 3, 4], dtype=np.int64)),
            ),
            negative=ProbeCase(
                "merge sort rejects empty input",
                lambda func: func(np.array([], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.algorithms.sorting.quicksort": ProbePlan(
            positive=ProbeCase(
                "quicksort over a short integer vector",
                lambda func: func(base),
                _assert_sorted_array(np.array([1, 2, 3, 4], dtype=np.int64)),
            ),
            negative=ProbeCase(
                "quicksort rejects empty input",
                lambda func: func(np.array([], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.algorithms.sorting.heapsort": ProbePlan(
            positive=ProbeCase(
                "heapsort over a short integer vector",
                lambda func: func(base),
                _assert_sorted_array(np.array([1, 2, 3, 4], dtype=np.int64)),
            ),
            negative=ProbeCase(
                "heapsort rejects empty input",
                lambda func: func(np.array([], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.algorithms.sorting.counting_sort": ProbePlan(
            positive=ProbeCase(
                "counting sort over non-negative integers",
                lambda func: func(base),
                _assert_sorted_array(np.array([1, 2, 3, 4], dtype=np.int64)),
            ),
            negative=ProbeCase(
                "counting sort rejects floating input",
                lambda func: func(np.array([1.0, 2.0])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.algorithms.sorting.radix_sort": ProbePlan(
            positive=ProbeCase(
                "radix sort over non-negative integers",
                lambda func: func(base),
                _assert_sorted_array(np.array([1, 2, 3, 4], dtype=np.int64)),
            ),
            negative=ProbeCase(
                "radix sort rejects negative integers",
                lambda func: func(np.array([-1, 2], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }
