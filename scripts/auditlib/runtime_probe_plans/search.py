"""Runtime probe plans for search and graph-search primitives."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_probe_plans() -> dict[str, Any]:
    from .. import runtime_probes as rt

    ProbeCase = rt.ProbeCase
    ProbePlan = rt.ProbePlan
    _assert_array = rt._assert_array
    _assert_scalar = rt._assert_scalar

    adjacency = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    traversal = np.array([0, 1, 2])
    return {
        "ageoa.algorithms.search.binary_search": ProbePlan(
            positive=ProbeCase(
                "binary search over a sorted vector",
                lambda func: func(np.array([1, 3, 5, 7]), 5),
                _assert_scalar(2),
            ),
            negative=ProbeCase(
                "binary search rejects unsorted input",
                lambda func: func(np.array([3, 1, 2]), 2),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.search.linear_search": ProbePlan(
            positive=ProbeCase(
                "linear search over a small vector",
                lambda func: func(np.array([4, 1, 4]), 1),
                _assert_scalar(1),
            ),
            negative=ProbeCase(
                "linear search rejects empty input",
                lambda func: func(np.array([]), 1),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.search.hash_lookup": ProbePlan(
            positive=ProbeCase(
                "hash lookup over a small vector",
                lambda func: func(np.array([4, 1, 4]), 4),
                _assert_scalar(0),
            ),
            negative=ProbeCase(
                "hash lookup rejects empty input",
                lambda func: func(np.array([]), 1),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.graph.bfs": ProbePlan(
            positive=ProbeCase(
                "breadth-first search on a 3-node graph",
                lambda func: func(adjacency, source=0),
                _assert_array(traversal),
            ),
            negative=ProbeCase(
                "breadth-first search rejects non-square adjacency",
                lambda func: func(np.array([[0.0, 1.0, 0.0]]), source=0),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.graph.dfs": ProbePlan(
            positive=ProbeCase(
                "depth-first search on a 3-node graph",
                lambda func: func(adjacency, source=0),
                _assert_array(traversal),
            ),
            negative=ProbeCase(
                "depth-first search rejects non-square adjacency",
                lambda func: func(np.array([[0.0, 1.0, 0.0]]), source=0),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.graph.dijkstra": ProbePlan(
            positive=ProbeCase(
                "dijkstra shortest paths on a small DAG",
                lambda func: func(adjacency, source=0),
                _assert_array(np.array([0.0, 1.0, 3.0])),
            ),
            negative=ProbeCase(
                "dijkstra rejects negative weights",
                lambda func: func(np.array([[0.0, -1.0], [0.0, 0.0]]), source=0),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.graph.bellman_ford": ProbePlan(
            positive=ProbeCase(
                "bellman-ford shortest paths on a small DAG",
                lambda func: func(adjacency, source=0),
                _assert_array(np.array([0.0, 1.0, 3.0])),
            ),
            negative=ProbeCase(
                "bellman-ford rejects non-square adjacency",
                lambda func: func(np.array([[0.0, 1.0, 0.0]]), source=0),
                expect_exception=True,
            ),
        ),
        "ageoa.algorithms.graph.floyd_warshall": ProbePlan(
            positive=ProbeCase(
                "floyd-warshall all-pairs shortest paths",
                lambda func: func(adjacency),
                _assert_array(
                    np.array(
                        [
                            [0.0, 1.0, 3.0],
                            [np.inf, 0.0, 2.0],
                            [np.inf, np.inf, 0.0],
                        ]
                    )
                ),
            ),
            negative=ProbeCase(
                "floyd-warshall rejects non-square adjacency",
                lambda func: func(np.array([[0.0, 1.0, 0.0]])),
                expect_exception=True,
            ),
        ),
    }
