"""Tests for graph algorithm atoms (Issue 11)."""

import numpy as np
import pytest
import icontract

from ageoa.algorithms.graph import (
    bellman_ford,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
)
from ageoa.ghost.abstract import AbstractArray
from ageoa.ghost.registry import REGISTRY


def _make_adj():
    """Simple 4-node directed graph."""
    adj = np.zeros((4, 4))
    adj[0, 1] = 1
    adj[0, 2] = 4
    adj[1, 2] = 2
    adj[2, 3] = 1
    return adj


class TestBFS:
    def test_traverses_graph(self):
        adj = _make_adj()
        result = bfs(adj, source=0)
        # Node 0 should be first in BFS order
        assert result[0] == 0

    def test_registered(self):
        assert "bfs" in REGISTRY

    def test_single_node(self):
        adj = np.zeros((1, 1))
        result = bfs(adj, source=0)
        assert result[0] == 0
        assert result.shape == (1,)

    def test_disconnected_graph(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = 1
        # Node 2 is disconnected
        result = bfs(adj, source=0)
        assert result[0] == 0
        assert result[2] == -1  # unreachable

    def test_precondition_non_square(self):
        with pytest.raises(icontract.ViolationError):
            bfs(np.zeros((2, 3)), source=0)


class TestDFS:
    def test_traverses_graph(self):
        adj = _make_adj()
        result = dfs(adj, source=0)
        assert result[0] == 0

    def test_registered(self):
        assert "dfs" in REGISTRY

    def test_single_node(self):
        adj = np.zeros((1, 1))
        result = dfs(adj, source=0)
        assert result[0] == 0

    def test_precondition_non_square(self):
        with pytest.raises(icontract.ViolationError):
            dfs(np.zeros((3, 2)), source=0)


class TestDijkstra:
    def test_shortest_paths(self):
        adj = _make_adj()
        dist = dijkstra(adj, source=0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0  # direct edge
        assert dist[2] == 3.0  # 0->1->2 = 1+2=3 < 0->2=4
        assert dist[3] == 4.0  # 0->1->2->3 = 1+2+1=4

    def test_registered(self):
        assert "dijkstra" in REGISTRY

    def test_single_node(self):
        adj = np.zeros((1, 1))
        dist = dijkstra(adj, source=0)
        assert dist[0] == 0.0

    def test_precondition_negative_weight(self):
        adj = np.array([[0, -1], [0, 0]], dtype=float)
        with pytest.raises(icontract.ViolationError):
            dijkstra(adj, source=0)

    def test_precondition_non_square(self):
        with pytest.raises(icontract.ViolationError):
            dijkstra(np.zeros((2, 3)), source=0)


class TestBellmanFord:
    def test_shortest_paths(self):
        adj = _make_adj()
        dist = bellman_ford(adj, source=0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0

    def test_registered(self):
        assert "bellman_ford" in REGISTRY

    def test_single_node(self):
        adj = np.zeros((1, 1))
        dist = bellman_ford(adj, source=0)
        assert dist[0] == 0.0

    def test_precondition_non_square(self):
        with pytest.raises(icontract.ViolationError):
            bellman_ford(np.zeros((2, 3)), source=0)


class TestFloydWarshall:
    def test_all_pairs(self):
        adj = _make_adj()
        dist = floyd_warshall(adj)
        assert dist.shape == (4, 4)
        assert dist[0, 0] == 0.0
        assert dist[0, 1] == 1.0

    def test_registered(self):
        assert "floyd_warshall" in REGISTRY

    def test_single_node(self):
        adj = np.zeros((1, 1))
        dist = floyd_warshall(adj)
        assert dist.shape == (1, 1)
        assert dist[0, 0] == 0.0

    def test_disconnected_graph(self):
        adj = np.zeros((3, 3))
        adj[0, 1] = 1
        dist = floyd_warshall(adj)
        assert dist[0, 1] == 1.0
        # Node 2 is disconnected, distance should be inf
        assert np.isinf(dist[0, 2])

    def test_precondition_non_square(self):
        with pytest.raises(icontract.ViolationError):
            floyd_warshall(np.zeros((2, 3)))
