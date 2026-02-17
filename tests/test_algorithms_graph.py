"""Tests for graph algorithm atoms (Issue 11)."""

import numpy as np
import pytest

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


class TestDFS:
    def test_traverses_graph(self):
        adj = _make_adj()
        result = dfs(adj, source=0)
        assert result[0] == 0

    def test_registered(self):
        assert "dfs" in REGISTRY


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


class TestBellmanFord:
    def test_shortest_paths(self):
        adj = _make_adj()
        dist = bellman_ford(adj, source=0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0

    def test_registered(self):
        assert "bellman_ford" in REGISTRY


class TestFloydWarshall:
    def test_all_pairs(self):
        adj = _make_adj()
        dist = floyd_warshall(adj)
        assert dist.shape == (4, 4)
        assert dist[0, 0] == 0.0
        assert dist[0, 1] == 1.0

    def test_registered(self):
        assert "floyd_warshall" in REGISTRY
