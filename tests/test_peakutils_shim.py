from __future__ import annotations

import numpy as np

import peakutils


def test_indexes_respects_threshold_and_distance() -> None:
    signal = np.array([0.0, 1.0, 0.0, 0.8, 0.0, 1.2, 0.0, 0.5, 0.0])

    peaks = peakutils.indexes(signal, thres=0.5, min_dist=2)

    assert peaks.tolist() == [1, 5]


def test_indexes_supports_absolute_threshold() -> None:
    signal = np.array([0.0, 0.4, 0.0, 0.9, 0.0])

    peaks = peakutils.indexes(signal, thres=0.5, thres_abs=True)

    assert peaks.tolist() == [3]
