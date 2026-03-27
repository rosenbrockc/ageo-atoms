from __future__ import annotations

import numpy as np
from biosppy.signals.tools import OnlineFilter

from ageoa.biosppy.online_filter.atoms import filterstateinit, filterstep


def test_online_filter_state_init_and_chunked_filter_match_direct_execution() -> None:
    b = np.array([0.5, 0.5], dtype=float)
    a = np.array([1.0], dtype=float)
    signal = np.linspace(0.0, 1.0, 8)

    (_, _, zi0), state0 = filterstateinit(b, a)
    assert zi0 is None

    (filtered_1, zi1), state1 = filterstep(signal[:4], state0)
    assert zi1 is not None

    (filtered_2, zi2), state2 = filterstep(signal[4:], state1)
    assert zi2 is not None
    assert state2.zi is not None

    chunked = np.concatenate([filtered_1, filtered_2])
    expected = np.convolve(signal, b, mode="full")[: signal.shape[0]]

    assert np.allclose(chunked, expected)


def test_online_filter_state_threads_across_chunks() -> None:
    b = np.array([1.0, -0.5], dtype=float)
    a = np.array([1.0], dtype=float)
    signal = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    (_, _, _), state0 = filterstateinit(b, a)
    (filtered_1, _), state1 = filterstep(signal[:2], state0)
    (filtered_2, _), _ = filterstep(signal[2:], state1)

    upstream = OnlineFilter(b=b, a=a)
    expected_1 = np.asarray(upstream.filter(signal=signal[:2])["filtered"], dtype=float)
    expected_2 = np.asarray(upstream.filter(signal=signal[2:])["filtered"], dtype=float)

    assert np.allclose(filtered_1, expected_1)
    assert np.allclose(filtered_2, expected_2)
