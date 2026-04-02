from __future__ import annotations

import numpy as np
import pytest
from biosppy.signals.tools import OnlineFilter
from auditlib import runtime_probes

_online_filter_atoms = runtime_probes.safe_import_module("ageoa.biosppy.online_filter.atoms")
_online_filter_codex_atoms = runtime_probes.safe_import_module("ageoa.biosppy.online_filter_codex.atoms")
_online_filter_v2_atoms = runtime_probes.safe_import_module("ageoa.biosppy.online_filter_v2.atoms")

filterstateinit = _online_filter_atoms.filterstateinit
filterstep = _online_filter_atoms.filterstep
codex_filterstateinit = _online_filter_codex_atoms.filterstateinit
codex_filterstep = _online_filter_codex_atoms.filterstep
v2_filterstateinit = _online_filter_v2_atoms.filterstateinit
v2_filterstep = _online_filter_v2_atoms.filterstep


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


@pytest.mark.parametrize(
    ("init_atom", "step_atom"),
    [
        (codex_filterstateinit, codex_filterstep),
        (v2_filterstateinit, v2_filterstep),
    ],
)
def test_refined_online_filter_variants_match_chunked_execution(init_atom, step_atom) -> None:
    b = np.array([0.5, 0.5], dtype=float)
    a = np.array([1.0], dtype=float)
    signal = np.linspace(0.0, 1.0, 8)

    (_, _, zi0), state0 = init_atom(b, a)
    assert zi0 is None

    (filtered_1, zi1), state1 = step_atom(signal[:4], state0)
    (filtered_2, zi2), _ = step_atom(signal[4:], state1)

    assert zi1 is not None
    assert zi2 is not None

    chunked = np.concatenate([filtered_1, filtered_2])
    expected = np.convolve(signal, b, mode="full")[: signal.shape[0]]

    assert np.allclose(chunked, expected)


@pytest.mark.parametrize(
    ("init_atom",),
    [
        (filterstateinit,),
        (codex_filterstateinit,),
        (v2_filterstateinit,),
    ],
)
def test_online_filter_state_init_matches_upstream_optional_signature(init_atom) -> None:
    with pytest.raises(TypeError):
        init_atom()


@pytest.mark.parametrize(
    ("init_atom", "step_atom"),
    [
        (filterstateinit, filterstep),
        (codex_filterstateinit, codex_filterstep),
        (v2_filterstateinit, v2_filterstep),
    ],
)
def test_online_filter_step_matches_upstream_optional_signal_signature(init_atom, step_atom) -> None:
    (_, _, _), state = init_atom(np.array([1.0], dtype=float), np.array([1.0], dtype=float))
    with pytest.raises(TypeError):
        step_atom(state=state)
