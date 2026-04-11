"""Reusable assertion helpers for runtime probe validators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp


def _assert_scalar(expected: float | int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        if isinstance(expected, float):
            assert np.isclose(float(result), expected)
        else:
            assert int(result) == expected

    return _validator


def _assert_array(expected: np.ndarray, *, atol: float = 1e-8) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        np.testing.assert_allclose(np.asarray(result), expected, atol=atol)

    return _validator


def _assert_sorted_array(expected: np.ndarray) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        np.testing.assert_array_equal(np.asarray(result), expected)

    return _validator


def _assert_monotonic_index_array(*, max_value: int | None = None) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray(result)
        assert values.ndim == 1
        if values.size:
            assert np.all(np.diff(values) >= 0)
            assert np.all(values >= 0)
            if max_value is not None:
                assert np.all(values <= max_value)

    return _validator


def _assert_sparse_shape(expected_shape: tuple[int, int]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert sp.issparse(result)
        assert tuple(result.shape) == expected_shape

    return _validator


def _assert_shape(expected_shape: tuple[int, ...]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert tuple(np.asarray(result).shape) == expected_shape

    return _validator


def _assert_pair_of_arrays() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        first = np.asarray(result[0])
        second = np.asarray(result[1])
        assert first.ndim >= 1
        assert second.ndim == 1
        if first.ndim >= 2:
            assert first.shape[0] == second.shape[0]

    return _validator


def _assert_pair_of_sorted_integer_arrays() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        for values in result:
            arr = np.asarray(values)
            assert arr.ndim == 1
            assert np.issubdtype(arr.dtype, np.integer)
            if arr.size:
                assert np.all(np.diff(arr) > 0)
                assert np.all(arr >= 0)

    return _validator


def _assert_online_filter_init_state() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        init_tuple, state = result
        assert isinstance(init_tuple, tuple)
        assert len(init_tuple) == 3
        coeff_b, coeff_a, zi = init_tuple
        np.testing.assert_allclose(np.asarray(coeff_b, dtype=float), np.array([0.5, 0.5], dtype=float))
        np.testing.assert_allclose(np.asarray(coeff_a, dtype=float), np.array([1.0], dtype=float))
        assert zi is None
        np.testing.assert_allclose(np.asarray(state.b, dtype=float), np.array([0.5, 0.5], dtype=float))
        np.testing.assert_allclose(np.asarray(state.a, dtype=float), np.array([1.0], dtype=float))
        assert state.zi is None

    return _validator


def _assert_online_filter_step_result() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        step_tuple, state = result
        assert isinstance(step_tuple, tuple)
        assert len(step_tuple) == 2
        filtered, zi = step_tuple
        filtered_array = np.asarray(filtered, dtype=float)
        zi_array = np.asarray(zi, dtype=float)
        np.testing.assert_allclose(filtered_array, np.array([0.0, 0.5, 1.5, 2.5], dtype=float))
        assert zi_array.ndim == 1
        assert zi_array.size >= 1
        assert state.zi is not None
        np.testing.assert_allclose(np.asarray(state.zi, dtype=float), zi_array)

    return _validator


def _assert_triple_of_arrays_matching_onsets() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 3
        amps = np.asarray(result[0])
        rises = np.asarray(result[1])
        decays = np.asarray(result[2])
        assert amps.ndim == rises.ndim == decays.ndim == 1
        assert amps.shape == rises.shape == decays.shape
        assert np.all(amps >= 0)
        assert np.all(rises >= 0)
        assert np.all(decays >= 0)

    return _validator


def _assert_optimize_result_near(expected_x0: float, *, atol: float = 1e-2) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert abs(float(result.x[0]) - expected_x0) < atol

    return _validator


def _assert_tuple(expected: tuple[Any, ...], *, atol: float = 1e-8) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == len(expected)
        for actual_item, expected_item in zip(result, expected):
            if isinstance(expected_item, float):
                assert np.isclose(float(actual_item), expected_item, atol=atol)
            else:
                assert actual_item == expected_item

    return _validator


def _assert_type(expected_type: type[Any]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, expected_type)

    return _validator


def _assert_dict_keys(expected_keys: set[str]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys

    return _validator


def _assert_unit_interval_shape(expected_shape: tuple[int, ...]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray(result, dtype=float)
        assert values.shape == expected_shape
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    return _validator


def _assert_finite_vector(expected_length: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray(result, dtype=float)
        assert values.shape == (expected_length,)
        assert np.all(np.isfinite(values))

    return _validator


def _assert_profitable_cycles() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray(result, dtype=float)
        assert values.ndim == 1
        assert values.size >= 1
        assert np.all(values > 1.0)

    return _validator


def _assert_float_mask(expected: np.ndarray) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        np.testing.assert_allclose(np.asarray(result, dtype=float), expected)

    return _validator


def _assert_market_maker_state() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result.keys()) == {"s", "q", "sigma", "gamma", "k", "T", "t"}
        assert np.isclose(float(result["s"]), 100.0)
        assert np.isclose(float(result["q"]), 2.0)

    return _validator


def _assert_inventory_adjusted_quotes() -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, dict)
        assert set(result.keys()) == {"reservation_price", "bid", "ask", "spread"}
        assert float(result["bid"]) < float(result["ask"])
        assert np.isclose(float(result["ask"]) - float(result["bid"]), float(result["spread"]))

    return _validator


def _assert_positive_weights(expected_length: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray(result, dtype=float)
        assert values.shape == (expected_length,)
        assert np.all(values >= 0.0)
        assert np.isclose(float(values.sum()), 1.0)

    return _validator


def _assert_int_pair(expected_first: int, expected_second: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert int(result[0]) == expected_first
        assert int(result[1]) == expected_second

    return _validator


def _assert_float_int_pair(expected_first: float, expected_second: int, *, atol: float = 1e-12) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.isclose(float(result[0]), expected_first, atol=atol)
        assert int(result[1]) == expected_second

    return _validator


def _assert_float_list(expected: list[float], *, atol: float = 1e-8) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = [float(item) for item in result]
        assert len(values) == len(expected)
        np.testing.assert_allclose(np.asarray(values, dtype=float), np.asarray(expected, dtype=float), atol=atol)

    return _validator


def _assert_nonincreasing_float_list(expected_last: float) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        values = np.asarray([float(item) for item in result], dtype=float)
        assert values.ndim == 1
        assert values.size >= 2
        assert np.all(np.diff(values) <= 1e-12)
        assert np.isclose(values[-1], expected_last)

    return _validator


def _assert_draw_bundle(expected_draws: int, expected_rng: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        draws, rng_state_out = result
        assert isinstance(draws, dict)
        assert "theta" in draws
        theta = np.asarray(draws["theta"], dtype=float)
        assert theta.shape == (expected_draws,)
        assert int(rng_state_out) == expected_rng

    return _validator


def _assert_dataset_state(expected_labels: list[str], expected_sequences: list[str]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, dict)
        assert result["sequence_labels"] == expected_labels
        assert result["sequence_strs"] == expected_sequences

    return _validator


def _assert_batch_plan(expected: list[list[int]]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, list)
        assert result == expected

    return _validator


def _assert_state_snapshot(expected_bandwidth: int, expected_remaining_iterations: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        state = result[0]
        assert isinstance(state, dict)
        assert state["bandwidth"] == expected_bandwidth
        assert state["remaining_iterations"] == expected_remaining_iterations
        assert isinstance(state["accumulated_permutation"], list)
        assert state["accumulated_permutation"] == list(range(len(state["accumulated_permutation"])))
        assert isinstance(state["working_matrix"], np.ndarray)

    return _validator


def _assert_search_space(expected_amplitude: float, expected_count: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        amplitude, truncation_values = result
        assert np.isclose(float(amplitude), expected_amplitude)
        truncation_values = np.asarray(truncation_values, dtype=float)
        assert truncation_values.shape == (expected_count,)
        assert np.isclose(truncation_values[0], 0.1)
        assert np.isclose(truncation_values[-1], 0.99)

    return _validator


def _assert_permutation_list(expected: list[int]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, list)
        assert result == expected

    return _validator


def _assert_value(expected: Any) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert result == expected

    return _validator


def _assert_quantum_solver_orchestrator_result(expected_num_solutions: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        solutions, count_dist = result
        assert isinstance(solutions, list)
        assert len(solutions) == expected_num_solutions
        assert all(isinstance(sol, list) for sol in solutions)
        assert isinstance(count_dist, dict)
        assert len(count_dist) > 0
        assert all(isinstance(k, str) for k in count_dist)
        assert all(isinstance(v, int) and v > 0 for v in count_dist.values())

    return _validator


def _assert_quantum_solution_extractor(expected_num_solutions: int) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        solutions, counts = result
        assert isinstance(solutions, list)
        assert len(solutions) == expected_num_solutions
        assert all(isinstance(sol, list) for sol in solutions)
        assert isinstance(counts, list)
        assert len(counts) == expected_num_solutions
        assert all(isinstance(v, int) and v >= 0 for v in counts)

    return _validator
