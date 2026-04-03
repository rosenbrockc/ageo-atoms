"""Conservative deterministic runtime probes for a safe atom subset."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial

from .io import safe_atom_stem, write_json
from .paths import AUDIT_PROBES_DIR, ROOT
from .runtime_probe_plans import (
    get_advancedvi_and_iqe_probe_plans,
    get_biosppy_probe_plans,
    get_foundation_probe_plans,
    get_hftbacktest_and_ingest_probe_plans,
    get_kalman_filter_probe_plans,
    get_mcmc_foundational_probe_plans,
    get_molecular_docking_probe_plans,
    get_pronto_probe_plans,
    get_quantfin_probe_plans,
)
from .semantics import utc_now, write_evidence_section

_PROBE_JULIA_PROJECT = "/tmp/ageoa_juliapkg_project"
_PROBE_JULIA_DEPOT = "/tmp/ageoa_julia_depot"
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _PROBE_JULIA_PROJECT)
os.environ.setdefault("JULIA_DEPOT_PATH", _PROBE_JULIA_DEPOT)


@dataclass(frozen=True)
class ProbeCase:
    """A deterministic positive or negative runtime probe."""

    description: str
    invoke: Callable[[Callable[..., Any]], Any]
    validate: Callable[[Any], None] | None = None
    expect_exception: bool = False


@dataclass(frozen=True)
class ProbePlan:
    """Probe plan for one allowlisted atom."""

    positive: ProbeCase
    negative: ProbeCase | None = None
    parity_used: bool = False


def install_ageoa_stub(root: Path = ROOT) -> None:
    """Install a lightweight `ageoa` package stub so probes avoid ageoa.__init__."""
    ageoa_dir = root / "ageoa"
    existing = sys.modules.get("ageoa")
    if existing is not None and getattr(existing, "__path__", None):
        return
    stub = types.ModuleType("ageoa")
    stub.__path__ = [str(ageoa_dir)]
    stub.__package__ = "ageoa"
    sys.modules["ageoa"] = stub


def install_package_stub(package_name: str, root: Path = ROOT) -> None:
    """Install a lightweight package stub for an ageoa subpackage."""
    if package_name == "ageoa":
        install_ageoa_stub(root)
        return
    if not package_name.startswith("ageoa."):
        return
    parent_name = package_name.rsplit(".", 1)[0]
    install_package_stub(parent_name, root)
    existing = sys.modules.get(package_name)
    if existing is not None and getattr(existing, "__path__", None):
        return
    package_dir = root / Path(*package_name.split("."))
    stub = types.ModuleType(package_name)
    stub.__path__ = [str(package_dir)]
    stub.__package__ = package_name
    sys.modules[package_name] = stub


def _load_alias_module(alias_name: str, alias_file: Path) -> Any:
    """Load a temporary top-level alias module from a sibling artefact file."""
    spec = importlib.util.spec_from_file_location(alias_name, alias_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for alias module {alias_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = module
    spec.loader.exec_module(module)
    return module


def _install_legacy_sibling_aliases(module_file: Path) -> list[str]:
    """Install temporary aliases for older sibling artefact naming patterns."""
    installed: list[str] = []
    legacy_candidates = {
        "state_models": module_file.with_name(f"{module_file.stem}_state.py"),
        "witnesses": module_file.with_name(f"{module_file.stem}_witnesses.py"),
    }
    for alias_name, alias_file in legacy_candidates.items():
        if alias_name in sys.modules or not alias_file.exists():
            continue
        _load_alias_module(alias_name, alias_file)
        installed.append(alias_name)
    return installed


def load_module_from_file(module_import_path: str, module_file: Path) -> Any:
    """Load a module directly from its source file while preserving package-relative imports."""
    package_name = module_import_path.rsplit(".", 1)[0]
    install_package_stub(package_name)
    spec = importlib.util.spec_from_file_location(module_import_path, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_import_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_import_path] = module
    module_dir = str(module_file.parent)
    added_sys_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_sys_path = True
    alias_names = _install_legacy_sibling_aliases(module_file)
    try:
        spec.loader.exec_module(module)
    finally:
        for alias_name in alias_names:
            sys.modules.pop(alias_name, None)
        if added_sys_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    return module


def safe_import_module(module_import_path: str) -> Any:
    """Import an ageoa submodule without executing ageoa.__init__."""
    install_ageoa_stub()
    try:
        return importlib.import_module(module_import_path)
    except Exception:
        if not module_import_path.startswith("ageoa."):
            raise
        module_file = ROOT / Path(*module_import_path.split("."))
        module_file = module_file.with_suffix(".py")
        if not module_file.exists():
            raise
        return load_module_from_file(module_import_path, module_file)


def _summarize_value(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "length": len(value)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:120]}


def _run_case(func: Callable[..., Any], case: ProbeCase | None) -> dict[str, Any]:
    if case is None:
        return {"status": "not_applicable", "description": None}
    try:
        result = case.invoke(func)
        if case.expect_exception:
            return {
                "status": "fail",
                "description": case.description,
                "message": "probe unexpectedly succeeded",
                "result_summary": _summarize_value(result),
            }
        if case.validate is not None:
            case.validate(result)
        return {
            "status": "pass",
            "description": case.description,
            "result_summary": _summarize_value(result),
        }
    except Exception as exc:  # noqa: BLE001 - evidence wants exception detail
        if case.expect_exception:
            return {
                "status": "pass",
                "description": case.description,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:240],
            }
        return {
            "status": "fail",
            "description": case.description,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:240],
        }


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


def _search_plans() -> dict[str, ProbePlan]:
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
                _assert_array(np.array([[0.0, 1.0, 3.0], [np.inf, 0.0, 2.0], [np.inf, np.inf, 0.0]])),
            ),
            negative=ProbeCase(
                "floyd-warshall rejects non-square adjacency",
                lambda func: func(np.array([[0.0, 1.0, 0.0]])),
                expect_exception=True,
            ),
        ),
    }


def _numpy_plans() -> dict[str, ProbePlan]:
    return {
        "ageoa.numpy.arrays.array": ProbePlan(
            positive=ProbeCase(
                "numpy.array over a short Python list",
                lambda func: func([1, 2, 3]),
                _assert_array(np.array([1, 2, 3])),
            ),
        ),
        "ageoa.numpy.arrays.zeros": ProbePlan(
            positive=ProbeCase(
                "numpy.zeros over a tiny shape",
                lambda func: func((2, 2)),
                _assert_array(np.zeros((2, 2))),
            ),
            negative=ProbeCase(
                "numpy.zeros rejects an invalid shape type",
                lambda func: func("bad-shape"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.dot": ProbePlan(
            positive=ProbeCase(
                "numpy.dot over short vectors",
                lambda func: func(np.array([1, 2]), np.array([3, 4])),
                _assert_scalar(11),
            ),
            negative=ProbeCase(
                "numpy.dot rejects incompatible dimensions",
                lambda func: func(np.array([1, 2]), np.array([1, 2, 3])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.vstack": ProbePlan(
            positive=ProbeCase(
                "numpy.vstack over two rows",
                lambda func: func([np.array([1, 2]), np.array([3, 4])]),
                _assert_array(np.array([[1, 2], [3, 4]])),
            ),
            negative=ProbeCase(
                "numpy.vstack rejects an empty tuple",
                lambda func: func([]),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.arrays.reshape": ProbePlan(
            positive=ProbeCase(
                "numpy.reshape over a 1D vector",
                lambda func: func(np.arange(6), (2, 3)),
                _assert_array(np.arange(6).reshape(2, 3)),
            ),
            negative=ProbeCase(
                "numpy.reshape rejects a missing array",
                lambda func: func(None, (2, 1)),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.emath.sqrt": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.sqrt over positive inputs",
                lambda func: func(np.array([1.0, 4.0, 9.0])),
                _assert_array(np.array([1.0, 2.0, 3.0])),
            ),
        ),
        "ageoa.numpy.emath.log": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.log over positive inputs",
                lambda func: func(np.array([1.0, np.e, np.e**2])),
                _assert_array(np.array([0.0, 1.0, 2.0])),
            ),
        ),
        "ageoa.numpy.emath.log10": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.log10 over powers of ten",
                lambda func: func(np.array([1.0, 10.0, 100.0])),
                _assert_array(np.array([0.0, 1.0, 2.0])),
            ),
        ),
        "ageoa.numpy.emath.power": ProbePlan(
            positive=ProbeCase(
                "numpy.emath.power over a small vector",
                lambda func: func(np.array([2.0, 3.0]), np.array([3.0, 2.0])),
                _assert_array(np.array([8.0, 9.0])),
            ),
        ),
    }


def _scipy_plans() -> dict[str, ProbePlan]:
    matrix = np.array([[4.0, 2.0], [1.0, 3.0]])
    vector = np.array([1.0, 2.0])
    lu = np.array([[4.0, 2.0], [0.25, 2.5]])
    piv = np.array([0, 1], dtype=np.int32)
    def _linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.asarray(x, dtype=float) + b
    return {
        "ageoa.scipy.fft.dct": ProbePlan(
            positive=ProbeCase(
                "scipy.fft.dct over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0]), norm="ortho"),
                _assert_array(np.array([3.46410162, -1.41421356, 0.0])),
            ),
            negative=ProbeCase(
                "scipy.fft.dct rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.fft.idct": ProbePlan(
            positive=ProbeCase(
                "scipy.fft.idct over a short real vector",
                lambda func: func(np.array([3.46410162, -1.41421356, 0.0]), norm="ortho"),
                _assert_array(np.array([1.0, 2.0, 3.0])),
            ),
            negative=ProbeCase(
                "scipy.fft.idct rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.solve": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.solve over a tiny system",
                lambda func: func(matrix, vector),
                _assert_array(np.array([-0.1, 0.7])),
            ),
            negative=ProbeCase(
                "scipy.linalg.solve rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]]), np.array([1.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.inv": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.inv over a tiny matrix",
                lambda func: func(matrix),
                _assert_array(np.array([[0.3, -0.2], [-0.1, 0.4]])),
            ),
            negative=ProbeCase(
                "scipy.linalg.inv rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.det": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.det over a tiny matrix",
                lambda func: func(matrix),
                _assert_scalar(10.0),
            ),
        ),
        "ageoa.scipy.linalg.lu_factor": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.lu_factor over a tiny matrix",
                lambda func: func(matrix),
                lambda result: (
                    np.testing.assert_allclose(np.asarray(result[0]), lu),
                    np.testing.assert_array_equal(np.asarray(result[1]), piv),
                ),
            ),
            negative=ProbeCase(
                "scipy.linalg.lu_factor rejects non-square matrices",
                lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.linalg.lu_solve": ProbePlan(
            positive=ProbeCase(
                "scipy.linalg.lu_solve over a tiny factored system",
                lambda func: func((lu, piv), vector),
                _assert_array(np.array([-0.1, 0.7])),
            ),
            negative=ProbeCase(
                "scipy.linalg.lu_solve rejects incompatible RHS",
                lambda func: func((lu, piv), np.array([1.0, 2.0, 3.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.optimize.curve_fit": ProbePlan(
            positive=ProbeCase(
                "scipy.optimize.curve_fit recovers a simple linear model",
                lambda func: func(
                    _linear_model,
                    np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
                    np.array([1.0, 3.0, 5.0, 7.0], dtype=float),
                ),
                lambda result: (
                    np.testing.assert_allclose(np.asarray(result[0]), np.array([2.0, 1.0]), atol=1e-6),
                    np.testing.assert_equal(np.asarray(result[1]).shape, (2, 2)),
                ),
            ),
            negative=ProbeCase(
                "scipy.optimize.curve_fit rejects mismatched input lengths",
                lambda func: func(
                    _linear_model,
                    np.array([0.0, 1.0], dtype=float),
                    np.array([1.0], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _numpy_fft_plans() -> dict[str, ProbePlan]:
    return {
        "ageoa.numpy.fft.fft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fft over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0])),
                _assert_array(np.fft.fft(np.array([1.0, 2.0, 3.0]))),
            ),
            negative=ProbeCase(
                "numpy.fft.fft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.ifft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.ifft over a short complex spectrum",
                lambda func: func(np.fft.fft(np.array([1.0, 2.0, 3.0]))),
                _assert_array(np.array([1.0, 2.0, 3.0]) + 0j),
            ),
            negative=ProbeCase(
                "numpy.fft.ifft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.rfft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.rfft over a short real vector",
                lambda func: func(np.array([1.0, 2.0, 3.0, 4.0])),
                _assert_array(np.fft.rfft(np.array([1.0, 2.0, 3.0, 4.0]))),
            ),
            negative=ProbeCase(
                "numpy.fft.rfft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.irfft": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.irfft over a short Hermitian spectrum",
                lambda func: func(np.fft.rfft(np.array([1.0, 2.0, 3.0, 4.0]))),
                _assert_array(np.array([1.0, 2.0, 3.0, 4.0])),
            ),
            negative=ProbeCase(
                "numpy.fft.irfft rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.fftfreq": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fftfreq over a short window",
                lambda func: func(4, d=0.5),
                _assert_array(np.array([0.0, 0.5, -1.0, -0.5])),
            ),
            negative=ProbeCase(
                "numpy.fft.fftfreq rejects non-positive n",
                lambda func: func(0),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft.fftshift": ProbePlan(
            positive=ProbeCase(
                "numpy.fft.fftshift over a short vector",
                lambda func: func(np.array([0, 1, 2, 3])),
                _assert_array(np.array([2, 3, 0, 1])),
            ),
            negative=ProbeCase(
                "numpy.fft.fftshift rejects None",
                lambda func: func(None),
                expect_exception=True,
            ),
        ),
    }


def _sorting_plans() -> dict[str, ProbePlan]:
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


def _scipy_sparse_graph_plans() -> dict[str, ProbePlan]:
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    signal = np.array([1.0, 0.0])
    eigenvectors = np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]])
    x_hat = np.array([-0.70710678, -0.70710678])
    return {
        "ageoa.scipy.sparse_graph.graph_laplacian": ProbePlan(
            positive=ProbeCase(
                "graph Laplacian over a symmetric 2-node graph",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(weights)).toarray(),
                _assert_array(laplacian),
            ),
            negative=ProbeCase(
                "graph Laplacian rejects asymmetric weights",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.graph_fourier_transform": ProbePlan(
            positive=ProbeCase(
                "graph Fourier transform on a 2-node Laplacian",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal),
                lambda result: (
                    np.testing.assert_allclose(np.asarray(result[0]), x_hat, atol=1e-6),
                    np.testing.assert_allclose(np.asarray(result[1]), np.array([0.0, 2.0]), atol=1e-6),
                    np.testing.assert_allclose(np.abs(np.asarray(result[2])), np.abs(eigenvectors), atol=1e-6),
                ),
            ),
            negative=ProbeCase(
                "graph Fourier transform rejects mismatched signal length",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), np.array([1.0, 0.0, 2.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.inverse_graph_fourier_transform": ProbePlan(
            positive=ProbeCase(
                "inverse graph Fourier transform on a 2-node basis",
                lambda func: func(x_hat, eigenvectors),
                _assert_array(signal, atol=1e-6),
            ),
            negative=ProbeCase(
                "inverse graph Fourier transform rejects mismatched coefficient count",
                lambda func: func(np.array([1.0]), eigenvectors),
                expect_exception=True,
            ),
        ),
        "ageoa.scipy.sparse_graph.heat_kernel_diffusion": ProbePlan(
            positive=ProbeCase(
                "heat kernel diffusion smooths a 2-node signal",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal, 0.5),
                _assert_array(np.array([0.68393972, 0.31606028]), atol=1e-6),
            ),
            negative=ProbeCase(
                "heat kernel diffusion rejects negative diffusion time",
                lambda func: func(__import__("scipy.sparse").sparse.csr_matrix(laplacian), signal, -0.5),
                expect_exception=True,
            ),
        ),
    }


def _scipy_stats_plans() -> dict[str, ProbePlan]:
    return {
        "scipy.stats.ttest_ind": ProbePlan(
            positive=ProbeCase(
                "ttest_ind over two distinct samples",
                lambda func: func(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
                lambda result: np.testing.assert_allclose(
                    np.array([result.statistic, result.pvalue]),
                    np.array([-3.6742346141747673, 0.021311641128756727]),
                    atol=1e-8,
                ),
            ),
            negative=ProbeCase(
                "ttest_ind rejects None input",
                lambda func: func(None, np.array([1.0, 2.0])),
                expect_exception=True,
            ),
        ),
        "scipy.stats.pearsonr": ProbePlan(
            positive=ProbeCase(
                "pearsonr over perfectly correlated samples",
                lambda func: func(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])),
                lambda result: np.testing.assert_allclose(
                    np.array([result.statistic, result.pvalue]),
                    np.array([1.0, 0.0]),
                    atol=1e-12,
                ),
            ),
            negative=ProbeCase(
                "pearsonr rejects too-short input",
                lambda func: func(np.array([1.0]), np.array([1.0])),
                expect_exception=True,
            ),
        ),
        "scipy.stats.norm": ProbePlan(
            positive=ProbeCase(
                "norm returns a frozen normal distribution",
                lambda func: func(loc=1.0, scale=2.0),
                lambda result: np.testing.assert_allclose(
                    np.array([result.mean(), result.std()]),
                    np.array([1.0, 2.0]),
                    atol=1e-12,
                ),
            ),
            negative=ProbeCase(
                "norm rejects non-positive scale",
                lambda func: func(scale=0.0),
                expect_exception=True,
            ),
        ),
    }


def _scipy_integrate_plans() -> dict[str, ProbePlan]:
    return {
        "scipy.integrate.quad": ProbePlan(
            positive=ProbeCase(
                "quad integrates x^2 from 0 to 1",
                lambda func: func(lambda x: x * x, 0.0, 1.0),
                lambda result: np.testing.assert_allclose(np.array(result[:2]), np.array([1.0 / 3.0, result[1]]), atol=1e-8),
            ),
            negative=ProbeCase(
                "quad rejects a missing function",
                lambda func: func(None, 0.0, 1.0),
                expect_exception=True,
            ),
        ),
        "scipy.integrate.simpson": ProbePlan(
            positive=ProbeCase(
                "simpson integrates a quadratic sample",
                lambda func: func(np.array([0.0, 1.0, 4.0]), x=np.array([0.0, 1.0, 2.0])),
                _assert_scalar(8.0 / 3.0),
            ),
            negative=ProbeCase(
                "simpson rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
        ),
        "scipy.integrate.solve_ivp": ProbePlan(
            positive=ProbeCase(
                "solve_ivp integrates y'=-y over a short interval",
                lambda func: func(lambda t, y: -y, (0.0, 1.0), np.array([1.0]), t_eval=np.array([0.0, 1.0])),
                lambda result: np.testing.assert_allclose(result.y[:, -1], np.array([np.exp(-1.0)]), atol=5e-3),
            ),
            negative=ProbeCase(
                "solve_ivp rejects missing initial condition",
                lambda func: func(lambda t, y: -y, (0.0, 1.0), None),
                expect_exception=True,
            ),
        ),
    }


def _numpy_fft_v2_plans() -> dict[str, ProbePlan]:
    signal = np.arange(8, dtype=float).reshape(2, 4)
    spectrum = np.fft.fftn(signal, s=(2, 4), axes=(0, 1), norm="backward")
    return {
        "ageoa.numpy.fft_v2.forwardmultidimensionalfft": ProbePlan(
            positive=ProbeCase(
                "forward N-D FFT over a small 2x4 signal",
                lambda func: func(signal, [2, 4], [0, 1], "backward"),
                _assert_array(spectrum),
            ),
            negative=ProbeCase(
                "reject a missing input array",
                lambda func: func(None, [2, 4], [0, 1], "backward"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft_v2.inversemultidimensionalfft": ProbePlan(
            positive=ProbeCase(
                "inverse N-D FFT reconstructs the original signal",
                lambda func: func(spectrum, [2, 4], [0, 1], "backward"),
                _assert_array(signal.astype(complex)),
            ),
            negative=ProbeCase(
                "reject a missing spectrum input",
                lambda func: func(None, [2, 4], [0, 1], "backward"),
                expect_exception=True,
            ),
        ),
        "ageoa.numpy.fft_v2.hermitianspectraltransform": ProbePlan(
            positive=ProbeCase(
                "Hermitian FFT over a symmetric complex spectrum",
                lambda func: func(np.array([1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j]), 4, -1, "backward"),
                _assert_shape((4,)),
            ),
            negative=ProbeCase(
                "reject a missing Hermitian input",
                lambda func: func(None, 4, -1, "backward"),
                expect_exception=True,
            ),
        ),
    }


def _numpy_search_sort_v2_plans() -> dict[str, ProbePlan]:
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


def _scipy_optimize_v2_plans() -> dict[str, ProbePlan]:
    def _quadratic(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float((x[0] - 1.0) ** 2)

    return {
        "ageoa.scipy.optimize_v2.differentialevolutionoptimization": ProbePlan(
            positive=ProbeCase(
                "Differential evolution minimizes a one-dimensional quadratic on bounded input",
                lambda func: func(
                    _quadratic,
                    [(0.0, 2.0)],
                    maxiter=24,
                    popsize=8,
                    tol=0.0,
                    atol=0.0,
                    rng=np.random.default_rng(7),
                    workers=1,
                    polish=True,
                ),
                _assert_optimize_result_near(1.0, atol=1e-1),
            ),
            negative=ProbeCase(
                "reject malformed bounds",
                lambda func: func(_quadratic, [(0.0,)], rng=np.random.default_rng(7)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.scipy.optimize_v2.shgoglobaloptimization": ProbePlan(
            positive=ProbeCase(
                "SHGO minimizes a one-dimensional quadratic on a bounded interval",
                lambda func: func(
                    _quadratic,
                    [(0.0, 2.0)],
                    (),
                    (),
                    16,
                    1,
                    None,
                    {},
                    {},
                    "simplicial",
                ),
                _assert_optimize_result_near(1.0),
            ),
            negative=ProbeCase(
                "reject malformed bounds",
                lambda func: func(_quadratic, [(0.0,)], (), (), 16, 1, None, {}, {}, "simplicial"),
                expect_exception=True,
            ),
        ),
    }


def _quant_engine_plans() -> dict[str, ProbePlan]:
    def _assert_calculate_ofi_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        ofi, state = result
        assert abs(float(ofi) - 3.0) < 1e-12
        assert list(state.ofi_stream or []) == [3.0]

    def _assert_queue_state(*, my_qty: int, orders_ahead: int | None) -> Callable[[Any], None]:
        def _assert(result: Any) -> None:
            assert isinstance(result, tuple) and len(result) == 2
            marker, state = result
            assert marker is None
            assert (state.my_qty or 0) == my_qty
            if orders_ahead is None:
                assert state.orders_ahead is None
            else:
                assert (state.orders_ahead or 0) == orders_ahead

        return _assert

    return {
        "ageoa.quant_engine.calculate_ofi": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic order-flow imbalance and append it to state",
                lambda func: func(
                    100.0,
                    10,
                    101.0,
                    4,
                    3,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(ofi_stream=[]),
                ),
                _assert_calculate_ofi_bundle,
            ),
            negative=ProbeCase(
                "reject a negative bid quantity",
                lambda func: func(
                    100.0,
                    -1,
                    101.0,
                    4,
                    3,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(ofi_stream=[]),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_vwap": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic vwap participation fill to inventory",
                lambda func: func(
                    20,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=9),
                ),
                _assert_queue_state(my_qty=7, orders_ahead=None),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for vwap execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=9),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_pov": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic pov fill against queue priority",
                lambda func: func(
                    5,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                _assert_queue_state(my_qty=8, orders_ahead=0),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for pov execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.quant_engine.execute_passive": ProbePlan(
            positive=ProbeCase(
                "apply a deterministic passive fill against queue priority",
                lambda func: func(
                    5,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                _assert_queue_state(my_qty=8, orders_ahead=0),
            ),
            negative=ProbeCase(
                "reject a non-positive trade quantity for passive execution",
                lambda func: func(
                    0,
                    safe_import_module("ageoa.quant_engine.state_models").LimitQueueState(my_qty=10, orders_ahead=3),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _particle_filter_and_pasqal_plans() -> dict[str, ProbePlan]:
    def _assert_filter_step_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 5
        prior_state, model_spec, control_t, observation_t, rng_key = result
        assert isinstance(prior_state, dict)
        assert prior_state["rng_seed"] == 7
        assert model_spec == {"transition": "unit"}
        np.testing.assert_allclose(np.asarray(control_t, dtype=float), np.array([0.5], dtype=float))
        np.testing.assert_allclose(np.asarray(observation_t, dtype=float), np.array([1.5], dtype=float))
        np.testing.assert_array_equal(np.asarray(rng_key, dtype=np.int64), np.array([7], dtype=np.int64))

    def _assert_particle_propagation_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        proposed, carry_weights, rng_key_next = result
        expected_noise = np.random.RandomState(5).randn(2)
        np.testing.assert_allclose(
            np.asarray(proposed, dtype=float),
            np.array([1.0, 2.0], dtype=float) + expected_noise,
        )
        np.testing.assert_allclose(
            np.asarray(carry_weights, dtype=float),
            np.array([0.25, 0.75], dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(rng_key_next, dtype=np.int64),
            np.array([6], dtype=np.int64),
        )

    def _assert_likelihood_reweight_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        normalized, log_likelihood = result
        particles = np.array([1.0, 2.0], dtype=float)
        carry_weights = np.array([0.4, 0.6], dtype=float)
        obs = 1.5
        log_lik = -0.5 * (particles - obs) ** 2
        log_weights = np.log(carry_weights + 1e-300) + log_lik
        max_lw = np.max(log_weights)
        weights_exp = np.exp(log_weights - max_lw)
        total = weights_exp.sum()
        expected_normalized = weights_exp / total
        expected_log_likelihood = float(max_lw + np.log(total) - np.log(len(particles)))
        np.testing.assert_allclose(np.asarray(normalized, dtype=float), expected_normalized)
        assert abs(float(log_likelihood) - expected_log_likelihood) < 1e-12

    def _assert_particle_filter_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        posterior, trace = result
        assert isinstance(posterior, dict)
        assert isinstance(trace, dict)
        assert "particles" in posterior and "weights" in posterior
        assert "log_likelihood" in trace and "ess" in trace

    def _pasqal_positive(func: Callable[..., Any]) -> Any:
        import networkx as nx

        state_mod = safe_import_module("ageoa.pasqal.docking_state")
        state = state_mod.MolecularDockingState()
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        return func(graph, 2, state)

    def _assert_pasqal_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        mappings, state = result
        assert isinstance(mappings, list)
        assert len(mappings) == 2
        assert all(isinstance(item, dict) for item in mappings)
        assert hasattr(state, "graph")

    def _pasqal_mwis_positive(func: Callable[..., Any]) -> Any:
        import networkx as nx

        state_mod = safe_import_module("ageoa.pasqal.docking_state")
        state = state_mod.MolecularDockingState()
        graph = nx.Graph()
        graph.add_nodes_from(
            [
                (0, {"weight": 1.0}),
                (1, {"weight": 2.0}),
                (2, {"weight": 1.5}),
            ]
        )
        graph.add_edges_from([(0, 1), (1, 2)])
        lattice = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
        return func(graph, lattice, 2, state)

    def _assert_pasqal_mwis_result(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        solutions, state = result
        assert isinstance(solutions, list)
        assert len(solutions) == 2
        assert all(solution == {0, 2} for solution in solutions)
        assert getattr(state, "graph", None) is not None
        assert getattr(state, "lattice_id_coord_dic", None) == {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}

    return {
        "ageoa.particle_filters.basic.filter_step_preparation_and_dispatch": ProbePlan(
            positive=ProbeCase(
                "prepare a deterministic particle-filter step bundle from prior state",
                lambda func: func(
                    {
                        "particles": np.array([1.0, 2.0], dtype=float),
                        "weights": np.array([0.4, 0.6], dtype=float),
                        "rng_seed": 7,
                    },
                    {"transition": "unit"},
                    np.array([0.5], dtype=float),
                    np.array([1.5], dtype=float),
                ),
                _assert_filter_step_bundle,
            ),
            negative=ProbeCase(
                "reject a missing prior state bundle",
                lambda func: func(None, {"transition": "unit"}, np.array([0.5], dtype=float), np.array([1.5], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.particle_propagation_kernel": ProbePlan(
            positive=ProbeCase(
                "propagate a deterministic particle pair with fixed RNG state",
                lambda func: func(
                    {
                        "particles": np.array([1.0, 2.0], dtype=float),
                        "weights": np.array([0.25, 0.75], dtype=float),
                    },
                    {"transition": "unit"},
                    np.array([0.5], dtype=float),
                    np.array([5], dtype=np.int64),
                ),
                _assert_particle_propagation_bundle,
            ),
            negative=ProbeCase(
                "reject a missing prior state during propagation",
                lambda func: func(None, {"transition": "unit"}, np.array([0.5], dtype=float), np.array([5], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.likelihood_reweight_kernel": ProbePlan(
            positive=ProbeCase(
                "reweight a deterministic two-particle proposal against a scalar observation",
                lambda func: func(
                    np.array([1.0, 2.0], dtype=float),
                    np.array([0.4, 0.6], dtype=float),
                    np.array([1.5], dtype=float),
                    {"likelihood": "gaussian"},
                ),
                _assert_likelihood_reweight_bundle,
            ),
            negative=ProbeCase(
                "reject a missing proposed particle array",
                lambda func: func(None, np.array([0.4, 0.6], dtype=float), np.array([1.5], dtype=float), {"likelihood": "gaussian"}),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.particle_filters.basic.resample_and_belief_projection": ProbePlan(
            positive=ProbeCase(
                "resample weighted particles into a posterior belief state",
                lambda func: func(
                    np.array([10.0, 20.0, 30.0]),
                    np.array([0.2, 0.3, 0.5]),
                    np.array([4], dtype=np.int64),
                    -1.25,
                ),
                _assert_particle_filter_result,
            ),
            negative=ProbeCase(
                "reject a non-numeric log likelihood",
                lambda func: func(
                    np.array([10.0, 20.0, 30.0]),
                    np.array([0.2, 0.3, 0.5]),
                    np.array([4], dtype=np.int64),
                    "bad",
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.pasqal.docking.sub_graph_embedder": ProbePlan(
            positive=ProbeCase(
                "extract deterministic subgraph mappings from a small graph",
                _pasqal_positive,
                _assert_pasqal_result,
            ),
            negative=ProbeCase(
                "reject a non-positive subgraph quantity",
                lambda func: func(
                    __import__("networkx").Graph(),
                    0,
                    safe_import_module("ageoa.pasqal.docking_state").MolecularDockingState(),
                ),
                expect_exception=True,
            ),
        ),
        "ageoa.pasqal.docking.quantum_mwis_solver": ProbePlan(
            positive=ProbeCase(
                "solve a deterministic path-graph MWIS heuristic twice",
                _pasqal_mwis_positive,
                _assert_pasqal_mwis_result,
            ),
            negative=ProbeCase(
                "reject a non-positive sample count",
                lambda func: func(
                    __import__("networkx").path_graph(3),
                    {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
                    0,
                    safe_import_module("ageoa.pasqal.docking_state").MolecularDockingState(),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _rust_robotics_plans() -> dict[str, ProbePlan]:
    def _assert_bicycle_dynamics_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        x_dot, jacobian, u_inferred = result
        np.testing.assert_allclose(np.asarray(x_dot, dtype=float), np.array([2.0, 0.0, 0.0, 0.5]))
        assert np.asarray(jacobian, dtype=float).shape == (4, 4)
        np.testing.assert_allclose(np.asarray(u_inferred, dtype=float), np.array([0.0, 0.5]))

    return {
        "ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics": ProbePlan(
            positive=ProbeCase(
                "evaluate bicycle kinematics and recover the acceleration control component",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    np.array([0.0, 0.5], dtype=float),
                    0.0,
                    np.array([2.0, 0.0, 0.0, 0.5], dtype=float),
                ),
                _assert_bicycle_dynamics_bundle,
            ),
            negative=ProbeCase(
                "reject a non-numeric evaluation time",
                lambda func: func(
                    {"lf": 1.2, "lr": 1.3, "L": 2.5},
                    np.array([0.0, 0.0, 0.0, 2.0], dtype=float),
                    np.array([0.0, 0.5], dtype=float),
                    "bad",
                    np.array([2.0, 0.0, 0.0, 0.5], dtype=float),
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _molecular_docking_plans() -> dict[str, ProbePlan]:
    def _add_quantum_link_positive(func: Callable[..., Any]) -> Any:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(["A", "B"])
        return func(graph, "A", "B", 3)

    def _assert_add_quantum_link_result(result: Any) -> None:
        import networkx as nx

        assert isinstance(result, nx.Graph)
        assert result.has_edge("A", "_qlink_A_B_0")
        assert result.has_edge("_qlink_A_B_0", "_qlink_A_B_1")
        assert result.has_edge("_qlink_A_B_1", "B")

    def _assert_permutation_rows(result: Any) -> None:
        arr = np.asarray(result)
        assert arr.ndim == 2
        width = arr.shape[1]
        expected = np.arange(width)
        for row in arr:
            np.testing.assert_array_equal(np.sort(row), expected)

    def _greedy_mapping_context() -> tuple[Any, Any]:
        import networkx as nx

        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        lattice = nx.Graph()
        lattice.add_edges_from([(10, 11), (11, 12)])
        return graph, lattice

    def _assert_mapping_context(result: Any) -> None:
        import networkx as nx

        assert isinstance(result, dict)
        assert isinstance(result["graph"], nx.Graph)
        assert isinstance(result["lattice"], nx.Graph)
        assert isinstance(result["lattice_instance"], nx.Graph)
        assert result["previously_generated_subgraphs"] == []
        assert result["seed"] == 7

    def _assert_initialized_frontier(result: Any) -> None:
        assert isinstance(result, dict)
        assert result["mapping"] == {0: 10}
        assert result["unmapping"] == {10: 0}
        assert result["unexpanded_nodes"] == {0}

    def _assert_greedy_extension(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        state, scores = result
        assert isinstance(state, dict)
        assert isinstance(scores, dict)
        assert state["mapping"] == {0: 10, 1: 11}
        assert state["unmapping"] == {10: 0, 11: 1}
        assert state["unexpanded_nodes"] == {0, 1}
        assert scores == {1: 1.0, 2: 0.0}

    def _assert_mapping_valid(result: Any) -> None:
        assert result is True

    def _assert_greedy_pipeline(result: Any) -> None:
        import networkx as nx

        assert isinstance(result, tuple) and len(result) == 2
        generated_subgraph, final_state = result
        assert isinstance(generated_subgraph, nx.Graph)
        assert set(generated_subgraph.nodes()) == {0, 1}
        assert set(generated_subgraph.edges()) == {(0, 1)}
        assert isinstance(final_state, dict)
        assert final_state["mapping"] == {0: 10, 1: 11}

    def _assert_greedy_mapping_d12_context(result: Any) -> None:
        import networkx as nx

        assert isinstance(result, dict)
        assert isinstance(result["graph"], nx.Graph)
        assert isinstance(result["lattice"], nx.Graph)
        assert isinstance(result["lattice_instance"], nx.Graph)
        assert result["seed"] == 11
        assert isinstance(result["previously_generated_subgraphs"], list)

    def _assert_greedy_mapping_d12_state(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        state, scores = result
        assert isinstance(state, dict)
        assert set(state) == {"mapping", "unmapping", "unexpanded_nodes"}
        assert isinstance(scores, list)
        assert len(scores) >= 1
        for item in scores:
            assert set(item) == {"node", "score"}

    def _assert_greedy_mapping_d12_validation(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        mapping_state, is_valid = result
        assert isinstance(mapping_state, dict)
        assert set(mapping_state) == {"mapping", "unmapping", "unexpanded_nodes"}
        assert is_valid is True

    def _assert_bandwidth_proposal(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 4
        iteration_state, candidate_permutation, candidate_matrix, candidate_bandwidth = result
        assert isinstance(iteration_state, np.ndarray)
        assert iteration_state.shape == (1,)
        assert isinstance(candidate_permutation, list)
        assert sorted(candidate_permutation) == [0, 1, 2]
        assert isinstance(candidate_matrix, np.ndarray)
        assert candidate_matrix.shape == (3, 3)
        assert isinstance(candidate_bandwidth, int)
        assert candidate_bandwidth >= 0

    def _assert_bandwidth_state_update(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        next_state, continue_search = result
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (1,)
        state = next_state[0]
        assert state["bandwidth"] == 1
        assert state["remaining_iterations"] == 99
        assert state["accumulated_permutation"] == [2, 1, 0]
        assert continue_search is True

    return {
        "ageoa.molecular_docking.greedy_mapping.assemblestaticmappingcontext": ProbePlan(
            positive=ProbeCase(
                "assemble deterministic static mapping context from graph and lattice inputs",
                lambda func: func(*_greedy_mapping_context(), [], 7),
                _assert_mapping_context,
            ),
            negative=ProbeCase(
                "reject a missing graph input",
                lambda func: func(None, _greedy_mapping_context()[1], [], 7),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping.initializefrontierfromstartnode": ProbePlan(
            positive=ProbeCase(
                "seed a frontier by mapping the starting node to the first free lattice node",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    0,
                    {},
                    {},
                    set(),
                ),
                _assert_initialized_frontier,
            ),
            negative=ProbeCase(
                "reject a missing mapping context",
                lambda func: func(None, 0, {}, {}, set()),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping.scoreandextendgreedycandidates": ProbePlan(
            positive=ProbeCase(
                "score candidate nodes by mapped-neighbor support and greedily extend one step",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    [1, 2],
                    {0},
                    [11, 12],
                    {0: 10},
                    {10: 0},
                    True,
                    True,
                ),
                _assert_greedy_extension,
            ),
            negative=ProbeCase(
                "reject a missing considered-node list",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    None,
                    {0},
                    [11, 12],
                    {0: 10},
                    {10: 0},
                    True,
                    True,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping.validatecurrentmapping": ProbePlan(
            positive=ProbeCase(
                "validate a deterministic edge-preserving partial graph-lattice mapping",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    {0: 10, 1: 11},
                    {10: 0, 11: 1},
                ),
                _assert_mapping_valid,
            ),
            negative=ProbeCase(
                "reject an inconsistent inverse mapping",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    {0: 10, 1: 11},
                    {10: 0, 11: 2},
                ),
                _assert_value(False),
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping.rungreedymappingpipeline": ProbePlan(
            positive=ProbeCase(
                "assemble the greedy-mapping pipeline output from a validated partial state",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 7,
                    },
                    0,
                    True,
                    True,
                    {"mapping": {0: 10}, "unmapping": {10: 0}, "unexpanded_nodes": {0}},
                    {"mapping": {0: 10, 1: 11}, "unmapping": {10: 0, 11: 1}, "unexpanded_nodes": {0, 1}},
                    True,
                ),
                _assert_greedy_pipeline,
            ),
            negative=ProbeCase(
                "reject a missing mapping context in the orchestration stage",
                lambda func: func(None, 0, True, True, {"mapping": {}}, {"mapping": {}}, True),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping_d12.init_problem_context": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic refined-ingest greedy-mapping context",
                lambda func: func(
                    _greedy_mapping_context()[0],
                    _greedy_mapping_context()[1],
                    [],
                    11,
                ),
                _assert_greedy_mapping_d12_context,
            ),
            negative=ProbeCase(
                "reject a missing graph in refined-ingest greedy-mapping context initialization",
                lambda func: func(None, _greedy_mapping_context()[1], [], 11),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion": ProbePlan(
            positive=ProbeCase(
                "construct one deterministic refined-ingest greedy mapping expansion state",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 11,
                    },
                    0,
                    {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()},
                    [0, 1, 2],
                    True,
                    True,
                ),
                _assert_greedy_mapping_d12_state,
            ),
            negative=ProbeCase(
                "reject a missing problem context in refined-ingest greedy mapping expansion",
                lambda func: func(None, 0, {"mapping": {}, "unmapping": {}, "unexpanded_nodes": set()}, [0, 1], True, True),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate": ProbePlan(
            positive=ProbeCase(
                "validate one deterministic refined-ingest greedy mapping state",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 11,
                    },
                    0,
                    True,
                    True,
                    {"mapping": {0: 10, 1: 11}, "unmapping": {10: 0, 11: 1}, "unexpanded_nodes": {0, 1}},
                ),
                _assert_greedy_mapping_d12_validation,
            ),
            negative=ProbeCase(
                "reject a missing mapping state in refined-ingest greedy mapping orchestration",
                lambda func: func(
                    {
                        "graph": _greedy_mapping_context()[0],
                        "lattice": _greedy_mapping_context()[1],
                        "lattice_instance": _greedy_mapping_context()[1],
                        "previously_generated_subgraphs": [],
                        "seed": 11,
                    },
                    0,
                    True,
                    True,
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_mwis_solver": ProbePlan(
            positive=ProbeCase(
                "MWIS solver falls back to a deterministic median threshold on 1D input",
                lambda func: func(np.array([1.0, 3.0, 2.0])),
                _assert_array(np.array([0.0, 1.0, 1.0])),
            ),
            negative=ProbeCase(
                "MWIS solver rejects empty input",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.add_quantum_link.addquantumlink": ProbePlan(
            positive=ProbeCase(
                "quantum link insertion creates a deterministic chain between nodes",
                _add_quantum_link_positive,
                _assert_add_quantum_link_result,
            ),
            negative=ProbeCase(
                "quantum link insertion rejects a missing graph",
                lambda func: func(None, "A", "B", 2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.mwis_sa.to_qubo": ProbePlan(
            positive=ProbeCase(
                "MWIS-to-QUBO conversion maps diagonal weights and edge penalties",
                lambda func: func(np.array([[2.0, 1.0], [1.0, 3.0]]), 5.0),
                _assert_array(np.array([[-2.0, 5.0], [5.0, -3.0]])),
            ),
            negative=ProbeCase(
                "MWIS-to-QUBO rejects a missing graph",
                lambda func: func(None, 5.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations": ProbePlan(
            positive=ProbeCase(
                "threshold-based permutation enumeration returns valid permutations",
                lambda func: func(
                    np.array(
                        [
                            [0.0, 2.0, 0.0],
                            [2.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                        ]
                    ),
                    2.0,
                    np.array([0.25, 0.75]),
                ),
                _assert_permutation_rows,
            ),
            negative=ProbeCase(
                "threshold permutation enumeration rejects a non-float amplitude",
                lambda func: func(np.eye(2), "bad", np.array([0.25])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.validate_square_matrix_shape": ProbePlan(
            positive=ProbeCase(
                "square-matrix validation passes through a 3x3 matrix",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
            ),
            negative=ProbeCase(
                "square-matrix validation rejects a rectangular matrix",
                lambda func: func(np.array([[1.0, 2.0, 3.0]])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances": ProbePlan(
            positive=ProbeCase(
                "weighted index distance calculation matches elementwise |value * (i-j)|",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
            ),
            negative=ProbeCase(
                "weighted distance calculation rejects a missing matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth": ProbePlan(
            positive=ProbeCase(
                "bandwidth aggregation returns the maximum weighted distance",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_scalar(2.0),
            ),
            negative=ProbeCase(
                "bandwidth aggregation rejects missing distances",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.validate_symmetric_input": ProbePlan(
            positive=ProbeCase(
                "symmetric-input validation passes through a symmetric matrix",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
            ),
            negative=ProbeCase(
                "symmetric-input validation rejects an asymmetric matrix",
                lambda func: func(np.array([[0.0, 1.0], [0.0, 0.0]])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.initialize_reduction_state": ProbePlan(
            positive=ProbeCase(
                "reduction-state initialization creates a working matrix snapshot",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_state_snapshot(1, 100),
            ),
            negative=ProbeCase(
                "reduction-state initialization rejects a missing matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity": ProbePlan(
            positive=ProbeCase(
                "threshold sparsity zeros entries below the threshold",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), 1.5),
                _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
            ),
            negative=ProbeCase(
                "threshold sparsity rejects a non-float threshold",
                lambda func: func(np.eye(2), "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.build_sparse_graph_view": ProbePlan(
            positive=ProbeCase(
                "sparse-graph view preserves the thresholded matrix content",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_array(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
            ),
            negative=ProbeCase(
                "sparse-graph view rejects a missing matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order": ProbePlan(
            positive=ProbeCase(
                "RCM ordering produces a valid reverse bandwidth permutation",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_array(np.array([2, 1, 0])),
            ),
            negative=ProbeCase(
                "RCM ordering rejects a missing sparse matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.build_threshold_search_space": ProbePlan(
            positive=ProbeCase(
                "threshold search space returns matrix amplitude and the 0.1..0.99 sweep",
                lambda func: func(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])),
                _assert_search_space(2.0, 90),
            ),
            negative=ProbeCase(
                "threshold search space rejects a missing matrix",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation": ProbePlan(
            positive=ProbeCase(
                "minimum-bandwidth selector returns the best candidate permutation",
                lambda func: func(
                    np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                    np.array([[0, 1, 2], [2, 1, 0]]),
                ),
                _assert_permutation_list([0, 1, 2]),
            ),
            negative=ProbeCase(
                "minimum-bandwidth selector rejects missing candidates",
                lambda func: func(np.eye(2), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.extract_final_permutation": ProbePlan(
            positive=ProbeCase(
                "final-permutation extraction returns the accumulated permutation",
                lambda func: func(
                    np.array(
                        [
                            {
                                "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                "accumulated_permutation": [0, 1, 2],
                                "bandwidth": 1,
                                "remaining_iterations": 100,
                            }
                        ],
                        dtype=object,
                    )
                ),
                _assert_permutation_list([0, 1, 2]),
            ),
            negative=ProbeCase(
                "final-permutation extraction rejects a missing terminal state",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.propose_greedy_permutation_step": ProbePlan(
            positive=ProbeCase(
                "propose one greedy reverse-Cuthill-McKee permutation step",
                lambda func: func(
                    np.array(
                        [
                            {
                                "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                "accumulated_permutation": [0, 1, 2],
                                "bandwidth": 2,
                                "remaining_iterations": 100,
                            }
                        ],
                        dtype=object,
                    )
                ),
                _assert_bandwidth_proposal,
            ),
            negative=ProbeCase(
                "reject a missing reduction state when proposing a greedy permutation step",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.minimize_bandwidth.update_state_with_improvement_criterion": ProbePlan(
            positive=ProbeCase(
                "accept an improved bandwidth candidate and decrement remaining iterations",
                lambda func: func(
                    np.array(
                        [
                            {
                                "working_matrix": np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                                "accumulated_permutation": [0, 1, 2],
                                "bandwidth": 2,
                                "remaining_iterations": 100,
                            }
                        ],
                        dtype=object,
                    ),
                    [2, 1, 0],
                    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]),
                    1,
                ),
                _assert_bandwidth_state_update,
            ),
            negative=ProbeCase(
                "reject a missing candidate matrix during bandwidth-state update",
                lambda func: func(
                    np.array(
                        [
                            {
                                "working_matrix": np.eye(2),
                                "accumulated_permutation": [0, 1],
                                "bandwidth": 1,
                                "remaining_iterations": 10,
                            }
                        ],
                        dtype=object,
                    ),
                    [0, 1],
                    None,
                    1,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _molecular_docking_quantum_solver_plans() -> dict[str, ProbePlan]:
    def _assert_quantum_problem_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 5
        register, parameters, permutation_list, backend_flags, num_sol = result
        assert register == {0: (0.0, 0.0), 1: (1.0, 0.0)}
        assert parameters["duration"] == 4000.0
        assert parameters["detuning_maximum"] == 5.0
        assert parameters["amplitude_maximum"] == 5.0
        assert parameters["register"] == register
        assert permutation_list == [[0, 1]]
        assert backend_flags == {"run_qutip": False, "run_emu_mps": False, "run_sv": True}
        assert num_sol == 2

    def _assert_quantum_sample_bundle(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        measurement_counts, final_register = result
        assert isinstance(measurement_counts, dict)
        assert measurement_counts
        assert set(measurement_counts.keys()).issubset({"00", "01", "10", "11"})
        assert sum(int(v) for v in measurement_counts.values()) == 500
        assert final_register == {0: (0.0, 0.0), 1: (1.0, 0.0)}

    def _assert_solution_list(expected: list[list[int]]) -> Callable[[Any], None]:
        def _assert(result: Any) -> None:
            assert isinstance(result, list)
            assert result == expected

        return _assert

    def _quantum_solver_graph() -> Any:
        import networkx as nx

        graph = nx.Graph()
        graph.add_node(0, weight=1.0)
        graph.add_node(1, weight=2.0)
        graph.add_edge(0, 1)
        return graph

    return {
        "ageoa.molecular_docking.quantum_solver.quantumproblemdefinition": ProbePlan(
            positive=ProbeCase(
                "prepare a deterministic two-node quantum problem definition bundle",
                lambda func: func(
                    _quantum_solver_graph(),
                    {0: (0.0, 0.0), 1: (1.0, 0.0)},
                    2,
                    False,
                ),
                _assert_quantum_problem_bundle,
            ),
            negative=ProbeCase(
                "reject a missing problem graph",
                lambda func: func(None, {0: (0.0, 0.0), 1: (1.0, 0.0)}, 2, False),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver.adiabaticquantumsampler": ProbePlan(
            positive=ProbeCase(
                "sample a deterministic two-node quantum register with a fixed RNG path",
                lambda func: func(
                    {0: (0.0, 0.0), 1: (1.0, 0.0)},
                    {
                        "graph": _quantum_solver_graph(),
                        "duration": 4000.0,
                        "detuning_maximum": 5.0,
                        "amplitude_maximum": 5.0,
                    },
                    [[0, 1]],
                    {"run_qutip": False, "run_emu_mps": False, "run_sv": True},
                ),
                _assert_quantum_sample_bundle,
            ),
            negative=ProbeCase(
                "reject a missing register definition",
                lambda func: func(
                    None,
                    {
                        "graph": _quantum_solver_graph(),
                        "duration": 4000.0,
                        "detuning_maximum": 5.0,
                        "amplitude_maximum": 5.0,
                    },
                    [[0, 1]],
                    {"run_qutip": False, "run_emu_mps": False, "run_sv": True},
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.quantum_solver.solutionextraction": ProbePlan(
            positive=ProbeCase(
                "extract the two most frequent bitstring solutions from a deterministic count map",
                lambda func: func(
                    {"10": 7, "01": 3},
                    {0: (0.0, 0.0), 1: (1.0, 0.0)},
                    2,
                ),
                _assert_solution_list([[0], [1]]),
            ),
            negative=ProbeCase(
                "reject a missing measurement count distribution",
                lambda func: func(None, {0: (0.0, 0.0), 1: (1.0, 0.0)}, 2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _biosppy_detector_plans() -> dict[str, ProbePlan]:
    def _synthetic_ecg() -> np.ndarray:
        fs = 1000.0
        duration = 10.0
        heart_rate = 75.0
        t = np.linspace(0.0, duration, int(duration * fs), endpoint=False)
        f0 = heart_rate / 60.0
        signal = np.zeros_like(t)
        peak_times = np.arange(0.5, duration, 1.0 / f0)
        for peak in peak_times:
            signal += 1.5 * np.exp(-((t - peak) ** 2) / (2 * (0.005 ** 2)))
        rng = np.random.RandomState(7)
        signal += 0.05 * rng.normal(size=len(t))
        return signal

    def _assert_peak_indices(result: Any) -> None:
        peaks = np.asarray(result)
        assert peaks.ndim == 1
        assert peaks.size > 0
        assert np.all(np.diff(peaks) >= 0)
        assert np.all(peaks >= 0)
        assert np.all(peaks < 10_000)

    signal = _synthetic_ecg()
    ppg_sampling_rate = 100.0
    ppg_time = np.linspace(0.0, 10.0, int(10.0 * ppg_sampling_rate), endpoint=False)
    ppg_signal = np.full_like(ppg_time, 0.02)
    for center in np.arange(0.5, 10.0, 1.0):
        ppg_signal += np.exp(-((ppg_time - center) ** 2) / (2 * (0.03 ** 2)))

    emg_sampling_rate = 1000.0
    emg_time = np.linspace(0.0, 2.0, int(2.0 * emg_sampling_rate), endpoint=False)
    emg_rest = 0.01 * np.sin(2 * np.pi * 10 * np.linspace(0.0, 0.4, int(0.4 * emg_sampling_rate), endpoint=False))
    emg_signal = 0.01 * np.sin(2 * np.pi * 10 * emg_time)
    emg_signal[700:1100] += 0.5 * np.sin(np.linspace(0.0, np.pi, 400))
    emg_signal[1300:1600] += 0.7 * np.sin(np.linspace(0.0, np.pi, 300))

    eda_sampling_rate = 100.0
    eda_time = np.linspace(0.0, 20.0, int(20.0 * eda_sampling_rate), endpoint=False)
    eda_signal = 0.1 + 0.02 * np.sin(2 * np.pi * 0.1 * eda_time)
    for center in (4.0, 10.0, 16.0):
        eda_signal += 0.5 * np.exp(-np.maximum(eda_time - center, 0.0) / 1.2) * (eda_time >= center)

    pcg_sampling_rate = 1000.0
    pcg_time = np.linspace(0.0, 4.0, int(4.0 * pcg_sampling_rate), endpoint=False)
    pcg_signal = np.zeros_like(pcg_time)
    for s1, s2 in [(0.4, 0.7), (1.4, 1.7), (2.4, 2.7), (3.4, 3.7)]:
        pcg_signal += np.exp(-((pcg_time - s1) ** 2) / (2 * (0.01 ** 2)))
        pcg_signal += 0.7 * np.exp(-((pcg_time - s2) ** 2) / (2 * (0.012 ** 2)))

    abp_sampling_rate = 1000.0
    abp_time = np.linspace(0.0, 10.0, int(10.0 * abp_sampling_rate), endpoint=False)
    abp_signal = np.zeros_like(abp_time)
    for center in np.arange(0.5, 10.0, 1.0):
        abp_signal += 0.8 * np.exp(-((abp_time - center) ** 2) / (2 * (0.02 ** 2)))
        abp_signal += 0.3 * np.exp(-((abp_time - (center + 0.08)) ** 2) / (2 * (0.03 ** 2)))

    return {
        "ageoa.biosppy.abp.audio_onset_detection": ProbePlan(
            positive=ProbeCase(
                "ABP onset detection returns monotonic onset indices on a synthetic pulse trace",
                lambda func: func(abp_signal, abp_sampling_rate),
                _assert_monotonic_index_array(max_value=len(abp_signal) - 1),
            ),
            negative=ProbeCase(
                "ABP onset detection rejects a missing signal",
                lambda func: func(None, abp_sampling_rate),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.bandpass_filter": ProbePlan(
            positive=ProbeCase(
                "ECG bandpass filtering preserves waveform shape on a synthetic ECG trace",
                lambda func: func(signal, sampling_rate=1000.0),
                _assert_shape(signal.shape),
            ),
            negative=ProbeCase(
                "ECG bandpass filtering rejects a missing signal",
                lambda func: func(None, sampling_rate=1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.r_peak_detection": ProbePlan(
            positive=ProbeCase(
                "R-peak detection returns monotonic peak indices on a synthetic ECG trace",
                lambda func: func(signal, sampling_rate=1000.0),
                _assert_monotonic_index_array(max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "R-peak detection rejects a negative sampling rate",
                lambda func: func(signal, sampling_rate=-1.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.peak_correction": ProbePlan(
            positive=ProbeCase(
                "Peak correction returns monotonic corrected peak indices on a synthetic ECG trace",
                lambda func: func(signal, np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                _assert_monotonic_index_array(max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "Peak correction rejects a missing filtered signal",
                lambda func: func(None, np.array([500, 1300], dtype=int), sampling_rate=1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.template_extraction": ProbePlan(
            positive=ProbeCase(
                "Template extraction returns templates and aligned peaks on a synthetic ECG trace",
                lambda func: func(signal, np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                _assert_pair_of_arrays(),
            ),
            negative=ProbeCase(
                "Template extraction rejects a missing filtered signal",
                lambda func: func(None, np.array([500, 1300], dtype=int), sampling_rate=1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.heart_rate_computation": ProbePlan(
            positive=ProbeCase(
                "Heart-rate computation returns aligned index and bpm arrays",
                lambda func: func(np.array([500, 1300, 2100, 2900, 3700, 4500, 5300, 6100, 6900, 7700], dtype=int), sampling_rate=1000.0),
                _assert_pair_of_arrays(),
            ),
            negative=ProbeCase(
                "Heart-rate computation rejects a negative sampling rate",
                lambda func: func(np.array([500, 1300, 2100], dtype=int), sampling_rate=-1.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.ssf_segmenter": ProbePlan(
            positive=ProbeCase(
                "SSF segmenter returns monotonic peak indices on a synthetic ECG trace",
                lambda func: func(signal, sampling_rate=1000.0),
                _assert_monotonic_index_array(max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "SSF segmenter rejects a missing signal",
                lambda func: func(None, sampling_rate=1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg.christov_segmenter": ProbePlan(
            positive=ProbeCase(
                "Christov segmenter returns monotonic peak indices on a synthetic ECG trace",
                lambda func: func(signal, sampling_rate=1000.0),
                _assert_monotonic_index_array(max_value=len(signal) - 1),
            ),
            negative=ProbeCase(
                "Christov segmenter rejects a missing signal",
                lambda func: func(None, sampling_rate=1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_detectors.hamilton_segmentation": ProbePlan(
            positive=ProbeCase(
                "Hamilton ECG segmentation detects peaks on a synthetic ECG trace",
                lambda func: func(signal, 1000.0),
                _assert_peak_indices,
            ),
            negative=ProbeCase(
                "Hamilton ECG segmentation rejects a missing signal",
                lambda func: func(None, 1000.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_detectors.thresholdbasedsignalsegmentation": ProbePlan(
            positive=ProbeCase(
                "ASI threshold segmentation detects peaks on a synthetic ECG trace",
                lambda func: func(signal, 1000.0, 5.0),
                _assert_peak_indices,
            ),
            negative=ProbeCase(
                "ASI threshold segmentation rejects a missing signal",
                lambda func: func(None, 1000.0, 5.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_detectors.hamilton_segmenter": ProbePlan(
            positive=ProbeCase(
                "Hamilton ECG segmenter detects peaks on a synthetic ECG trace",
                lambda func: func(signal, 1000.0),
                _assert_peak_indices,
            ),
            negative=ProbeCase(
                "Hamilton ECG segmenter rejects a non-numeric sampling rate",
                lambda func: func(signal, "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.eda.gamboa_segmenter": ProbePlan(
            positive=ProbeCase(
                "EDA onset segmentation returns monotonic indices on a synthetic phasic signal",
                lambda func: func(eda_signal, eda_sampling_rate),
                _assert_monotonic_index_array(max_value=len(eda_signal) - 1),
            ),
            negative=ProbeCase(
                "EDA onset segmentation rejects an empty signal",
                lambda func: func(np.asarray([], dtype=float), eda_sampling_rate),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.eda.eda_feature_extraction": ProbePlan(
            positive=ProbeCase(
                "EDA feature extraction returns aligned amplitude, rise-time, and decay arrays",
                lambda func: func(eda_signal, np.array([400, 1000, 1600], dtype=int), eda_sampling_rate),
                _assert_triple_of_arrays_matching_onsets(),
            ),
            negative=ProbeCase(
                "EDA feature extraction rejects a missing signal",
                lambda func: func(None, np.array([400, 1000], dtype=int), eda_sampling_rate),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ppg_detectors.detect_signal_onsets_elgendi2013": ProbePlan(
            positive=ProbeCase(
                "Elgendi PPG onset detection finds the synthetic pulse train onsets",
                lambda func: func(ppg_signal, ppg_sampling_rate, 0.111, 0.667, 0.02, 0.3),
                _assert_sorted_array(np.array([50, 150, 250, 350, 450, 550, 650, 750, 850, 950])),
            ),
            negative=ProbeCase(
                "Elgendi PPG onset detection rejects a non-numeric sampling rate",
                lambda func: func(ppg_signal, "bad", 0.111, 0.667, 0.02, 0.3),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ppg_detectors.detectonsetevents": ProbePlan(
            positive=ProbeCase(
                "Kavsaoğlu PPG onset detection finds the synthetic pulse train events",
                lambda func: func(ppg_signal, ppg_sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
                _assert_sorted_array(np.array([78, 178, 278, 378, 478, 578, 678, 778, 878])),
            ),
            negative=ProbeCase(
                "Kavsaoğlu PPG onset detection rejects a missing signal",
                lambda func: func(None, ppg_sampling_rate, 0.2, 4, 60.0, 0.3, 180.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.pcg.shannon_energy": ProbePlan(
            positive=ProbeCase(
                "PCG Shannon-energy envelope preserves signal shape and non-negativity",
                lambda func: func(pcg_signal),
                _assert_shape(pcg_signal.shape),
            ),
            negative=ProbeCase(
                "PCG Shannon-energy envelope rejects a non-array signal",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.pcg.pcg_segmentation": ProbePlan(
            positive=ProbeCase(
                "PCG segmentation returns alternating S1 and S2 peaks from a synthetic envelope",
                lambda func: func(np.maximum(pcg_signal, 0.0), pcg_sampling_rate),
                _assert_pair_of_sorted_integer_arrays(),
            ),
            negative=ProbeCase(
                "PCG segmentation rejects an empty envelope",
                lambda func: func(np.asarray([], dtype=float), pcg_sampling_rate),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.emg_detectors.detect_onsets_with_rest_aware_thresholds": ProbePlan(
            positive=ProbeCase(
                "rest-aware EMG onset detection returns a valid empty onset array for the quiet synthetic trace",
                lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 20, 10, 1.0, 0.5),
                _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
            ),
            negative=ProbeCase(
                "rest-aware EMG onset detection rejects a missing signal",
                lambda func: func(None, emg_rest, emg_sampling_rate, 20, 10, 1.0, 0.5),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.emg_detectors.bonato_onset_detection": ProbePlan(
            positive=ProbeCase(
                "Bonato EMG onset detection returns a valid onset array for the quiet synthetic trace",
                lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05, 3, 2),
                _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
            ),
            negative=ProbeCase(
                "Bonato EMG onset detection rejects a missing signal",
                lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05, 3, 2),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.emg_detectors.threshold_based_onset_detection": ProbePlan(
            positive=ProbeCase(
                "threshold-based EMG onset detection returns a valid onset array for the quiet synthetic trace",
                lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05),
                _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
            ),
            negative=ProbeCase(
                "threshold-based EMG onset detection rejects a missing signal",
                lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.emg_detectors.solnik_onset_detect": ProbePlan(
            positive=ProbeCase(
                "Solnik EMG onset detection returns a valid onset array for the quiet synthetic trace",
                lambda func: func(emg_signal, emg_rest, emg_sampling_rate, 1.0, 0.05),
                _assert_monotonic_index_array(max_value=len(emg_signal) - 1),
            ),
            negative=ProbeCase(
                "Solnik EMG onset detection rejects a missing signal",
                lambda func: func(None, emg_rest, emg_sampling_rate, 1.0, 0.05),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _biosppy_sqi_plans() -> dict[str, ProbePlan]:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    detector_1 = np.array([20, 60, 100, 140, 180])
    detector_2 = np.array([21, 59, 101, 141, 179])
    return {
        "ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018": ProbePlan(
            positive=ProbeCase(
                "ZZ2018 composite SQI classifies a small synthetic signal",
                lambda func: func(signal, detector_1, detector_2, 1000.0, 50, 64, "simple"),
                _assert_value("Barely acceptable"),
            ),
            negative=ProbeCase(
                "ZZ2018 composite SQI rejects a non-numeric sampling rate",
                lambda func: func(signal, detector_1, detector_2, "bad", 50, 64, "simple"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi": ProbePlan(
            positive=ProbeCase(
                "beat-agreement SQI returns the expected agreement score",
                lambda func: func(detector_1, detector_2, 1000.0, "simple", 50),
                _assert_scalar(100.0),
            ),
            negative=ProbeCase(
                "beat-agreement SQI rejects a non-numeric sampling rate",
                lambda func: func(detector_1, detector_2, "bad", "simple", 50),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi": ProbePlan(
            positive=ProbeCase(
                "frequency-power SQI returns the expected band-power ratio",
                lambda func: func(signal, 1000.0, 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                _assert_scalar(0.0),
            ),
            negative=ProbeCase(
                "frequency-power SQI rejects a non-numeric sampling rate",
                lambda func: func(signal, "bad", 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi": ProbePlan(
            positive=ProbeCase(
                "refined-ingest ZZ2018 composite SQI assembles the expected score bundle",
                lambda func: func(signal, detector_1, detector_2, 1000.0, 50, 64, "simple", 100.0, 0.0, 1.5),
                _assert_dict_keys({"b_sqi", "f_sqi", "k_sqi"}),
            ),
            negative=ProbeCase(
                "refined-ingest ZZ2018 composite SQI rejects a non-numeric sampling rate",
                lambda func: func(signal, detector_1, detector_2, "bad", 50, 64, "simple", 100.0, 0.0, 1.5),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_zz2018_d12.computebeatagreementsqi": ProbePlan(
            positive=ProbeCase(
                "refined-ingest beat-agreement SQI returns the expected agreement score",
                lambda func: func(detector_1, detector_2, 1000.0, "simple", 50),
                _assert_scalar(100.0),
            ),
            negative=ProbeCase(
                "refined-ingest beat-agreement SQI rejects a non-numeric sampling rate",
                lambda func: func(detector_1, detector_2, "bad", "simple", 50),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.ecg_zz2018_d12.computefrequencysqi": ProbePlan(
            positive=ProbeCase(
                "refined-ingest frequency-power SQI returns the expected band-power ratio",
                lambda func: func(signal, 1000.0, 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                _assert_scalar(0.0),
            ),
            negative=ProbeCase(
                "refined-ingest frequency-power SQI rejects a non-numeric sampling rate",
                lambda func: func(signal, "bad", 64, np.array([5, 15]), np.array([5, 40]), "simple"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _biosppy_online_filter_plans() -> dict[str, ProbePlan]:
    coeff_b = np.array([0.5, 0.5], dtype=float)
    coeff_a = np.array([1.0], dtype=float)
    signal = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    def _invoke_filterstep(func: Callable[..., Any]) -> Any:
        module = importlib.import_module(func.__module__)
        _, state = module.filterstateinit(coeff_b, coeff_a)
        return func(signal, state)

    def _invoke_invalid_filterstep(func: Callable[..., Any]) -> Any:
        module = importlib.import_module(func.__module__)
        _, state = module.filterstateinit(coeff_b, coeff_a)
        return func(None, state)

    return {
        "ageoa.biosppy.online_filter.filterstateinit": ProbePlan(
            positive=ProbeCase(
                "initialize a chunked BioSPPy OnlineFilter state bundle",
                lambda func: func(coeff_b, coeff_a),
                _assert_online_filter_init_state(),
            ),
            negative=ProbeCase(
                "reject a zero leading denominator coefficient",
                lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.online_filter.filterstep": ProbePlan(
            positive=ProbeCase(
                "filter one chunk with a serialized OnlineFilter state",
                _invoke_filterstep,
                _assert_online_filter_step_result(),
            ),
            negative=ProbeCase(
                "reject a missing signal chunk",
                _invoke_invalid_filterstep,
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.online_filter_codex.filterstateinit": ProbePlan(
            positive=ProbeCase(
                "initialize a chunked BioSPPy OnlineFilter state bundle",
                lambda func: func(coeff_b, coeff_a),
                _assert_online_filter_init_state(),
            ),
            negative=ProbeCase(
                "reject a zero leading denominator coefficient",
                lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.online_filter_codex.filterstep": ProbePlan(
            positive=ProbeCase(
                "filter one chunk with a serialized OnlineFilter state",
                _invoke_filterstep,
                _assert_online_filter_step_result(),
            ),
            negative=ProbeCase(
                "reject a missing signal chunk",
                _invoke_invalid_filterstep,
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.online_filter_v2.filterstateinit": ProbePlan(
            positive=ProbeCase(
                "initialize a chunked BioSPPy OnlineFilter state bundle",
                lambda func: func(coeff_b, coeff_a),
                _assert_online_filter_init_state(),
            ),
            negative=ProbeCase(
                "reject a zero leading denominator coefficient",
                lambda func: func(coeff_b, np.array([0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.biosppy.online_filter_v2.filterstep": ProbePlan(
            positive=ProbeCase(
                "filter one chunk with a serialized OnlineFilter state",
                _invoke_filterstep,
                _assert_online_filter_step_result(),
            ),
            negative=ProbeCase(
                "reject a missing signal chunk",
                _invoke_invalid_filterstep,
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _conjugate_prior_and_small_mcmc_plans() -> dict[str, ProbePlan]:
    target_log = lambda x: float(-0.5 * np.dot(x, x))
    mala_mean = lambda x: 2.0 * np.asarray(x, dtype=float)

    def _assert_de_kernel(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        state, rng = result
        assert np.asarray(state).shape == (3, 2)
        assert np.asarray(rng).shape == (2,)

    return {
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel": ProbePlan(
            positive=ProbeCase(
                "compute a Beta-Binomial posterior update from binary observations",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0, 1.0], dtype=float),
                ),
                _assert_array(np.array([5.0, 4.0], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing prior vector",
                lambda func: func(None, np.eye(2), np.array([1.0, 0.0], dtype=float)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.conjugate_priors.beta_binom.posterior_randmodel_weighted": ProbePlan(
            positive=ProbeCase(
                "compute a weighted Beta-Binomial posterior update",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0], dtype=float),
                    np.array([1.0, 0.5, 2.0], dtype=float),
                ),
                _assert_array(np.array([5.0, 3.5], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a missing weight vector",
                lambda func: func(
                    np.array([2.0, 3.0], dtype=float),
                    np.eye(2),
                    np.array([1.0, 0.0, 1.0], dtype=float),
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.conjugate_priors.normal.normal_gamma_posterior_update": ProbePlan(
            positive=ProbeCase(
                "compute a Normal-Gamma posterior update from sufficient statistics",
                lambda func: func(
                    {"mu0": 0.0, "kappa0": 1.0, "alpha0": 2.0, "beta0": 3.0},
                    {"n": 4.0, "mean": 1.5, "var": 2.0},
                ),
                _assert_value({"mu0": 1.2, "kappa0": 5.0, "alpha0": 4.0, "beta0": 7.9}),
            ),
            negative=ProbeCase(
                "reject missing sufficient statistics",
                lambda func: func({"mu0": 0.0, "kappa0": 1.0, "alpha0": 2.0, "beta0": 3.0}, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel": ProbePlan(
            positive=ProbeCase(
                "build and run one Differential Evolution transition kernel on a small population",
                lambda func: func(target_log)(
                    np.array([[0.0, 0.5], [1.0, -0.5], [-0.25, 0.75]], dtype=float),
                    np.array([3, 5], dtype=np.int64),
                ),
                _assert_de_kernel,
            ),
            negative=ProbeCase(
                "reject a missing target log-kernel oracle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment": ProbePlan(
            positive=ProbeCase(
                "compute the deterministic MALA proposal adjustment term",
                lambda func: func(0.5, np.array([1.0, -1.0], dtype=float), mala_mean),
                _assert_array(np.array([1.25, -1.25], dtype=float)),
            ),
            negative=ProbeCase(
                "reject a non-numeric step size",
                lambda func: func("bad", np.array([1.0, -1.0], dtype=float), mala_mean),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


def _mcmc_foundational_plans() -> dict[str, ProbePlan]:
    target_log = lambda x: float(-0.5 * np.dot(x, x))
    tensor_fn = lambda x: np.eye(x.shape[0], dtype=float)

    def _assert_mh_tuple(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        state, rng = result
        assert np.asarray(state).shape == (2,)
        assert np.asarray(rng).shape == (2,)

    def _assert_hmc_init(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        kernel_spec, chain_state = result
        np.testing.assert_allclose(np.asarray(kernel_spec, dtype=float), np.array([0.1, 4.0, 2.0], dtype=float))
        assert np.asarray(chain_state).shape == (5,)

    def _assert_hmc_rng(result: Any) -> None:
        arr = np.asarray(result)
        assert arr.shape == (1,)
        assert np.issubdtype(arr.dtype, np.integer)
        np.testing.assert_array_equal(arr, np.array([7], dtype=np.int64))

    def _assert_nuts_init(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        nuts_state, rng_key = result
        assert np.asarray(nuts_state).shape == (5,)
        np.testing.assert_array_equal(np.asarray(rng_key), np.array([7], dtype=np.int64))

    def _assert_mini_hmc_init(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        chain_state, kernel_static = result
        assert np.asarray(chain_state).shape == (4,)
        np.testing.assert_allclose(np.asarray(kernel_static, dtype=float), np.array([0.1, 4.0, 1.0], dtype=float))

    def _assert_hmc_transition(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        state_out, prng_key_out, stats = result
        assert np.asarray(state_out).shape == (5,)
        assert np.asarray(prng_key_out).shape == (1,)
        assert isinstance(stats, dict)
        assert {"accepted", "accept_prob", "delta_H"} <= set(stats)

    def _assert_nuts_transition(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 4
        samples, trace, nuts_state_out, rng_key_out = result
        assert np.asarray(samples).shape == (2, 1)
        assert np.asarray(trace).shape == (3, 3)
        assert np.asarray(nuts_state_out).shape == (5,)
        assert np.asarray(rng_key_out).shape == (1,)

    def _assert_mini_hmc_proposal(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))

    def _assert_mini_hmc_transition(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        chain_state_out, transition_stats = result
        state = np.asarray(chain_state_out, dtype=float)
        stats = np.asarray(transition_stats, dtype=float)
        assert state.shape == (4,)
        assert stats.shape == (3,)
        assert np.all(np.isfinite(stats))
        assert 0.0 <= stats[1] <= 1.0

    def _assert_sampling_loop(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 3
        samples, trace, hmc_state_out = result
        assert np.asarray(samples).shape == (2, 1)
        assert np.asarray(trace).shape == (3, 3)
        assert np.asarray(hmc_state_out).shape == (4,)

    def _assert_nuts_tree(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.shape == (1,)
        assert np.all(np.isfinite(arr))

    def _assert_advancedhmc_tempering(result: Any) -> None:
        assert np.isclose(float(result), 1.0)

    def _assert_advancedhmc_transition(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        z_out, is_valid = result
        arr = np.asarray(z_out, dtype=float)
        assert arr.shape == (2,)
        assert np.allclose(arr, np.array([0.65, -0.6], dtype=float))
        assert is_valid is True

    def _invoke_rwmh(func: Callable[..., Any]) -> Any:
        kernel = func(target_log)
        return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

    def _invoke_rmhmc(func: Callable[..., Any]) -> Any:
        kernel = func(target_log, tensor_fn)
        return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

    def _invoke_hmc_builder(func: Callable[..., Any]) -> Any:
        kernel = func(target_log)
        return kernel(np.array([0.5, -0.5], dtype=float), np.array([1, 2], dtype=np.int64))

    def _invoke_nuts(func: Callable[..., Any]) -> Any:
        module = safe_import_module("ageoa.mcmc_foundational.mini_mcmc.nuts_llm.atoms")
        nuts_state, rng_key = module.initializenutsstate(target_log, 0.2, 0.8, 7)
        return func(nuts_state, rng_key, 2, 1)

    return {
        "ageoa.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel": ProbePlan(
            positive=ProbeCase(
                "run one deterministic AEES Metropolis-Hastings step with an explicit RNG state",
                lambda func: func(1.0, target_log, np.array([1, 2], dtype=np.int64)),
                _assert_mh_tuple,
            ),
            negative=ProbeCase(
                "reject a non-numeric tempering value",
                lambda func: func("bad", target_log, np.array([1, 2], dtype=np.int64)),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle": ProbePlan(
            positive=ProbeCase(
                "evaluate the tempered log-kernel for a candidate state",
                lambda func: func(np.array([1.0, -1.0], dtype=float), 0.5),
                _assert_scalar(0.0),
            ),
            negative=ProbeCase(
                "reject a non-numeric tempering value",
                lambda func: func(np.array([1.0, -1.0], dtype=float), "bad"),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle": ProbePlan(
            positive=ProbeCase(
                "build and run one HMC transition kernel from a target log-density oracle",
                _invoke_hmc_builder,
                _assert_mh_tuple,
            ),
            negative=ProbeCase(
                "reject a missing log-density oracle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel": ProbePlan(
            positive=ProbeCase(
                "build and run one RMHMC transition kernel from oracle and tensor functions",
                _invoke_rmhmc,
                _assert_mh_tuple,
            ),
            negative=ProbeCase(
                "reject a missing tensor oracle",
                lambda func: func(target_log, None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel": ProbePlan(
            positive=ProbeCase(
                "build and run one random-walk Metropolis kernel",
                _invoke_rwmh,
                _assert_mh_tuple,
            ),
            negative=ProbeCase(
                "reject a missing target log-kernel oracle",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic HMC kernel spec and chain state",
                lambda func: func(target_log, np.array([0.5, -0.5], dtype=float), 0.1, 4),
                _assert_hmc_init,
            ),
            negative=ProbeCase(
                "reject a non-numeric step size",
                lambda func: func(target_log, np.array([0.5, -0.5], dtype=float), "bad", 4),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng": ProbePlan(
            positive=ProbeCase(
                "initialize a deterministic mini-mcmc sampler RNG key",
                lambda func: func(7),
                _assert_hmc_rng,
            ),
            negative=ProbeCase(
                "reject a non-integer sampler seed",
                lambda func: func(7.5),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel": ProbePlan(
            positive=ProbeCase(
                "run one seeded HMC transition from an explicit chain state and kernel spec",
                lambda func: func(
                    np.array([0.5, -0.5, -0.25, -0.5, 0.5], dtype=float),
                    np.array([0.1, 4.0, 2.0], dtype=float),
                    np.array([7], dtype=np.int64),
                    target_log,
                ),
                _assert_hmc_transition,
            ),
            negative=ProbeCase(
                "reject a missing kernel specification",
                lambda func: func(
                    np.array([0.5, -0.5, -0.25, -0.5, 0.5], dtype=float),
                    None,
                    np.array([7], dtype=np.int64),
                    target_log,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc.initializehmcstate": ProbePlan(
            positive=ProbeCase(
                "initialize deterministic mini-mcmc HMC state and static kernel parameters",
                lambda func: func(target_log, np.array([0.5], dtype=float), 0.1, 4, 7),
                _assert_mini_hmc_init,
            ),
            negative=ProbeCase(
                "reject a non-numeric step size for mini-mcmc HMC initialization",
                lambda func: func(target_log, np.array([0.5], dtype=float), "bad", 4, 7),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel": ProbePlan(
            positive=ProbeCase(
                "run one deterministic leapfrog proposal step in mini-mcmc HMC",
                lambda func: func(
                    np.array([0.5, -0.125, 0.0], dtype=float),
                    np.array([0.1, 2.0, 1.0], dtype=float),
                    target_log,
                ),
                _assert_mini_hmc_proposal,
            ),
            negative=ProbeCase(
                "reject a missing kernel specification for mini-mcmc leapfrog",
                lambda func: func(np.array([0.5, -0.125, 0.0], dtype=float), None, target_log),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition": ProbePlan(
            positive=ProbeCase(
                "run one deterministic mini-mcmc HMC Metropolis transition",
                lambda func: func(
                    np.array([0.5, -0.125, 0.0, 7.0], dtype=float),
                    np.array([0.1, 2.0, 1.0], dtype=float),
                    np.array([0.51, -0.13, -0.2601, -0.50995, 0.049495, -0.05049995, -0.00499975], dtype=float),
                ),
                _assert_mini_hmc_transition,
            ),
            negative=ProbeCase(
                "reject a missing proposal state for mini-mcmc HMC transition",
                lambda func: func(np.array([0.5, -0.125, 0.0, 7.0], dtype=float), np.array([0.1, 2.0, 1.0], dtype=float), None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.hmc.runsamplingloop": ProbePlan(
            positive=ProbeCase(
                "run a tiny deterministic mini-mcmc HMC sampling loop",
                lambda func: func(np.array([0.5, -0.125, 0.0, 7.0], dtype=float), 2, 1),
                _assert_sampling_loop,
            ),
            negative=ProbeCase(
                "reject a missing initial mini-mcmc HMC state",
                lambda func: func(None, 2, 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build": ProbePlan(
            positive=ProbeCase(
                "build a deterministic shallow NUTS subtree in mini-mcmc",
                lambda func: func(
                    1,
                    0.1,
                    -1.0,
                    np.array([0.5], dtype=float),
                    target_log,
                    lambda state, step_size, direction: np.asarray(state, dtype=float) + direction * step_size,
                    1,
                ),
                _assert_nuts_tree,
            ),
            negative=ProbeCase(
                "reject a non-numeric step size for mini-mcmc NUTS tree build",
                lambda func: func(1, "bad", -1.0, np.array([0.5], dtype=float), target_log, lambda state, step_size, direction: state, 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation": ProbePlan(
            positive=ProbeCase(
                "compute a deterministic AdvancedHMC tempering factor at the midpoint step",
                lambda func: func(
                    np.array([1.0, -1.0], dtype=float),
                    np.array([0.5, 0.25], dtype=float),
                    2,
                    4,
                ),
                _assert_advancedhmc_tempering,
            ),
            negative=ProbeCase(
                "reject a missing AdvancedHMC step count",
                lambda func: func(
                    np.array([1.0, -1.0], dtype=float),
                    np.array([0.5, 0.25], dtype=float),
                    2,
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition": ProbePlan(
            positive=ProbeCase(
                "run one deterministic AdvancedHMC phase-point transition",
                lambda func: func(
                    np.array([1.0, 2.0], dtype=float),
                    np.array([0.1, -0.2], dtype=float),
                    np.array([0.5, 0.0], dtype=float),
                    1.5,
                ),
                _assert_advancedhmc_transition,
            ),
            negative=ProbeCase(
                "reject a missing AdvancedHMC tempering scale",
                lambda func: func(
                    np.array([1.0, 2.0], dtype=float),
                    np.array([0.1, -0.2], dtype=float),
                    np.array([0.5, 0.0], dtype=float),
                    None,
                ),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate": ProbePlan(
            positive=ProbeCase(
                "initialize deterministic mini-mcmc NUTS state and RNG key",
                lambda func: func(target_log, 0.2, 0.8, 7),
                _assert_nuts_init,
            ),
            negative=ProbeCase(
                "reject a non-numeric target acceptance probability",
                lambda func: func(target_log, 0.2, "bad", 7),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions": ProbePlan(
            positive=ProbeCase(
                "run a small seeded NUTS transition loop with explicit discard and collection counts",
                _invoke_nuts,
                _assert_nuts_transition,
            ),
            negative=ProbeCase(
                "reject a missing RNG key",
                lambda func: func(np.zeros(7, dtype=float), None, 2, 1),
                expect_exception=True,
            ),
            parity_used=True,
        ),
    }


PROBE_PLANS = {}
PROBE_PLANS.update(_search_plans())
PROBE_PLANS.update(_numpy_plans())
PROBE_PLANS.update(_numpy_fft_plans())
PROBE_PLANS.update(_scipy_plans())
PROBE_PLANS.update(_sorting_plans())
PROBE_PLANS.update(_scipy_sparse_graph_plans())
PROBE_PLANS.update(_scipy_stats_plans())
PROBE_PLANS.update(_scipy_integrate_plans())
PROBE_PLANS.update(_numpy_fft_v2_plans())
PROBE_PLANS.update(_numpy_search_sort_v2_plans())
PROBE_PLANS.update(_scipy_optimize_v2_plans())
PROBE_PLANS.update(get_foundation_probe_plans())
PROBE_PLANS.update(get_advancedvi_and_iqe_probe_plans())
PROBE_PLANS.update(_quant_engine_plans())
PROBE_PLANS.update(_particle_filter_and_pasqal_plans())
PROBE_PLANS.update(_rust_robotics_plans())
PROBE_PLANS.update(get_quantfin_probe_plans())
PROBE_PLANS.update(get_kalman_filter_probe_plans())
PROBE_PLANS.update(get_mcmc_foundational_probe_plans())
PROBE_PLANS.update(get_hftbacktest_and_ingest_probe_plans())
PROBE_PLANS.update(get_molecular_docking_probe_plans())
PROBE_PLANS.update(get_biosppy_probe_plans())
PROBE_PLANS.update(get_pronto_probe_plans())
PROBE_PLANS.update(_conjugate_prior_and_small_mcmc_plans())


def build_runtime_probe(record: dict[str, Any]) -> dict[str, Any]:
    """Run the safe deterministic probe plan for one atom, or skip it."""
    base = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "probe_status": "skipped",
        "positive_probe": {"status": "not_applicable"},
        "negative_probe": {"status": "not_applicable"},
        "parity_used": False,
        "skip_reason": None,
        "exception_type": None,
        "exception_message": None,
    }
    if record.get("skeleton"):
        base["probe_status"] = "skipped"
        base["skip_reason"] = "skeleton_wrapper"
        section = {
            "status": "fail",
            "findings": ["RUNTIME_NOT_IMPLEMENTED"],
            "notes": ["Wrapper is a skeleton or raises NotImplementedError."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
        }
        return section

    plan = PROBE_PLANS.get(record["atom_name"])
    if plan is None:
        section = {
            "status": "not_applicable",
            "findings": ["RUNTIME_PROBE_SKIPPED"],
            "notes": ["Atom is outside the conservative safe probe allowlist."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
            "skip_reason": "unsupported_scope",
        }
        return section

    try:
        module = safe_import_module(record["module_import_path"])
        func = getattr(module, record["wrapper_symbol"])
    except Exception as exc:  # noqa: BLE001 - want to record import failure details
        return {
            "status": "partial",
            "findings": ["RUNTIME_IMPORT_FAIL"],
            "notes": ["Import failed before the runtime probe could execute."],
            "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
            **base,
            "probe_status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:240],
        }

    positive = _run_case(func, plan.positive)
    negative = _run_case(func, plan.negative)
    findings: list[str] = []
    notes: list[str] = []

    if positive["status"] == "pass":
        findings.append("RUNTIME_PROBE_PASS")
    else:
        findings.append("RUNTIME_PROBE_FAIL")
        if positive.get("exception_type"):
            notes.append(f"Positive probe raised {positive['exception_type']}: {positive.get('exception_message', '')}")

    if negative["status"] == "pass":
        findings.append("RUNTIME_CONTRACT_NEGATIVE_PASS")
    elif negative["status"] == "fail":
        findings.append("RUNTIME_CONTRACT_NEGATIVE_FAIL")
        if negative.get("exception_type"):
            notes.append(f"Negative probe raised {negative['exception_type']}: {negative.get('exception_message', '')}")

    if positive["status"] == "fail":
        status = "fail"
    elif negative["status"] == "fail":
        status = "fail"
    elif negative["status"] == "not_applicable":
        status = "partial"
    else:
        status = "pass"

    return {
        "status": status,
        "findings": findings,
        "notes": notes,
        "source_refs": [{"path": record["module_path"], "line": record.get("wrapper_line")}],
        **base,
        "probe_status": "executed",
        "positive_probe": positive,
        "negative_probe": negative,
        "parity_used": plan.parity_used,
        "exception_type": positive.get("exception_type"),
        "exception_message": positive.get("exception_message"),
    }


def write_runtime_probe(record: dict[str, Any]) -> dict[str, Any]:
    """Run, persist, and merge runtime probe evidence for one atom."""
    section = build_runtime_probe(record)
    write_json(AUDIT_PROBES_DIR / f"{safe_atom_stem(record['atom_id'])}.json", section)
    write_evidence_section(record["atom_id"], "runtime_probe", section)
    return section
