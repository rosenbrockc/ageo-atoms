"""Conservative deterministic runtime probes for a safe atom subset."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from .io import safe_atom_stem, write_json
from .paths import AUDIT_PROBES_DIR, ROOT
from .semantics import utc_now, write_evidence_section


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


def load_module_from_file(module_import_path: str, module_file: Path) -> Any:
    """Load a module directly from its source file while preserving package-relative imports."""
    package_name = module_import_path.rsplit(".", 1)[0]
    install_package_stub(package_name)
    spec = importlib.util.spec_from_file_location(module_import_path, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_import_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_import_path] = module
    spec.loader.exec_module(module)
    return module


def safe_import_module(module_import_path: str) -> Any:
    """Import an ageoa submodule without executing ageoa.__init__."""
    install_ageoa_stub()
    try:
        return importlib.import_module(module_import_path)
    except ImportError:
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


def _assert_sparse_shape(expected_shape: tuple[int, int]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert sp.issparse(result)
        assert tuple(result.shape) == expected_shape

    return _validator


def _assert_shape(expected_shape: tuple[int, ...]) -> Callable[[Any], None]:
    def _validator(result: Any) -> None:
        assert tuple(np.asarray(result).shape) == expected_shape

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


def _scipy_optimize_v2_plans() -> dict[str, ProbePlan]:
    def _quadratic(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float((x[0] - 1.0) ** 2)

    return {
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


def _advancedvi_and_iqe_plans() -> dict[str, ProbePlan]:
    def _seeded_hawkes(func: Callable[..., Any]) -> Any:
        state = np.random.get_state()
        try:
            np.random.seed(7)
            return func(0.2, 0.3, 1.5, 2.0)
        finally:
            np.random.set_state(state)

    def _seeded_heston(func: Callable[..., Any]) -> Any:
        state = np.random.get_state()
        try:
            np.random.seed(11)
            return func(100.0, 0.04, 0.1, 1.2, 0.04, 0.3, 1.0, 0.25, 3)
        finally:
            np.random.set_state(state)

    def _assert_hawkes_points(result: Any) -> None:
        arr = np.asarray(result, dtype=float)
        assert arr.ndim == 1
        if arr.size:
            assert np.all(np.diff(arr) >= 0)
            assert np.all(arr > 0.0)
            assert np.all(arr <= 2.0)

    def _assert_heston_paths(result: Any) -> None:
        assert isinstance(result, tuple) and len(result) == 2
        s_paths, v_paths = result
        s_paths = np.asarray(s_paths, dtype=float)
        v_paths = np.asarray(v_paths, dtype=float)
        assert s_paths.shape == (3, 4)
        assert v_paths.shape == (3, 4)
        assert np.all(s_paths > 0.0)
        assert np.all(v_paths >= 0.0)

    return {
        "ageoa.advancedvi.core.evaluate_log_probability_density": ProbePlan(
            positive=ProbeCase(
                "evaluate a diagonal Gaussian log density on a small vector",
                lambda func: func(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, -1.0])),
                _assert_scalar(-2.8378770664093453),
            ),
            negative=ProbeCase(
                "reject a missing parameter vector",
                lambda func: func(None, np.array([1.0, -1.0])),
                expect_exception=True,
            ),
        ),
        "ageoa.institutional_quant_engine.hawkes_process.hawkesprocesssimulator": ProbePlan(
            positive=ProbeCase(
                "simulate a Hawkes trajectory with a fixed NumPy seed",
                _seeded_hawkes,
                _assert_hawkes_points,
            ),
            negative=ProbeCase(
                "reject a non-numeric beta value",
                lambda func: func(0.2, 0.3, "bad", 2.0),
                expect_exception=True,
            ),
        ),
        "ageoa.institutional_quant_engine.heston_model.hestonpathsampler": ProbePlan(
            positive=ProbeCase(
                "sample Heston price and variance paths with a fixed NumPy seed",
                _seeded_heston,
                _assert_heston_paths,
            ),
            negative=ProbeCase(
                "reject an invalid correlation coefficient",
                lambda func: func(100.0, 0.04, 2.0, 1.2, 0.04, 0.3, 1.0, 0.25, 3),
                expect_exception=True,
            ),
        ),
    }


def _particle_filter_and_pasqal_plans() -> dict[str, ProbePlan]:
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

    return {
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
    }


def _hftbacktest_and_ingest_family_plans() -> dict[str, ProbePlan]:
    class _DummyBlock:
        def __init__(self, value: int) -> None:
            self.value = value

    return {
        "ageoa.hftbacktest.initialize_glft_state": ProbePlan(
            positive=ProbeCase(
                "initialize GLFT state returns zero coefficients",
                lambda func: func(),
                _assert_tuple((0.0, 0.0)),
            ),
            parity_used=True,
        ),
        "ageoa.hftbacktest.update_glft_coefficients": ProbePlan(
            positive=ProbeCase(
                "GLFT coefficient update on a simple numeric input",
                lambda func: func(0.0, 0.0, 2.0, 0.1, 1.0, 2.0, 4.0),
                _assert_tuple((0.05, 0.27465307216702745)),
            ),
            negative=ProbeCase(
                "GLFT update rejects non-numeric xi",
                lambda func: func(0.0, 0.0, "bad", 0.1, 1.0, 2.0, 4.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.hftbacktest.evaluate_spread_conditions": ProbePlan(
            positive=ProbeCase(
                "spread evaluation computes a half-spread and validity flag",
                lambda func: func(2.0, 0.5, 1.5, 2.0, 0.25, 1.0),
                _assert_tuple((1.25, True)),
            ),
            negative=ProbeCase(
                "spread evaluation rejects non-numeric c1",
                lambda func: func("bad", 0.5, 1.5, 2.0, 0.25, 1.0),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.market_making_avellaneda": ProbePlan(
            positive=ProbeCase(
                "Avellaneda-Stoikov spread series over a small price vector",
                lambda func: func(np.array([100.0, 101.0, 102.0])),
                _assert_array(np.array([1.2907704244963784, 1.2907704244963784, 1.2907704244963784])),
            ),
            negative=ProbeCase(
                "Avellaneda-Stoikov rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.almgren_chriss_execution": ProbePlan(
            positive=ProbeCase(
                "Almgren-Chriss execution returns a linear liquidation trajectory",
                lambda func: func(np.array([12.0, 0.0, 0.0, 0.0])),
                _assert_array(np.array([12.0, 9.0, 6.0, 3.0])),
            ),
            negative=ProbeCase(
                "Almgren-Chriss rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.pin_informed_trading": ProbePlan(
            positive=ProbeCase(
                "PIN estimator computes order-flow imbalance",
                lambda func: func(np.array([10.0, 8.0, 3.0, 1.0])),
                _assert_array(np.array([0.6363636363636364])),
            ),
            negative=ProbeCase(
                "PIN estimator rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.institutional_quant_engine.limit_order_queue_estimator": ProbePlan(
            positive=ProbeCase(
                "queue estimator normalizes cumulative queue position",
                lambda func: func(np.array([2.0, 3.0, 5.0])),
                _assert_array(np.array([0.2, 0.5, 1.0])),
            ),
            negative=ProbeCase(
                "queue estimator rejects empty data",
                lambda func: func(np.array([])),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.mint.incremental_attention.enable_incremental_state_configuration": ProbePlan(
            positive=ProbeCase(
                "incremental attention decorates a class with state accessors",
                lambda func: func(_DummyBlock),
                _assert_type(type),
            ),
            negative=ProbeCase(
                "incremental attention rejects a missing class",
                lambda func: func(None),
                expect_exception=True,
            ),
            parity_used=True,
        ),
        "ageoa.molecular_docking.greedy_subgraph.greedy_maximum_subgraph": ProbePlan(
            positive=ProbeCase(
                "greedy maximum subgraph picks non-conflicting high-score nodes",
                lambda func: func(
                    np.array(
                        [
                            [False, True, False],
                            [True, False, False],
                            [False, False, False],
                        ],
                        dtype=bool,
                    ),
                    np.array([2.0, 1.0, 3.0]),
                ),
                _assert_array(np.array([True, False, True])),
            ),
            negative=ProbeCase(
                "greedy maximum subgraph rejects a missing score vector",
                lambda func: func(np.zeros((2, 2), dtype=bool), None),
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

    return {
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
    }


def _sklearn_image_plans() -> dict[str, ProbePlan]:
    image = np.arange(16, dtype=float).reshape(4, 4)
    volume = np.arange(8, dtype=float).reshape(2, 2, 2)
    return {
        "ageoa.sklearn.images.extract_patches_2d": ProbePlan(
            positive=ProbeCase(
                "extract 2x2 patches from a 4x4 image",
                lambda func: func(image, (2, 2)),
                _assert_shape((9, 2, 2)),
            ),
            negative=ProbeCase(
                "reject patch sizes larger than the image",
                lambda func: func(image, (5, 5)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.reconstruct_from_patches_2d": ProbePlan(
            positive=ProbeCase(
                "reconstruct a 4x4 image from extracted 2x2 patches",
                lambda func: func(np.arange(36, dtype=float).reshape(9, 2, 2), (4, 4)),
                _assert_shape((4, 4)),
            ),
            negative=ProbeCase(
                "reject incompatible patch layouts",
                lambda func: func(np.arange(16, dtype=float).reshape(4, 2, 2), (4, 4)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.img_to_graph": ProbePlan(
            positive=ProbeCase(
                "build an image graph for a 2x2x2 volume",
                lambda func: func(volume),
                _assert_sparse_shape((8, 8)),
            ),
            negative=ProbeCase(
                "reject a scalar input instead of an image volume",
                lambda func: func(np.array(1.0)),
                expect_exception=True,
            ),
        ),
        "ageoa.sklearn.images.grid_to_graph": ProbePlan(
            positive=ProbeCase(
                "build a voxel grid graph for a 2x2x2 lattice",
                lambda func: func(2, 2, 2),
                _assert_sparse_shape((8, 8)),
            ),
            negative=ProbeCase(
                "reject a zero-sized grid dimension",
                lambda func: func(0, 2, 2),
                expect_exception=True,
            ),
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
PROBE_PLANS.update(_scipy_optimize_v2_plans())
PROBE_PLANS.update(_advancedvi_and_iqe_plans())
PROBE_PLANS.update(_particle_filter_and_pasqal_plans())
PROBE_PLANS.update(_hftbacktest_and_ingest_family_plans())
PROBE_PLANS.update(_molecular_docking_plans())
PROBE_PLANS.update(_sklearn_image_plans())


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
