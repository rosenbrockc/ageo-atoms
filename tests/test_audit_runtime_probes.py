from __future__ import annotations

from types import SimpleNamespace

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str = "ageoa.algorithms.search", wrapper_symbol: str = "binary_search") -> dict:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def test_runtime_probe_passes_for_safe_real_atom() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.algorithms.search.binary_search", "ageoa.algorithms.search", "binary_search")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_skips_unsupported_atom() -> None:
    probe = runtime_probes.build_runtime_probe(_record("ageoa.example.unsupported", "ageoa.example", "atom"))
    assert probe["status"] == "not_applicable"
    assert probe["skip_reason"] == "unsupported_scope"
    assert probe["findings"] == ["RUNTIME_PROBE_SKIPPED"]


def test_runtime_probe_passes_for_numpy_fft() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.numpy.fft.fft", "ageoa.numpy.fft", "fft")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sparse_graph_laplacian() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.scipy.sparse_graph.graph_laplacian", "ageoa.scipy.sparse_graph", "graph_laplacian")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_sklearn_image_grid_to_graph() -> None:
    probe = runtime_probes.build_runtime_probe(
        _record("ageoa.sklearn.images.grid_to_graph", "ageoa.sklearn.images.atoms", "grid_to_graph")
    )
    assert probe["status"] == "pass"
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_records_positive_failure(monkeypatch) -> None:
    atom_name = "ageoa.example.fail"
    monkeypatch.setitem(
        runtime_probes.PROBE_PLANS,
        atom_name,
        runtime_probes.ProbePlan(
            positive=runtime_probes.ProbeCase("always fails", lambda func: func(), None),
            negative=runtime_probes.ProbeCase("negative passes", lambda func: (_ for _ in ()).throw(ValueError("bad")), expect_exception=True),
        ),
    )
    monkeypatch.setattr(runtime_probes, "safe_import_module", lambda _: SimpleNamespace(atom=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    probe = runtime_probes.build_runtime_probe(_record(atom_name, "ageoa.example", "atom"))
    assert probe["status"] == "fail"
    assert "RUNTIME_PROBE_FAIL" in probe["findings"]
    assert probe["positive_probe"]["exception_type"] == "RuntimeError"
