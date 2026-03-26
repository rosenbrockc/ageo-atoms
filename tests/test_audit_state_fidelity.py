from __future__ import annotations

from auditlib import state_fidelity


def test_state_fidelity_flags_unknown_state_update_field(tmp_path, monkeypatch) -> None:
    wrapper_dir = tmp_path / "pkg"
    wrapper_dir.mkdir()
    (wrapper_dir / "state_models.py").write_text(
        """
class ExampleState:
    known: int
"""
    )
    (wrapper_dir / "atoms.py").write_text(
        """
def predictstep(state):
    return state.model_copy(update={"invented_field": 1})
"""
    )
    monkeypatch.setattr(state_fidelity, "ROOT", tmp_path)
    result = state_fidelity.analyze_state_fidelity(
        {
            "module_path": "pkg/atoms.py",
            "wrapper_symbol": "predictstep",
            "wrapper_line": 2,
            "stateful": True,
        }
    )
    assert result["status"] == "fail"
    assert "STATE_FABRICATED_FIELD" in result["findings"]
    assert "STATE_QUERY_MUTATION_CONFUSION" in result["findings"]


def test_state_fidelity_passes_for_state_reader(tmp_path, monkeypatch) -> None:
    wrapper_dir = tmp_path / "pkg"
    wrapper_dir.mkdir()
    (wrapper_dir / "state_models.py").write_text(
        """
class ExampleState:
    stance: float
"""
    )
    (wrapper_dir / "atoms.py").write_text(
        """
def querystance(state):
    return state.stance
"""
    )
    monkeypatch.setattr(state_fidelity, "ROOT", tmp_path)
    result = state_fidelity.analyze_state_fidelity(
        {
            "module_path": "pkg/atoms.py",
            "wrapper_symbol": "querystance",
            "wrapper_line": 2,
            "stateful": True,
        }
    )
    assert result["status"] == "pass"
    assert result["findings"] == []
