from __future__ import annotations

from auditlib import return_fidelity


def test_return_fidelity_flags_ignored_upstream_value(tmp_path, monkeypatch) -> None:
    module = tmp_path / "mod.py"
    module.write_text(
        """
def wrapper(x):
    value = upstream(x)
    cached = 1
    return cached
"""
    )
    monkeypatch.setattr(return_fidelity, "ROOT", tmp_path)
    result = return_fidelity.analyze_return_fidelity(
        {
            "module_path": "mod.py",
            "wrapper_symbol": "wrapper",
            "wrapper_line": 2,
            "docstring_summary": "Return the upstream value.",
        }
    )
    assert result["status"] == "fail"
    assert "RETURN_IGNORES_UPSTREAM_VALUE" in result["findings"]


def test_return_fidelity_flags_fabricated_attribute(tmp_path, monkeypatch) -> None:
    module = tmp_path / "mod.py"
    module.write_text(
        """
def wrapper(x):
    result = upstream(x)
    return result.synthetic
"""
    )
    monkeypatch.setattr(return_fidelity, "ROOT", tmp_path)
    result = return_fidelity.analyze_return_fidelity(
        {
            "module_path": "mod.py",
            "wrapper_symbol": "wrapper",
            "wrapper_line": 2,
            "docstring_summary": "Return upstream output.",
        }
    )
    assert result["status"] == "fail"
    assert "RETURN_FABRICATED_ATTRIBUTE" in result["findings"]
