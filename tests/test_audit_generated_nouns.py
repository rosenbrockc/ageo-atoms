from __future__ import annotations

from auditlib import generated_nouns


def test_generated_nouns_flags_undocumented_state_key(tmp_path, monkeypatch) -> None:
    wrapper = tmp_path / "mod.py"
    wrapper.write_text(
        """
def wrapper(state):
    return state.model_copy(update={"latent_inventory": 1})
"""
    )
    allowlist = tmp_path / "generated_nouns.json"
    allowlist.write_text('{"allowlisted_nouns": ["metadata"]}\n')
    monkeypatch.setattr(generated_nouns, "ROOT", tmp_path)
    monkeypatch.setattr(generated_nouns, "AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH", allowlist)
    monkeypatch.setattr(generated_nouns, "load_atom_evidence", lambda _: {"upstream_mapping": {"function": "update_state"}})
    result = generated_nouns.analyze_generated_nouns(
        {
            "atom_id": "atom",
            "module_path": "mod.py",
            "wrapper_symbol": "wrapper",
            "wrapper_line": 2,
            "docstring_summary": "Update state.",
            "upstream_symbols": {},
        }
    )
    assert result["status"] == "partial"
    assert "NOUN_UNDOCUMENTED_STATE" in result["findings"]


def test_generated_nouns_respects_allowlist(tmp_path, monkeypatch) -> None:
    wrapper = tmp_path / "mod.py"
    wrapper.write_text(
        """
def wrapper():
    return {"metadata": 1}
"""
    )
    allowlist = tmp_path / "generated_nouns.json"
    allowlist.write_text('{"allowlisted_nouns": ["metadata"]}\n')
    monkeypatch.setattr(generated_nouns, "ROOT", tmp_path)
    monkeypatch.setattr(generated_nouns, "AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH", allowlist)
    monkeypatch.setattr(generated_nouns, "load_atom_evidence", lambda _: {})
    result = generated_nouns.analyze_generated_nouns(
        {
            "atom_id": "atom",
            "module_path": "mod.py",
            "wrapper_symbol": "wrapper",
            "wrapper_line": 2,
            "docstring_summary": "Return metadata output.",
            "upstream_symbols": {},
        }
    )
    assert result["status"] == "pass"
    assert "NOUN_ALLOWLISTED_DERIVATION" in result["findings"]
    assert "NOUN_UNDOCUMENTED_OUTPUT" not in result["findings"]
