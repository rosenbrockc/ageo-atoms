from __future__ import annotations

from auditlib.heuristic_assets import (
    validate_atom_heuristic_metadata_asset,
    validate_canonical_heuristic_asset,
    validate_family_heuristic_registry_asset,
)


def test_canonical_heuristic_audit_rejects_domain_jargon_in_shared_fields() -> None:
    findings = validate_canonical_heuristic_asset(
        {
            "heuristics": [
                {
                    "heuristic_id": "interval_instability",
                    "display_name": "ECG Interval Instability",
                    "dejargonized_meaning": "A heart-rate oriented signal becomes unstable.",
                }
            ]
        }
    )

    codes = {finding["code"] for finding in findings}
    assert "HEURISTIC_CANONICAL_DISPLAY_JARGON" in codes
    assert "HEURISTIC_CANONICAL_MEANING_JARGON" in codes


def test_family_registry_audit_rejects_canonical_redefinition() -> None:
    findings = validate_family_heuristic_registry_asset(
        {
            "heuristic_bindings": [
                {
                    "heuristic_id": "interval_instability",
                    "display_name": "Bad Redefinition",
                    "family_notes": ["Local interpretation."],
                }
            ]
        }
    )

    codes = {finding["code"] for finding in findings}
    assert "HEURISTIC_FAMILY_REDEFINITION" in codes


def test_atom_metadata_audit_warns_when_cross_family_meaning_reintroduces_jargon() -> None:
    findings = validate_atom_heuristic_metadata_asset(
        {
            "atom_fqdn": "ageoa.example.atom",
            "heuristic_outputs": [
                {
                    "heuristic": {
                        "heuristic_id": "quality_instability",
                        "applicability_scope": "cross_family",
                        "dejargonized_meaning": "This ECG quality score predicts heart signal brittleness.",
                    }
                }
            ],
        }
    )

    codes = {finding["code"] for finding in findings}
    assert "HEURISTIC_METADATA_CANONICAL_MEANING_JARGON" in codes
