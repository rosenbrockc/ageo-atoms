from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_signal_processing_heuristic_registry_stays_generic() -> None:
    registry = _load(ROOT / "data" / "heuristic_registries" / "signal_processing.json")

    assert registry["asset_id"] == "family.signal_processing.heuristics.v1"
    assert registry["family"] == "signal_processing"
    assert set(registry["family_aliases"]) == {"signal_event_rate", "signal_detect_measure"}
    assert registry["audit"]["review_status"] == "transitional"

    heuristic_ids = {entry["heuristic_id"] for entry in registry["entries"]}
    assert heuristic_ids == {
        "boundary_discontinuity",
        "quality_instability",
        "interval_instability",
    }

    for entry in registry["entries"]:
        assert entry["admissibility_notes"]
        assert entry["escalation_conditions"]
        assert entry["family_notes"]
        assert all(
            token not in entry["heuristic_id"]
            for token in ("ecg", "heart", "bpm", "signal")
        )


def test_signal_processing_atom_heuristic_metadata_is_cross_family() -> None:
    metadata = _load(
        ROOT / "ageoa" / "biosppy" / "ecg_zz2018_d12" / "heuristic_metadata.json"
    )

    assert metadata["atom_fqdn"] == "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi"
    assert metadata["dejargonized_summary"]
    assert metadata["references"]
    assert metadata["maintainers"] == ["ageo-atoms"]

    output = metadata["heuristic_outputs"][0]
    heuristic = output["heuristic"]
    assert output["role"] == "gating"
    assert output["semantic_kind"] == "categorical_label"
    assert heuristic["heuristic_id"] == "quality_instability"
    assert heuristic["applicability_scope"] == "cross_family"
    assert heuristic["supported_action_classes"] == [
        "gate_or_validate",
        "branch_and_compare",
    ]
    assert heuristic["family_notes"]
    assert heuristic["compatibility_aliases"] == [
        "signal_quality_variance",
        "signal_quality_kurtosis_cv",
    ]
