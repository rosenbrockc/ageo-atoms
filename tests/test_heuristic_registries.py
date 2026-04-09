from __future__ import annotations

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
AGEOA_DIR = ROOT / "ageoa"
HEURISTICS_DIR = AGEOA_DIR / "heuristics"


def _ensure_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules.setdefault(name, module)


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("ageoa", AGEOA_DIR)
_ensure_package("ageoa.heuristics", HEURISTICS_DIR)
heuristics_registry = _load_module(
    "ageoa.heuristics.registry", HEURISTICS_DIR / "registry.py"
)
heuristics_atom_metadata = _load_module(
    "ageoa.heuristics.atom_metadata", HEURISTICS_DIR / "atom_metadata.py"
)

CanonicalHeuristicRegistryAsset = heuristics_registry.CanonicalHeuristicRegistryAsset
FamilyHeuristicBinding = heuristics_registry.FamilyHeuristicBinding
FamilyHeuristicRegistryAsset = heuristics_registry.FamilyHeuristicRegistryAsset
HeuristicActionClass = heuristics_registry.HeuristicActionClass
HeuristicDefinition = heuristics_registry.HeuristicDefinition
load_canonical_heuristic_registry = heuristics_registry.load_canonical_heuristic_registry
load_family_heuristic_registries = heuristics_registry.load_family_heuristic_registries
load_atom_heuristic_metadata_records = (
    heuristics_atom_metadata.load_atom_heuristic_metadata_records
)
load_atom_heuristic_metadata_for_fqdn = (
    heuristics_atom_metadata.load_atom_heuristic_metadata_for_fqdn
)

BANNED_TOKENS = {"ecg", "ppg", "eeg", "emg", "pcg", "bpm", "beat", "heart", "rr", "signal", "waveform"}


def test_canonical_registry_is_loaded_and_dejargonized() -> None:
    asset = load_canonical_heuristic_registry()

    assert isinstance(asset, CanonicalHeuristicRegistryAsset)
    assert asset.asset_id == "family.shared_heuristics.v1"
    assert asset.audit.references
    assert len(asset.heuristics) >= 10

    ids = [item.heuristic_id for item in asset.heuristics]
    assert len(ids) == len(set(ids))
    assert all(item.display_name for item in asset.heuristics)
    assert all(item.dejargonized_meaning for item in asset.heuristics)
    assert all(item.supported_action_classes for item in asset.heuristics)
    assert all(set(item.heuristic_id.split("_")).isdisjoint(BANNED_TOKENS) for item in asset.heuristics)


@pytest.mark.parametrize(
    "heuristic_id",
    ["rr_irregularity", "signal_quality_variance", "ecg_peak_drift"],
)
def test_canonical_heuristic_definition_rejects_family_jargon_in_identifier(heuristic_id: str) -> None:
    with pytest.raises(ValidationError):
        HeuristicDefinition.model_validate(
            {
                "heuristic_id": heuristic_id,
                "display_name": "Bad Example",
                "dejargonized_meaning": "Should not be accepted.",
                "evidence_type": "scalar_score",
                "supported_action_classes": ["gate_or_validate"],
            }
        )


def test_family_registries_reference_canonical_heuristics_without_redefinition() -> None:
    canonical = load_canonical_heuristic_registry()
    canonical_map = {item.heuristic_id: item for item in canonical.heuristics}
    registries = load_family_heuristic_registries()

    assert {registry.family for registry in registries} == {
        "signal_event_rate",
        "divide_and_conquer",
        "sequential_filter",
    }

    shared_ids: set[str] = set()
    for registry in registries:
        assert isinstance(registry, FamilyHeuristicRegistryAsset)
        assert registry.audit.references
        assert registry.audit.dejargonized_summary
        for binding in registry.heuristic_bindings:
            assert isinstance(binding, FamilyHeuristicBinding)
            assert binding.heuristic_id in canonical_map
            assert binding.family_notes
            assert binding.supported_action_classes
            assert binding.references
            shared_ids.add(binding.heuristic_id)
            canonical_actions = set(canonical_map[binding.heuristic_id].supported_action_classes)
            assert set(binding.supported_action_classes).issubset(canonical_actions)

    assert "constraint_violation_risk" in shared_ids
    assert len(shared_ids) >= 6


def test_family_binding_model_forbids_canonical_field_leaks() -> None:
    with pytest.raises(ValidationError):
        FamilyHeuristicBinding.model_validate(
            {
                "heuristic_id": "coverage_fragmentation",
                "display_name": "Leaky Field",
                "family_notes": ["Should not accept shared-field redefinition."],
                "supported_action_classes": ["gate_or_validate"],
                "references": [{"title": "Reference"}],
            }
        )


def test_signal_family_registry_exposes_alias_and_generic_actions() -> None:
    registries = {registry.family: registry for registry in load_family_heuristic_registries()}
    signal_registry = registries["signal_event_rate"]

    assert "signal_detect_measure" in signal_registry.family_aliases
    interval_entry = next(
        entry
        for entry in signal_registry.heuristic_bindings
        if entry.heuristic_id == "interval_instability"
    )
    assert interval_entry.action_priority == [
        HeuristicActionClass.INSERT_CORRECTION,
        HeuristicActionClass.SMOOTH_OR_AGGREGATE,
    ]
    assert interval_entry.admissibility_notes
    assert interval_entry.escalation_conditions


def test_signal_atom_metadata_loads_and_keeps_canonical_heuristic_generic() -> None:
    metadata = load_atom_heuristic_metadata_for_fqdn(
        "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi"
    )

    assert metadata is not None
    assert metadata.dejargonized_summary
    assert metadata.references
    output = metadata.heuristic_outputs[0]
    heuristic = output.heuristic
    assert heuristic.heuristic_id == "quality_instability"
    assert heuristic.applicability_scope == "cross_family"
    assert heuristic.supported_action_classes == [
        "gate_or_validate",
        "branch_and_compare",
    ]


def test_multi_record_signal_metadata_loads_multiple_atoms_from_one_asset() -> None:
    records = load_atom_heuristic_metadata_records(
        ROOT / "ageoa" / "biosppy" / "ecg_zz2018" / "heuristic_metadata.json"
    )

    fqdns = {record.atom_fqdn for record in records}
    assert {
        "ageoa.biosppy.ecg_zz2018.calculatecompositesqi_zz2018",
        "ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi",
        "ageoa.biosppy.ecg_zz2018.calculatebeatagreementsqi",
    }.issubset(fqdns)


def test_non_signal_atom_metadata_uses_shared_residual_heuristic() -> None:
    metadata = load_atom_heuristic_metadata_for_fqdn(
        "ageoa.kalman_filters.filter_rs.evaluatemeasurementoracle"
    )

    assert metadata is not None
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == (
        "residual_structure_after_transform"
    )
    assert metadata.heuristic_outputs[0].heuristic.applicability_scope == "cross_family"


def test_sequential_filter_registry_proves_second_family_rollout() -> None:
    registries = {registry.family: registry for registry in load_family_heuristic_registries()}
    sequential = registries["sequential_filter"]

    assert set(sequential.family_aliases) == {"kalman_filter", "particle_filter"}
    entry_ids = {entry.heuristic_id for entry in sequential.heuristic_bindings}
    assert "residual_structure_after_transform" in entry_ids
    assert "alignment_error" in entry_ids
    assert "constraint_violation_risk" in entry_ids
