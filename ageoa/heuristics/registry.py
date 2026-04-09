from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data" / "heuristics"
FAMILY_DIR = DATA_DIR / "families"
_HEURISTIC_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_BANNED_SHARED_TOKENS = {
    "ecg",
    "ppg",
    "eeg",
    "emg",
    "pcg",
    "bpm",
    "sqi",
    "baseline",
    "wander",
    "qrs",
    "beat",
    "heart",
    "rr",
    "signal",
    "waveform",
}


class HeuristicEvidenceType(str, Enum):
    SCALAR_SCORE = "scalar_score"
    BOOLEAN_FLAG = "boolean_flag"
    DISTRIBUTION_SUMMARY = "distribution_summary"
    CATEGORICAL_LABEL = "categorical_label"
    STRUCTURED_SUMMARY = "structured_summary"


class HeuristicActionClass(str, Enum):
    PRECONDITION = "precondition"
    REPLACE_STAGE = "replace_stage"
    SPLIT_STAGE = "split_stage"
    INSERT_CORRECTION = "insert_correction"
    GATE_OR_VALIDATE = "gate_or_validate"
    SMOOTH_OR_AGGREGATE = "smooth_or_aggregate"
    BRANCH_AND_COMPARE = "branch_and_compare"


class HeuristicProducerKind(str, Enum):
    ATOM_OUTPUT = "atom_output"
    DIAGNOSTIC_ATOM = "diagnostic_atom"
    RUNTIME_TRANSFORM = "runtime_transform"
    COMPATIBILITY_MAPPING = "compatibility_mapping"


class HeuristicApplicabilityScope(str, Enum):
    CROSS_FAMILY = "cross_family"
    FAMILY_LOCAL = "family_local"
    SKELETON_LOCAL = "skeleton_local"


class HeuristicReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    citation: str = ""
    url: str = ""
    note: str = ""


class HeuristicAssetAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: str = ""
    source_kind: str = "local_asset"
    review_status: str = "draft"
    rationale: str = ""
    dejargonized_summary: str = ""
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[HeuristicReference] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "HeuristicAssetAudit":
        if not self.dejargonized_summary.strip():
            raise ValueError("Heuristic assets must include a dejargonized summary")
        if not self.references:
            raise ValueError("Heuristic assets must include at least one reference")
        return self


class HeuristicDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heuristic_id: str
    display_name: str
    dejargonized_meaning: str
    evidence_type: HeuristicEvidenceType
    value_kind: str = ""
    value_shape: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    producer_kind: HeuristicProducerKind = HeuristicProducerKind.RUNTIME_TRANSFORM
    applicability_scope: HeuristicApplicabilityScope = (
        HeuristicApplicabilityScope.CROSS_FAMILY
    )
    uncertainty_notes: list[str] = Field(default_factory=list)
    supported_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    provenance_requirements: list[str] = Field(default_factory=list)
    compatibility_aliases: list[str] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "HeuristicDefinition":
        heuristic_id = self.heuristic_id.strip()
        if not _HEURISTIC_ID_RE.fullmatch(heuristic_id):
            raise ValueError("heuristic_id must be snake_case with lowercase alphanumeric tokens")
        tokens = [token for token in heuristic_id.split("_") if token]
        banned = sorted(token for token in tokens if token in _BANNED_SHARED_TOKENS)
        if banned:
            raise ValueError(
                "Canonical heuristic identifiers must stay de-jargonized; "
                f"found domain-specific tokens: {', '.join(banned)}"
            )
        if not self.display_name.strip():
            raise ValueError("display_name is required")
        if not self.dejargonized_meaning.strip():
            raise ValueError("dejargonized_meaning is required")
        if not self.supported_action_classes:
            raise ValueError("supported_action_classes must not be empty")
        return self


class CanonicalHeuristicRegistryAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_version: str
    family: str
    domain: str
    name: str
    summary: str
    heuristics: list[HeuristicDefinition] = Field(default_factory=list)
    audit: HeuristicAssetAudit = Field(default_factory=HeuristicAssetAudit)

    @model_validator(mode="after")
    def _validate(self) -> "CanonicalHeuristicRegistryAsset":
        ids = [item.heuristic_id for item in self.heuristics]
        duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
        if duplicate_ids:
            raise ValueError(
                "Duplicate canonical heuristic identifiers are not allowed: "
                + ", ".join(duplicate_ids)
            )
        return self


class FamilyHeuristicBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heuristic_id: str
    sanctioned_producer_kinds: list[HeuristicProducerKind] = Field(default_factory=list)
    family_notes: list[str] = Field(default_factory=list)
    supported_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    action_priority: list[HeuristicActionClass] = Field(default_factory=list)
    expected_evidence_strength: Literal["weak", "moderate", "strong"] = "moderate"
    admissibility_notes: list[str] = Field(default_factory=list)
    escalation_conditions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    references: list[HeuristicReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "FamilyHeuristicBinding":
        if not _HEURISTIC_ID_RE.fullmatch(self.heuristic_id):
            raise ValueError("heuristic_id must be snake_case with lowercase alphanumeric tokens")
        if not self.family_notes:
            raise ValueError("family_notes must not be empty")
        if not self.supported_action_classes:
            raise ValueError("supported_action_classes must not be empty")
        if self.action_priority:
            allowed = set(self.supported_action_classes)
            invalid = [item for item in self.action_priority if item not in allowed]
            if invalid:
                raise ValueError(
                    "action_priority must be a subset of supported_action_classes"
                )
        if not self.references:
            raise ValueError("family heuristic bindings must include at least one reference")
        return self


class FamilyHeuristicRegistryAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_version: str
    family: str
    family_aliases: list[str] = Field(default_factory=list)
    skeleton_scope: str = ""
    domain: str
    name: str
    summary: str
    heuristic_bindings: list[FamilyHeuristicBinding] = Field(default_factory=list)
    audit: HeuristicAssetAudit = Field(default_factory=HeuristicAssetAudit)

    @model_validator(mode="after")
    def _validate(self) -> "FamilyHeuristicRegistryAsset":
        ids = [item.heuristic_id for item in self.heuristic_bindings]
        duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
        if duplicate_ids:
            raise ValueError(
                "Duplicate heuristic bindings are not allowed: " + ", ".join(duplicate_ids)
            )
        return self


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_canonical_heuristic_registry(
    path: str | Path | None = None,
) -> CanonicalHeuristicRegistryAsset:
    registry_path = Path(path) if path is not None else DATA_DIR / "canonical_registry.json"
    return CanonicalHeuristicRegistryAsset.model_validate(_load_json(registry_path))


def load_family_heuristic_registry(path: str | Path) -> FamilyHeuristicRegistryAsset:
    return FamilyHeuristicRegistryAsset.model_validate(_load_json(Path(path)))


def load_family_heuristic_registries(
    directory: str | Path | None = None,
) -> tuple[FamilyHeuristicRegistryAsset, ...]:
    registry_dir = Path(directory) if directory is not None else FAMILY_DIR
    if not registry_dir.exists():
        return tuple()
    return tuple(
        load_family_heuristic_registry(path)
        for path in sorted(registry_dir.glob("*.json"))
    )


def known_heuristic_ids(
    registry: CanonicalHeuristicRegistryAsset | None = None,
) -> tuple[str, ...]:
    canonical = registry or load_canonical_heuristic_registry()
    return tuple(sorted(item.heuristic_id for item in canonical.heuristics))


HeuristicDefinition.model_rebuild()
CanonicalHeuristicRegistryAsset.model_rebuild()
FamilyHeuristicBinding.model_rebuild()
FamilyHeuristicRegistryAsset.model_rebuild()
