from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import HeuristicDefinition, HeuristicReference

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ATOM_ROOT = PACKAGE_ROOT / "ageoa"


class HeuristicOutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_name: str
    output_path: str = ""
    role: Literal["advisory", "gating", "structural"] = "advisory"
    semantic_kind: str = ""
    heuristic: HeuristicDefinition
    provenance_notes: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "HeuristicOutputContract":
        if not self.output_name.strip():
            raise ValueError("output_name is required")
        return self


class AtomHeuristicMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_fqdn: str
    summary: str
    dejargonized_summary: str
    heuristic_outputs: list[HeuristicOutputContract] = Field(default_factory=list)
    references: list[HeuristicReference] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    maintainers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "AtomHeuristicMetadata":
        if not self.dejargonized_summary.strip():
            raise ValueError("Atom heuristic metadata must include a dejargonized summary")
        if not self.references:
            raise ValueError("Atom heuristic metadata must include at least one reference")
        output_names = [item.output_name for item in self.heuristic_outputs]
        duplicate_output_names = sorted(
            {name for name in output_names if output_names.count(name) > 1}
        )
        if duplicate_output_names:
            raise ValueError(
                "Duplicate heuristic output names are not allowed: "
                + ", ".join(duplicate_output_names)
            )
        heuristic_ids = [item.heuristic.heuristic_id for item in self.heuristic_outputs]
        duplicate_ids = sorted(
            {name for name in heuristic_ids if heuristic_ids.count(name) > 1}
        )
        if duplicate_ids:
            raise ValueError(
                "Duplicate canonical heuristic identifiers are not allowed: "
                + ", ".join(duplicate_ids)
            )
        return self


def load_atom_heuristic_metadata_records(
    path: str | Path,
) -> tuple[AtomHeuristicMetadata, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        payloads = raw
    elif isinstance(raw, dict) and isinstance(raw.get("records"), list):
        payloads = raw.get("records", [])
    else:
        payloads = [raw]
    return tuple(
        AtomHeuristicMetadata.model_validate(item)
        for item in payloads
        if isinstance(item, dict)
    )


def load_atom_heuristic_metadata(path: str | Path) -> AtomHeuristicMetadata:
    records = load_atom_heuristic_metadata_records(path)
    if not records:
        raise ValueError(f"No atom heuristic metadata records found in {path}")
    if len(records) != 1:
        raise ValueError(
            "load_atom_heuristic_metadata expects exactly one record; "
            f"found {len(records)} in {path}"
        )
    return records[0]


@lru_cache(maxsize=1)
def load_all_atom_heuristic_metadata() -> tuple[AtomHeuristicMetadata, ...]:
    records: list[AtomHeuristicMetadata] = []
    for path in sorted(ATOM_ROOT.glob("**/heuristic_metadata.json")):
        records.extend(load_atom_heuristic_metadata_records(path))
    return tuple(records)


def load_atom_heuristic_metadata_for_fqdn(atom_fqdn: str) -> AtomHeuristicMetadata | None:
    target = str(atom_fqdn or "").strip()
    if not target:
        return None
    for record in load_all_atom_heuristic_metadata():
        if record.atom_fqdn == target:
            return record
    return None
