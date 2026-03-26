"""Data models for deterministic audit tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UpstreamMapping:
    """Deterministic upstream mapping for one atom."""

    repo: str | None = None
    module: str | None = None
    function: str | None = None
    language: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)


@dataclass
class AtomRecord:
    """Inventory record for one registered atom."""

    atom_id: str
    atom_name: str
    atom_key: str
    module_import_path: str
    module_path: str
    wrapper_symbol: str
    wrapper_line: int
    domain_family: str
    module_family: str
    source_kind: str
    risk_tier: str
    upstream_symbols: dict[str, Any]
    upstream_version: str | None
    source_revision: str | None
    review_basis_at: str | None
    stateful: bool
    ffi: bool
    skeleton: bool
    has_state_models: bool
    has_witnesses: bool
    has_cdg: bool
    has_references: bool
    has_parity_tests: bool
    structural_status: str
    runtime_status: str
    semantic_status: str
    developer_semantics_status: str
    parity_test_status: str
    references_status: str
    overall_verdict: str
    acceptability_score: int | None
    acceptability_band: str | None
    max_reviewable_verdict: str | None
    blocking_findings: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    argument_names: list[str] = field(default_factory=list)
    required_parameter_names: list[str] = field(default_factory=list)
    argument_details: list[dict[str, Any]] = field(default_factory=list)
    return_annotation: str | None = None
    decorator_count: int = 0
    require_count: int = 0
    ensure_count: int = 0
    witness_binding: str | None = None
    placeholder_witness: bool = False
    has_docstring: bool = False
    docstring_summary: str | None = None
    has_weak_types: bool = False
    weak_type_annotations: list[str] = field(default_factory=list)
    uses_varargs: bool = False
    uses_kwargs: bool = False
    inventory_notes: list[str] = field(default_factory=list)
    stateful_kind: str = "none"
    stochastic: bool = False
    procedural: bool = False
    authoritative_sources: list[dict[str, Any]] = field(default_factory=list)
    risk_reasons: list[str] = field(default_factory=list)
    status_basis: dict[str, Any] = field(default_factory=dict)
    structural_findings: list[str] = field(default_factory=list)
    structural_finding_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)
