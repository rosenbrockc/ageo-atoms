"""Deterministic signature fidelity checks."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .io import safe_atom_stem, write_json
from .paths import AUDIT_EVIDENCE_DIR
from .upstream import UpstreamMapping, resolve_upstream_signature


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _status_for_findings(mapping_found: bool, findings: list[str], upstream_signature: dict[str, Any] | None) -> str:
    if not mapping_found:
        return "unknown"
    if upstream_signature is None:
        return "unknown"
    critical = {"FIDELITY_SIGNATURE_MISSING_REQUIRED", "FIDELITY_SIGNATURE_INVENTED_PARAMETER"}
    if any(code in critical for code in findings):
        return "fail"
    if findings:
        return "partial"
    return "pass"


def build_signature_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Compare one wrapper signature to its mapped upstream symbol."""
    mapping_dict = record.get("upstream_symbols") or {}
    mapping = UpstreamMapping(**mapping_dict) if mapping_dict else None
    wrapper_params = list(record.get("argument_names", []))
    wrapper_required = list(record.get("required_parameter_names", []))
    findings: list[str] = []
    notes: list[str] = []
    upstream_signature: dict[str, Any] | None = None
    source_kind: str | None = None
    source_ref: str | None = None

    if mapping is None or not mapping.module or not mapping.function:
        findings.append("FIDELITY_UPSTREAM_UNMAPPED")
    else:
        upstream_signature, source_kind, source_ref = resolve_upstream_signature(mapping)
        if upstream_signature is None:
            findings.append("FIDELITY_UPSTREAM_SIGNATURE_UNAVAILABLE")
        else:
            upstream_params = upstream_signature["parameter_names"]
            upstream_required = upstream_signature["required_parameter_names"]
            missing_required = [name for name in upstream_required if name not in wrapper_params]
            invented = [name for name in wrapper_params if name not in upstream_params]
            if missing_required:
                findings.append("FIDELITY_SIGNATURE_MISSING_REQUIRED")
                notes.append("missing_required=" + ",".join(missing_required))
            if invented:
                findings.append("FIDELITY_SIGNATURE_INVENTED_PARAMETER")
                notes.append("invented_parameters=" + ",".join(invented))
            if not missing_required and not invented and wrapper_params != upstream_params:
                findings.append("FIDELITY_SIGNATURE_ORDER_MISMATCH")
            if wrapper_required != [name for name in upstream_required if name in wrapper_params]:
                findings.append("FIDELITY_REQUIREDNESS_MISMATCH")

    if record.get("uses_varargs"):
        findings.append("FIDELITY_PUBLIC_VARARGS")
    if record.get("uses_kwargs"):
        findings.append("FIDELITY_PUBLIC_KWARGS")
    if record.get("has_weak_types"):
        findings.append("FIDELITY_WEAK_TYPES")

    evidence = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "atom_key": record["atom_key"],
        "mapping_found": mapping is not None and bool(mapping.module and mapping.function),
        "upstream_mapping": {} if mapping is None else asdict(mapping),
        "wrapper_signature": {
            "parameter_names": wrapper_params,
            "required_parameter_names": wrapper_required,
            "return_annotation": record.get("return_annotation"),
        },
        "upstream_signature": upstream_signature,
        "upstream_signature_source": source_kind,
        "upstream_signature_ref": source_ref,
        "findings": sorted(set(findings)),
        "notes": notes,
    }
    evidence["fidelity_status"] = _status_for_findings(
        mapping_found=evidence["mapping_found"],
        findings=evidence["findings"],
        upstream_signature=upstream_signature,
    )
    if upstream_signature is not None:
        evidence["missing_required_parameters"] = [
            name for name in upstream_signature["required_parameter_names"] if name not in wrapper_params
        ]
        evidence["invented_parameters"] = [
            name for name in wrapper_params if name not in upstream_signature["parameter_names"]
        ]
    else:
        evidence["missing_required_parameters"] = []
        evidence["invented_parameters"] = []
    return evidence


def write_signature_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Write per-atom signature fidelity evidence."""
    evidence = build_signature_evidence(record)
    write_json(AUDIT_EVIDENCE_DIR / f"{safe_atom_stem(record['atom_id'])}.json", evidence)
    return evidence
