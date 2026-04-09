"""Manifest validation for deterministic audit tooling."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

from .heuristic_assets import evaluate_heuristic_asset
from .io import write_json
from .paths import AUDIT_MANIFEST_VALIDATION_PATH, ROOT

REQUIRED_TOP_LEVEL = {"schema_version", "metadata", "summary", "atoms", "inventory_errors"}
REQUIRED_METADATA = {"generated_at", "repo", "generator", "phase"}
REQUIRED_SUMMARY = {
    "atom_count",
    "inventory_error_count",
    "family_counts",
    "source_kind_counts",
    "risk_tier_counts",
    "unmapped_upstream_count",
}
REQUIRED_ATOM_FIELDS = {
    "atom_id",
    "atom_key",
    "atom_name",
    "module_path",
    "wrapper_symbol",
    "wrapper_line",
    "source_kind",
    "risk_tier",
    "stateful_kind",
    "stochastic",
    "procedural",
    "authoritative_sources",
    "risk_reasons",
    "status_basis",
}
ALLOWED_SOURCE_KINDS = {"hand_written", "generated_ingest", "refined_ingest", "skeleton"}
ALLOWED_RISK_TIERS = {"high", "medium", "low"}
ALLOWED_STATEFUL_KINDS = {"none", "explicit_state_model", "argument_state", "return_state", "implicit_stateful"}
HEURISTIC_ERROR_CODES = {"HEURISTIC_DOCSTRING_MISSING", "HEURISTIC_DOCSTRING_PLACEHOLDER"}


def _finding(level: str, code: str, message: str, *, atom_id: str | None = None, path: str | None = None) -> dict[str, Any]:
    finding = {"level": level, "code": code, "message": message}
    if atom_id:
        finding["atom_id"] = atom_id
    if path:
        finding["path"] = path
    return finding


def _validate_top_level(payload: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    missing = REQUIRED_TOP_LEVEL - set(payload)
    for key in sorted(missing):
        findings.append(_finding("error", "MANIFEST_TOP_LEVEL_MISSING", f"Missing top-level key '{key}'"))
    metadata = payload.get("metadata", {})
    for key in sorted(REQUIRED_METADATA - set(metadata)):
        findings.append(_finding("error", "MANIFEST_METADATA_MISSING", f"Missing metadata key '{key}'"))
    summary = payload.get("summary", {})
    for key in sorted(REQUIRED_SUMMARY - set(summary)):
        findings.append(_finding("error", "MANIFEST_SUMMARY_MISSING", f"Missing summary key '{key}'"))


def _validate_wrapper_reference(record: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    module_path = record.get("module_path")
    if not module_path:
        return
    wrapper_path = ROOT / module_path
    atom_id = record.get("atom_id")
    if not wrapper_path.exists():
        findings.append(_finding("error", "MANIFEST_PATH_MISSING", f"Wrapper path does not exist: {module_path}", atom_id=atom_id, path=module_path))
        return
    try:
        tree = ast.parse(wrapper_path.read_text())
    except SyntaxError as exc:
        findings.append(_finding("error", "MANIFEST_WRAPPER_PARSE_FAIL", f"Wrapper path is not parseable: {exc.msg}", atom_id=atom_id, path=module_path))
        return
    target_line = record.get("wrapper_line")
    target_name = record.get("wrapper_symbol")
    matched = any(
        isinstance(node, ast.FunctionDef) and node.name == target_name and node.lineno == target_line
        for node in tree.body
    )
    if not matched:
        findings.append(
            _finding(
                "error",
                "MANIFEST_WRAPPER_LINE_MISMATCH",
                f"Function '{target_name}' was not found at line {target_line}",
                atom_id=atom_id,
                path=module_path,
            )
        )


def validate_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a manifest payload and return a machine-readable report."""
    findings: list[dict[str, Any]] = []
    _validate_top_level(payload, findings)

    atoms = payload.get("atoms", [])
    atom_ids = Counter(record.get("atom_id") for record in atoms)
    atom_names = Counter(record.get("atom_name") for record in atoms)
    heuristic_atom_count = 0
    heuristic_error_count = 0
    heuristic_warning_count = 0

    for atom_id, count in atom_ids.items():
        if atom_id and count > 1:
            findings.append(_finding("error", "MANIFEST_DUPLICATE_ATOM_ID", f"Duplicate atom_id '{atom_id}' appears {count} times", atom_id=atom_id))
    for atom_name, count in atom_names.items():
        if atom_name and count > 1:
            findings.append(_finding("warning", "MANIFEST_DUPLICATE_ATOM_NAME", f"Duplicate atom_name '{atom_name}' appears {count} times"))

    for record in atoms:
        atom_id = record.get("atom_id")
        missing = REQUIRED_ATOM_FIELDS - set(record)
        for key in sorted(missing):
            findings.append(_finding("error", "MANIFEST_ATOM_FIELD_MISSING", f"Missing atom field '{key}'", atom_id=atom_id))
        source_kind = record.get("source_kind")
        if source_kind not in ALLOWED_SOURCE_KINDS:
            findings.append(_finding("error", "MANIFEST_BAD_SOURCE_KIND", f"Invalid source_kind '{source_kind}'", atom_id=atom_id))
        risk_tier = record.get("risk_tier")
        if risk_tier not in ALLOWED_RISK_TIERS:
            findings.append(_finding("error", "MANIFEST_BAD_RISK_TIER", f"Invalid risk_tier '{risk_tier}'", atom_id=atom_id))
        stateful_kind = record.get("stateful_kind")
        if stateful_kind not in ALLOWED_STATEFUL_KINDS:
            findings.append(_finding("error", "MANIFEST_BAD_STATEFUL_KIND", f"Invalid stateful_kind '{stateful_kind}'", atom_id=atom_id))
        if not record.get("atom_key"):
            findings.append(_finding("error", "MANIFEST_BAD_ATOM_KEY", "atom_key must be non-empty", atom_id=atom_id))
        if not isinstance(record.get("authoritative_sources"), list):
            findings.append(_finding("error", "MANIFEST_BAD_AUTHORITATIVE_SOURCES", "authoritative_sources must be a list", atom_id=atom_id))
        _validate_wrapper_reference(record, findings)

        heuristic = evaluate_heuristic_asset(record)
        if heuristic["status"] != "not_applicable":
            heuristic_atom_count += 1
            for code in heuristic["findings"]:
                level = "error" if code in HEURISTIC_ERROR_CODES else "warning"
                if level == "error":
                    heuristic_error_count += 1
                else:
                    heuristic_warning_count += 1
                findings.append(
                    _finding(
                        level,
                        code,
                        "; ".join(heuristic["notes"]) or "Heuristic asset documentation requires review.",
                        atom_id=atom_id,
                        path=record.get("module_path"),
                    )
                )

    errors = [finding for finding in findings if finding["level"] == "error"]
    warnings = [finding for finding in findings if finding["level"] == "warning"]
    return {
        "schema_version": "1.0",
        "ok": not errors,
        "summary": {
            "atom_count": len(atoms),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "heuristic_atom_count": heuristic_atom_count,
            "heuristic_error_count": heuristic_error_count,
            "heuristic_warning_count": heuristic_warning_count,
        },
        "findings": findings,
    }


def write_validation_report(report: dict[str, Any]) -> None:
    """Write the manifest validation report."""
    write_json(AUDIT_MANIFEST_VALIDATION_PATH, report)
