"""Unified structural audit orchestration."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit as prompt_audit
import type_and_isomorphism_audit as cdg_audit

from .io import ensure_dir, write_json
from .paths import (
    AUDIT_DIR,
    AUDIT_STRUCTURAL_FINDINGS_CSV_PATH,
    AUDIT_STRUCTURAL_REPORT_PATH,
    ROOT,
)

VERIFY_SCRIPT = ROOT.parent / "ageo-matcher" / "scripts" / "verify_atoms_repo.py"

FILE_RULE_MAP = {
    "A-PARSE": ("STRUCT_PARSE_FAIL", "error"),
    "A-REG": ("STRUCT_REGISTER_MISSING", "error"),
    "A-CONTRACT": ("STRUCT_REQUIRE_OR_ENSURE_MISSING", "warning"),
    "A-KWARGS": ("STRUCT_PUBLIC_KWARGS", "warning"),
    "A-DOC": ("STRUCT_DOCSTRING_MISSING", "warning"),
    "A-IMPORT": ("STRUCT_IMPORT_HEAVY", "warning"),
    "A-TYPE": ("STRUCT_WEAK_TYPES", "warning"),
    "W-MISSING": ("STRUCT_WITNESS_FILE_MISSING", "error"),
    "W-PARSE": ("STRUCT_WITNESS_PARSE_FAIL", "error"),
    "W-TYPE": ("STRUCT_WITNESS_TYPE_MISSING", "warning"),
    "W-PURE": ("STRUCT_WITNESS_IMPURE", "warning"),
    "W-NONE": ("STRUCT_WITNESS_RETURNS_NONE", "warning"),
    "C-MISSING": ("STRUCT_CDG_MISSING", "error"),
    "C-PARSE": ("STRUCT_CDG_INVALID", "error"),
    "C-LEAF": ("STRUCT_CDG_INVALID", "error"),
    "C-CONSTRAINT": ("STRUCT_CDG_INVALID", "warning"),
    "C-DECOMP": ("STRUCT_CDG_INVALID", "error"),
    "C-DEPTH": ("STRUCT_CDG_INVALID", "warning"),
    "C-SELFLOOP": ("STRUCT_CDG_INVALID", "error"),
    "C-DUP": ("STRUCT_CDG_INVALID", "error"),
    "C-CYCLE": ("STRUCT_CDG_INVALID", "error"),
    "C-ORPHAN": ("STRUCT_CDG_INVALID", "warning"),
    "C-FIELDS": ("STRUCT_CDG_INVALID", "warning"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _group_key(record: dict[str, Any]) -> str:
    return record["atom_key"].split(":", 1)[0]


def _companion_paths(record: dict[str, Any]) -> dict[str, Path]:
    wrapper_path = ROOT / record["module_path"]
    if wrapper_path.name == "atoms.py":
        return {
            "wrapper": wrapper_path,
            "witnesses": wrapper_path.parent / "witnesses.py",
            "cdg": wrapper_path.parent / "cdg.json",
        }
    stem = wrapper_path.stem
    candidates = {
        "wrapper": wrapper_path,
        "witnesses": wrapper_path.parent / f"{stem}_witnesses.py",
        "cdg": wrapper_path.parent / f"{stem}_cdg.json",
    }
    if not candidates["witnesses"].exists():
        candidates["witnesses"] = wrapper_path.parent / "witnesses.py"
    if not candidates["cdg"].exists():
        candidates["cdg"] = wrapper_path.parent / "cdg.json"
    return candidates


def _finding(
    record: dict[str, Any],
    code: str,
    severity: str,
    source: str,
    message: str,
    *,
    path: str | None = None,
    line: int | None = None,
    column: int | None = None,
    raw_code: str | None = None,
) -> dict[str, Any]:
    return {
        "atom_id": record["atom_id"],
        "atom_key": record["atom_key"],
        "domain_family": record["domain_family"],
        "module_path": record["module_path"],
        "wrapper_symbol": record["wrapper_symbol"],
        "code": code,
        "severity": severity,
        "source": source,
        "message": message,
        "path": path or record["module_path"],
        "line": line,
        "column": column,
        "raw_code": raw_code,
    }


def _row_specific_message(message: str, wrapper_symbol: str) -> bool:
    return message.startswith(f"{wrapper_symbol}:")


def _normalize_prompt_audit_violation(record: dict[str, Any], violation: Any, source_name: str) -> dict[str, Any]:
    code, severity = FILE_RULE_MAP.get(violation.rule, ("STRUCT_UNKNOWN", "warning"))
    message = violation.message
    if violation.rule == "A-REG" and "not the outermost" in message:
        code = "STRUCT_REGISTER_NOT_OUTERMOST"
    if violation.rule == "A-CONTRACT":
        if "@icontract.require" in message:
            code = "STRUCT_REQUIRE_MISSING"
        elif "@icontract.ensure" in message:
            code = "STRUCT_ENSURE_MISSING"
    if violation.rule == "A-DOC" and "Args:" in message:
        code = "STRUCT_DOCSTRING_PLACEHOLDER"
    return _finding(record, code, severity, source_name, message, raw_code=violation.rule)


def _collect_prompt_audit_findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["module_path"]].append(record)

    for module_path, module_records in grouped.items():
        sample = module_records[0]
        paths = _companion_paths(sample)
        audits = [("audit.py", prompt_audit.audit_atoms(paths["wrapper"]))]
        if paths["witnesses"].exists():
            audits.append(("audit.py", prompt_audit.audit_witnesses(paths["witnesses"])))
        elif sample.get("has_witnesses"):
            pass
        else:
            dummy = prompt_audit.FileAudit(str(paths["witnesses"]))
            dummy.fail("W-MISSING", "witnesses.py not found")
            audits.append(("audit.py", dummy))
        if paths["cdg"].exists() or sample.get("has_cdg"):
            audits.append(("audit.py", prompt_audit.audit_cdg(paths["cdg"])))
        else:
            dummy = prompt_audit.FileAudit(str(paths["cdg"]))
            dummy.fail("C-MISSING", "cdg.json not found")
            audits.append(("audit.py", dummy))

        for source_name, audit_result in audits:
            for violation in audit_result.violations:
                targeted = [record for record in module_records if _row_specific_message(violation.message, record["wrapper_symbol"])]
                recipients = targeted or module_records
                for record in recipients:
                    findings.append(_normalize_prompt_audit_violation(record, violation, source_name))
    return findings


def _collect_manifest_heuristic_findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for record in records:
        if record.get("placeholder_witness"):
            findings.append(_finding(record, "STRUCT_WITNESS_PLACEHOLDER", "warning", "manifest_heuristics", "register_atom uses a placeholder witness"))
        if "placeholder_docstring" in record.get("inventory_notes", []):
            findings.append(_finding(record, "STRUCT_DOCSTRING_PLACEHOLDER", "warning", "manifest_heuristics", "Docstring summary looks placeholder-like"))
        if record.get("skeleton"):
            findings.append(_finding(record, "STRUCT_STUB_PUBLIC_API", "error", "manifest_heuristics", "Public atom raises NotImplementedError"))
        if record.get("has_weak_types"):
            findings.append(_finding(record, "STRUCT_WEAK_TYPES", "warning", "manifest_heuristics", "Weak public type annotations detected"))
        if record.get("uses_varargs"):
            findings.append(_finding(record, "STRUCT_PUBLIC_VARARGS", "warning", "manifest_heuristics", "Public signature uses *args"))
        if record.get("uses_kwargs"):
            findings.append(_finding(record, "STRUCT_PUBLIC_KWARGS", "warning", "manifest_heuristics", "Public signature uses **kwargs"))
    return findings


def _collect_type_audit_findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)
    try:
        audits = [cdg_audit.audit_types_per_cdg(cdg) for cdg in cdg_audit.load_all_cdgs()]
    except Exception:
        return findings
    for audit_result in audits:
        recipients = grouped.get(audit_result.atom_name)
        if not recipients:
            continue
        if audit_result.health_score < 0.85:
            for record in recipients:
                findings.append(
                    _finding(
                        record,
                        "STRUCT_CDG_TYPE_HEALTH_LOW",
                        "warning",
                        "type_and_isomorphism_audit.py",
                        f"CDG type health score is {audit_result.health_score:.3f}",
                    )
                )
        if audit_result.non_normalizable:
            for record in recipients:
                findings.append(
                    _finding(
                        record,
                        "STRUCT_CDG_TYPE_NON_NORMALIZED",
                        "warning",
                        "type_and_isomorphism_audit.py",
                        f"CDG has {len(audit_result.non_normalizable)} non-normalizable type descriptors",
                    )
                )
    return findings


def _collect_verify_findings(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not VERIFY_SCRIPT.exists():
        return findings, {"available": False, "ok": False, "reason": "missing_script"}
    try:
        result = subprocess.run(
            ["python", str(VERIFY_SCRIPT), ".", "--package", "ageoa", "--json"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    except Exception as exc:
        return findings, {"available": True, "ok": False, "reason": "execution_error", "message": str(exc)}
    if not result.stdout.strip():
        return findings, {"available": True, "ok": False, "reason": "empty_output", "exit_code": result.returncode}
    payload = json.loads(result.stdout)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_path[str((ROOT / record["module_path"]).resolve())].append(record)
    unmapped_issue_count = 0
    for issue in payload.get("issues", []):
        recipients = by_path.get(issue["path"], [])
        if not recipients:
            unmapped_issue_count += 1
            continue
        raw_code = issue.get("code", "verify_atoms_repo")
        for record in recipients:
            if issue.get("symbol", "").startswith("witness_"):
                code = "STRUCT_WITNESS_MISSING"
            elif raw_code == "undefined-name":
                code = "STRUCT_EXPORT_OR_REGISTRY_ISSUE"
            else:
                code = "STRUCT_EXPORT_OR_REGISTRY_ISSUE"
            findings.append(
                _finding(
                    record,
                    code,
                    "warning",
                    "verify_atoms_repo.py",
                    issue.get("message", raw_code),
                    path=str(Path(issue["path"]).relative_to(ROOT)),
                    line=issue.get("line"),
                    column=issue.get("column"),
                    raw_code=raw_code,
                )
            )
    status = {
        "available": True,
        "ok": payload.get("ok", False),
        "exit_code": result.returncode,
        "issue_count": payload.get("issue_count", 0),
        "mapped_issue_count": len(findings),
        "unmapped_issue_count": unmapped_issue_count,
    }
    return findings, status


def _structural_status(findings: list[dict[str, Any]]) -> str:
    if any(finding["severity"] == "error" for finding in findings):
        return "fail"
    if findings:
        return "partial"
    return "pass"


def _action_for_code(code: str) -> str | None:
    mapping = {
        "STRUCT_PARSE_FAIL": "fix wrapper parse errors",
        "STRUCT_REGISTER_MISSING": "add register_atom to the public wrapper",
        "STRUCT_REGISTER_NOT_OUTERMOST": "move register_atom to the outermost decorator position",
        "STRUCT_REQUIRE_MISSING": "add a meaningful icontract.require decorator",
        "STRUCT_ENSURE_MISSING": "add a meaningful icontract.ensure decorator",
        "STRUCT_WITNESS_FILE_MISSING": "add or repair the companion witness module",
        "STRUCT_WITNESS_PLACEHOLDER": "replace placeholder witness bindings with typed witnesses",
        "STRUCT_CDG_MISSING": "add the missing CDG artifact",
        "STRUCT_CDG_INVALID": "repair the CDG structure and schema",
        "STRUCT_DOCSTRING_MISSING": "add a real docstring",
        "STRUCT_DOCSTRING_PLACEHOLDER": "replace placeholder docstring text",
        "STRUCT_STUB_PUBLIC_API": "implement or clearly downgrade the stubbed public atom",
        "STRUCT_EXPORT_OR_REGISTRY_ISSUE": "repair missing imports or export/registry wiring",
    }
    return mapping.get(code)


def build_structural_report(manifest: dict[str, Any], *, include_verify: bool = True) -> dict[str, Any]:
    """Build a normalized structural report from existing audit sources."""
    records = manifest.get("atoms", [])
    findings = []
    source_status: dict[str, Any] = {
        "audit.py": {"available": True, "ok": True},
        "type_and_isomorphism_audit.py": {"available": True, "ok": True},
    }
    findings.extend(_collect_prompt_audit_findings(records))
    findings.extend(_collect_manifest_heuristic_findings(records))
    findings.extend(_collect_type_audit_findings(records))
    if include_verify:
        verify_findings, verify_status = _collect_verify_findings(records)
        findings.extend(verify_findings)
        source_status["verify_atoms_repo.py"] = verify_status
    else:
        source_status["verify_atoms_repo.py"] = {"available": False, "ok": False, "reason": "skipped"}

    code_counts = Counter(finding["code"] for finding in findings)
    severity_counts = Counter(finding["severity"] for finding in findings)
    family_counts = Counter(finding["domain_family"] for finding in findings)
    source_counts = Counter(finding["source"] for finding in findings)
    atoms_with_findings = {finding["atom_id"] for finding in findings}
    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "summary": {
            "atom_count": len(records),
            "finding_count": len(findings),
            "atoms_with_findings": len(atoms_with_findings),
            "by_code": dict(sorted(code_counts.items())),
            "by_severity": dict(sorted(severity_counts.items())),
            "by_family": dict(sorted(family_counts.items())),
            "by_source": dict(sorted(source_counts.items())),
        },
        "source_status": source_status,
        "findings": sorted(findings, key=lambda item: (item["atom_id"], item["severity"], item["code"], item["message"])),
    }


def integrate_structural_results(manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Apply structural findings back onto manifest rows."""
    findings_by_atom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in report.get("findings", []):
        findings_by_atom[finding["atom_id"]].append(finding)

    for record in manifest.get("atoms", []):
        findings = findings_by_atom.get(record["atom_id"], [])
        codes = sorted({finding["code"] for finding in findings})
        actions = []
        for code in codes:
            action = _action_for_code(code)
            if action and action not in actions:
                actions.append(action)
        record["structural_findings"] = codes
        record["structural_finding_details"] = findings
        record["structural_status"] = _structural_status(findings)
        for code in codes:
            if code not in record.get("blocking_findings", []):
                record.setdefault("blocking_findings", []).append(code)
        for action in actions:
            if action not in record.get("required_actions", []):
                record.setdefault("required_actions", []).append(action)
        record.setdefault("status_basis", {}).setdefault("structural", [])
        record["status_basis"]["structural"] = sorted({finding["source"] for finding in findings}) or ["no_structural_findings"]
    manifest["summary"]["structural_status_counts"] = dict(
        sorted(Counter(record["structural_status"] for record in manifest.get("atoms", [])).items())
    )
    return manifest


def write_structural_report(report: dict[str, Any]) -> None:
    ensure_dir(AUDIT_DIR)
    write_json(AUDIT_STRUCTURAL_REPORT_PATH, report)
    with AUDIT_STRUCTURAL_FINDINGS_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["atom_id", "atom_key", "code", "severity", "source", "path", "line", "column", "message"],
        )
        writer.writeheader()
        for finding in report.get("findings", []):
            writer.writerow(
                {
                    "atom_id": finding["atom_id"],
                    "atom_key": finding["atom_key"],
                    "code": finding["code"],
                    "severity": finding["severity"],
                    "source": finding["source"],
                    "path": finding["path"],
                    "line": finding.get("line"),
                    "column": finding.get("column"),
                    "message": finding["message"],
                }
            )
