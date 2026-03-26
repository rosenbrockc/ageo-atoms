"""Semantic evidence persistence and portfolio rollup helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .io import read_json, safe_atom_stem, write_json
from .paths import AUDIT_EVIDENCE_DIR, AUDIT_MANIFEST_PATH, AUDIT_SEMANTIC_REPORT_PATH

SEMANTIC_ACTIONS = {
    "RUNTIME_IMPORT_FAIL": "repair the wrapper import path or document why runtime probing is not applicable",
    "RUNTIME_PROBE_FAIL": "repair the wrapper so a safe positive runtime probe succeeds",
    "RUNTIME_CONTRACT_NEGATIVE_FAIL": "repair input validation so obviously invalid inputs are rejected",
    "RUNTIME_PROBE_SKIPPED": "add a safe runtime probe or narrow the wrapper scope documentation",
    "RUNTIME_NOT_IMPLEMENTED": "implement the wrapper or clearly downgrade the public atom",
    "RETURN_FABRICATED_ATTRIBUTE": "return values directly anchored to upstream outputs or documented wrapper state",
    "RETURN_IGNORES_UPSTREAM_VALUE": "thread the upstream return value into the wrapper output or document the derivation",
    "RETURN_DERIVED_ARTIFACT_UNDOCUMENTED": "document or simplify derived return artifacts",
    "STATE_FABRICATED_FIELD": "align state updates with the declared state model fields",
    "STATE_REHYDRATION_MISSING": "rehydrate the required state fields or downgrade the wrapper semantics",
    "STATE_QUERY_MUTATION_CONFUSION": "separate query and mutation semantics or rename the wrapper",
    "NOUN_UNDOCUMENTED_OUTPUT": "document generated output nouns or align them with upstream terminology",
    "NOUN_UNDOCUMENTED_STATE": "document generated state nouns or align them with declared state fields",
    "NOUN_LOW_UPSTREAM_ALIGNMENT": "review the wrapper terminology against the upstream/source terms",
}

SEMANTIC_FINDING_PREFIXES = (
    "RUNTIME_",
    "RETURN_",
    "STATE_",
    "NOUN_",
    "SEMANTIC_",
)

SEMANTIC_BLOCKING_CODES = {
    "RUNTIME_PROBE_FAIL",
    "RUNTIME_CONTRACT_NEGATIVE_FAIL",
    "RUNTIME_NOT_IMPLEMENTED",
    "RETURN_FABRICATED_ATTRIBUTE",
    "RETURN_IGNORES_UPSTREAM_VALUE",
    "STATE_FABRICATED_FIELD",
    "STATE_QUERY_MUTATION_CONFUSION",
    "SEMANTIC_FIDELITY_FAIL",
    "SEMANTIC_REVIEW_REQUIRED",
}


def utc_now() -> str:
    """Return a stable UTC timestamp for generated artifacts."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def evidence_path_for(atom_id: str):
    """Return the canonical evidence path for one atom id."""
    return AUDIT_EVIDENCE_DIR / f"{safe_atom_stem(atom_id)}.json"


def load_atom_evidence(atom_id: str) -> dict[str, Any]:
    """Load combined evidence for one atom."""
    path = evidence_path_for(atom_id)
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_evidence_section(atom_id: str, category: str, section: dict[str, Any]) -> dict[str, Any]:
    """Merge a semantic evidence section into the canonical per-atom evidence file."""
    evidence = load_atom_evidence(atom_id)
    if not evidence.get("atom_id"):
        evidence["atom_id"] = atom_id
    evidence.setdefault("schema_version", "1.0")
    evidence[category] = section
    write_json(evidence_path_for(atom_id), evidence)
    return evidence


def load_semantic_sections(atom_id: str) -> dict[str, dict[str, Any]]:
    """Load only the phase-5 semantic evidence categories for one atom."""
    evidence = load_atom_evidence(atom_id)
    sections: dict[str, dict[str, Any]] = {}
    for category in ("runtime_probe", "return_fidelity", "state_fidelity", "generated_nouns"):
        section = evidence.get(category)
        if isinstance(section, dict):
            sections[category] = section
    return sections


def _detail(category: str, section: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": category,
        "status": section.get("status", "unknown"),
        "findings": list(section.get("findings", [])),
        "notes": list(section.get("notes", [])),
        "source_refs": list(section.get("source_refs", [])),
    }


def _runtime_status(sections: dict[str, dict[str, Any]]) -> str:
    runtime = sections.get("runtime_probe")
    if not runtime:
        return "unknown"
    return str(runtime.get("status", "unknown"))


def _developer_semantics_status(record: dict[str, Any], sections: dict[str, dict[str, Any]]) -> str:
    nouns = sections.get("generated_nouns")
    if nouns:
        noun_status = nouns.get("status", "unknown")
        if noun_status in {"fail", "partial", "pass"}:
            return str(noun_status)
    if record.get("has_docstring"):
        return "unknown"
    return "unknown"


def _semantic_status(sections: dict[str, dict[str, Any]]) -> str:
    if not sections:
        return "unknown"
    runtime = sections.get("runtime_probe", {})
    runtime_status = runtime.get("status", "unknown")
    findings = {
        finding
        for section in sections.values()
        for finding in section.get("findings", [])
        if isinstance(finding, str)
    }
    if runtime_status == "fail" or findings & {
        "RETURN_FABRICATED_ATTRIBUTE",
        "RETURN_IGNORES_UPSTREAM_VALUE",
        "STATE_FABRICATED_FIELD",
        "STATE_QUERY_MUTATION_CONFUSION",
    }:
        return "fail"
    if runtime_status == "pass" and all(
        section.get("status", "unknown") in {"pass", "not_applicable"}
        for section in sections.values()
        if section is not runtime
    ):
        return "pass"
    if any(section.get("status", "unknown") in {"partial", "fail", "unknown"} for section in sections.values()):
        return "partial"
    return "unknown"


def build_semantic_report(manifest: dict[str, Any], structural_report: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Roll semantic evidence into a portfolio report and updated manifest."""
    semantic_atoms: list[dict[str, Any]] = []
    runtime_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    developer_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    structural_report = structural_report or {}
    source_artifacts = {
        "manifest": str(AUDIT_MANIFEST_PATH),
        "structural_report": structural_report.get("generated_at"),
        "evidence_dir": str(AUDIT_EVIDENCE_DIR),
    }

    action_values = set(SEMANTIC_ACTIONS.values())
    for record in manifest.get("atoms", []):
        sections = load_semantic_sections(record["atom_id"])
        runtime_status = _runtime_status(sections)
        semantic_status = _semantic_status(sections)
        developer_status = _developer_semantics_status(record, sections)
        details = [_detail(category, section) for category, section in sorted(sections.items())]
        findings = sorted(
            {
                finding
                for section in sections.values()
                for finding in section.get("findings", [])
                if isinstance(finding, str)
            }
        )
        if runtime_status == "pass":
            findings.append("SEMANTIC_RUNTIME_SUPPORT_PRESENT")
        else:
            findings.append("SEMANTIC_RUNTIME_SUPPORT_MISSING")
        if any(code in findings for code in ("RETURN_FABRICATED_ATTRIBUTE", "RETURN_IGNORES_UPSTREAM_VALUE", "STATE_FABRICATED_FIELD", "STATE_QUERY_MUTATION_CONFUSION")):
            findings.append("SEMANTIC_FIDELITY_FAIL")
        if semantic_status in {"fail", "partial"}:
            findings.append("SEMANTIC_REVIEW_REQUIRED")
        findings = sorted(set(findings))

        prior_blocking = [
            code
            for code in record.get("blocking_findings", [])
            if not any(code.startswith(prefix) for prefix in SEMANTIC_FINDING_PREFIXES)
        ]
        blocking = sorted(set(prior_blocking + [code for code in findings if code in SEMANTIC_BLOCKING_CODES]))

        prior_actions = [action for action in record.get("required_actions", []) if action not in action_values]
        semantic_actions = [SEMANTIC_ACTIONS[code] for code in findings if code in SEMANTIC_ACTIONS]
        required_actions = sorted(dict.fromkeys(prior_actions + semantic_actions))

        record["runtime_status"] = runtime_status
        record["semantic_status"] = semantic_status
        record["developer_semantics_status"] = developer_status
        record["blocking_findings"] = blocking
        record["required_actions"] = required_actions
        record["semantic_findings"] = findings
        record["semantic_finding_details"] = details
        record.setdefault("status_basis", {})["runtime"] = ["runtime_probe"] if "runtime_probe" in sections else ["no_runtime_probe_evidence"]
        record["status_basis"]["semantic"] = sorted(sections) or ["no_semantic_evidence"]
        record["status_basis"]["developer_semantics"] = ["generated_nouns"] if "generated_nouns" in sections else ["no_generated_noun_evidence"]

        runtime_counts[runtime_status] += 1
        semantic_counts[semantic_status] += 1
        developer_counts[developer_status] += 1
        code_counts.update(findings)
        category_counts.update(sections.keys())
        semantic_atoms.append(
            {
                "atom_id": record["atom_id"],
                "atom_name": record["atom_name"],
                "runtime_status": runtime_status,
                "semantic_status": semantic_status,
                "developer_semantics_status": developer_status,
                "findings": findings,
                "required_actions": required_actions,
                "evidence_categories": sorted(sections),
            }
        )

    manifest.setdefault("summary", {})
    manifest["summary"]["runtime_status_counts"] = dict(sorted(runtime_counts.items()))
    manifest["summary"]["semantic_status_counts"] = dict(sorted(semantic_counts.items()))
    manifest["summary"]["developer_semantics_status_counts"] = dict(sorted(developer_counts.items()))

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "summary": {
            "atom_count": len(manifest.get("atoms", [])),
            "runtime_status_counts": dict(sorted(runtime_counts.items())),
            "semantic_status_counts": dict(sorted(semantic_counts.items())),
            "developer_semantics_status_counts": dict(sorted(developer_counts.items())),
            "atoms_with_runtime_support": runtime_counts.get("pass", 0),
            "atoms_with_semantic_findings": sum(1 for atom in semantic_atoms if atom["findings"]),
            "by_finding_code": dict(sorted(code_counts.items())),
            "by_category": dict(sorted(category_counts.items())),
        },
        "source_artifacts": source_artifacts,
        "atoms": semantic_atoms,
    }
    return manifest, report


def write_semantic_report(report: dict[str, Any]) -> None:
    """Persist the semantic portfolio report."""
    write_json(AUDIT_SEMANTIC_REPORT_PATH, report)
