"""Deterministic acceptability scoring."""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from typing import Any

from .io import safe_atom_stem, write_json
from .paths import AUDIT_RESULTS_DIR, AUDIT_SCORES_PATH

PLACEHOLDER_DOC_PATTERNS = (
    "auto-generated",
    "derived deterministically from inputs",
    "skeleton for future ingestion",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.split(r"[^A-Za-z0-9]+", expanded.lower())
    return {token for token in tokens if token}


def _score_structural(record: dict[str, Any]) -> tuple[int, list[str], str]:
    score = 20
    findings = list(record.get("structural_findings", []))
    status = record.get("structural_status", "unknown")
    if not findings and status == "pass":
        return score, [], "pass"
    penalties = {
        "STRUCT_STUB_PUBLIC_API": 20,
        "STRUCT_REGISTER_MISSING": 10,
        "STRUCT_REQUIRE_MISSING": 5,
        "STRUCT_ENSURE_MISSING": 5,
        "STRUCT_WITNESS_PLACEHOLDER": 4,
        "STRUCT_WITNESS_FILE_MISSING": 3,
        "STRUCT_CDG_MISSING": 3,
        "STRUCT_DOCSTRING_MISSING": 4,
        "STRUCT_DOCSTRING_PLACEHOLDER": 4,
        "STRUCT_WEAK_TYPES": 3,
    }
    for code in findings:
        score -= penalties.get(code, 2)
    score = max(0, score)
    if status not in {"pass", "partial", "fail"}:
        if score >= 16:
            status = "pass"
        elif score >= 8:
            status = "partial"
        else:
            status = "fail"
    return score, findings, status


def _score_runtime(record: dict[str, Any], evidence: dict[str, Any] | None) -> tuple[int, list[str], str]:
    runtime = evidence.get("runtime_probe") if evidence else None
    if isinstance(runtime, dict):
        findings = list(runtime.get("findings", []))
        status = runtime.get("status", "unknown")
        if status == "pass":
            return 14, findings, "pass"
        if status == "partial":
            return 9, findings or ["RUNTIME_PROBE_SKIPPED"], "partial"
        if status == "fail":
            return 0, findings or ["RUNTIME_PROBE_FAIL"], "fail"
        if status == "not_applicable":
            return 6, findings or ["RUNTIME_PROBE_SKIPPED"], "partial"
        return 6, findings or ["RUNTIME_NO_PROBE_EVIDENCE"], "unknown"
    if record.get("skeleton"):
        return 0, ["RUNTIME_NOT_IMPLEMENTED"], "fail"
    if record.get("has_parity_tests"):
        return 8, ["RUNTIME_NO_PROBE_EVIDENCE"], "unknown"
    return 6, ["RUNTIME_NO_PROBE_EVIDENCE"], "unknown"


def _score_fidelity(record: dict[str, Any], evidence: dict[str, Any] | None) -> tuple[int, list[str], str]:
    score = 35
    findings: list[str] = []
    evidence = evidence or {}
    if not evidence or not evidence.get("mapping_found"):
        score = 15
        findings.append("FIDELITY_UPSTREAM_UNMAPPED")
    if evidence.get("upstream_signature") is None:
        score = min(score, 14)
        findings.append("FIDELITY_UPSTREAM_SIGNATURE_UNAVAILABLE")
    for finding in evidence.get("findings", []) if evidence else []:
        if finding == "FIDELITY_SIGNATURE_MISSING_REQUIRED":
            score -= 14
            findings.append(finding)
        elif finding == "FIDELITY_SIGNATURE_INVENTED_PARAMETER":
            score -= 12
            findings.append(finding)
        elif finding == "FIDELITY_SIGNATURE_ORDER_MISMATCH":
            score -= 4
            findings.append(finding)
        elif finding == "FIDELITY_REQUIREDNESS_MISMATCH":
            score -= 4
            findings.append(finding)
        elif finding == "FIDELITY_PUBLIC_VARARGS":
            score -= 4
            findings.append(finding)
        elif finding == "FIDELITY_PUBLIC_KWARGS":
            score -= 4
            findings.append(finding)
        elif finding == "FIDELITY_WEAK_TYPES":
            score -= 5
            findings.append(finding)
    for category in ("return_fidelity", "state_fidelity", "generated_nouns"):
        section = evidence.get(category) if evidence else None
        if not isinstance(section, dict):
            continue
        for finding in section.get("findings", []):
            if finding in {"RETURN_FABRICATED_ATTRIBUTE", "RETURN_IGNORES_UPSTREAM_VALUE", "STATE_FABRICATED_FIELD", "STATE_QUERY_MUTATION_CONFUSION"}:
                score -= 16
            elif finding in {"RETURN_DERIVED_ARTIFACT_UNDOCUMENTED", "STATE_REHYDRATION_MISSING"}:
                score -= 7
            elif finding in {"NOUN_UNDOCUMENTED_OUTPUT", "NOUN_UNDOCUMENTED_STATE", "NOUN_LOW_UPSTREAM_ALIGNMENT"}:
                score -= 5
            findings.append(finding)
    score = max(0, score)
    if any(
        item in findings
        for item in (
            "FIDELITY_SIGNATURE_MISSING_REQUIRED",
            "FIDELITY_SIGNATURE_INVENTED_PARAMETER",
            "RETURN_FABRICATED_ATTRIBUTE",
            "RETURN_IGNORES_UPSTREAM_VALUE",
            "STATE_FABRICATED_FIELD",
            "STATE_QUERY_MUTATION_CONFUSION",
        )
    ):
        status = "fail"
    elif findings:
        status = "partial"
    else:
        status = "pass"
    return score, findings, status


def _score_developer_semantics(record: dict[str, Any], evidence: dict[str, Any] | None) -> tuple[int, list[str], str]:
    score = 15
    findings: list[str] = []
    summary = (record.get("docstring_summary") or "").strip().lower()
    if not summary:
        score -= 5
        findings.append("SEMANTICS_DOCSTRING_ABSENT")
    elif any(pattern in summary for pattern in PLACEHOLDER_DOC_PATTERNS):
        score -= 5
        findings.append("SEMANTICS_DOCSTRING_PLACEHOLDER")
    wrapper_tokens = _tokenize(record.get("wrapper_symbol"))
    upstream_tokens = set()
    if evidence and evidence.get("upstream_mapping"):
        mapping = evidence["upstream_mapping"]
        upstream_tokens |= _tokenize(mapping.get("function"))
        upstream_tokens |= _tokenize(mapping.get("module"))
    if wrapper_tokens and upstream_tokens and not (wrapper_tokens & upstream_tokens):
        score -= 3
        findings.append("SEMANTICS_LOW_NAME_OVERLAP")
    if record.get("source_kind") == "generated_ingest" and summary.startswith("stateless wrapper"):
        score -= 2
        findings.append("SEMANTICS_GENERATED_ABSTRACTION")
    nouns = evidence.get("generated_nouns") if evidence else None
    if isinstance(nouns, dict):
        for finding in nouns.get("findings", []):
            if finding in {"NOUN_UNDOCUMENTED_OUTPUT", "NOUN_UNDOCUMENTED_STATE"}:
                score -= 4
                findings.append(finding)
            elif finding == "NOUN_LOW_UPSTREAM_ALIGNMENT":
                score -= 5
                findings.append(finding)
    score = max(0, score)
    if any(code in findings for code in {"NOUN_UNDOCUMENTED_OUTPUT", "NOUN_UNDOCUMENTED_STATE", "NOUN_LOW_UPSTREAM_ALIGNMENT"}):
        status = "partial" if score >= 7 else "fail"
    elif score >= 12:
        status = "pass"
    elif score >= 7:
        status = "partial"
    else:
        status = "fail"
    return score, findings, status


def _score_trust_support(record: dict[str, Any]) -> tuple[int, list[str], str]:
    score = 0
    findings: list[str] = []
    if record.get("has_references"):
        score += 4
    else:
        findings.append("TRUST_REFERENCES_MISSING")
    if record.get("source_revision") or record.get("upstream_version"):
        score += 6
    else:
        findings.append("TRUST_PROVENANCE_MISSING")
    if score >= 8:
        status = "pass"
    elif score >= 4:
        status = "partial"
    else:
        status = "unknown"
    return score, findings, status


def _band_for_score(score: int) -> str:
    if score <= 19:
        return "broken_candidate"
    if score <= 49:
        return "misleading_candidate"
    if score <= 69:
        return "limited_acceptability"
    if score <= 84:
        return "acceptable_with_limits_candidate"
    return "review_ready"


def score_acceptability(record: dict[str, Any], signature_evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Score a single atom deterministically."""
    structural_score, structural_findings, structural_status = _score_structural(record)
    runtime_score, runtime_findings, runtime_status = _score_runtime(record, signature_evidence)
    fidelity_score, fidelity_findings, semantic_status = _score_fidelity(record, signature_evidence)
    semantics_score, semantics_findings, developer_status = _score_developer_semantics(record, signature_evidence)
    trust_score, trust_findings, trust_status = _score_trust_support(record)

    score = structural_score + runtime_score + fidelity_score + semantics_score + trust_score
    caps: list[int] = []
    blockers: list[str] = []

    if structural_status == "fail" or runtime_status == "fail":
        caps.append(19)
        blockers.extend(
            code
            for code in structural_findings + runtime_findings
            if code in {"STRUCT_STUB_PUBLIC_API", "RUNTIME_NOT_IMPLEMENTED", "RUNTIME_PROBE_FAIL", "RUNTIME_CONTRACT_NEGATIVE_FAIL"}
        )
    if "FIDELITY_UPSTREAM_UNMAPPED" in fidelity_findings:
        caps.append(59)
    if any(
        code in fidelity_findings
        for code in (
            "FIDELITY_SIGNATURE_MISSING_REQUIRED",
            "FIDELITY_SIGNATURE_INVENTED_PARAMETER",
            "RETURN_FABRICATED_ATTRIBUTE",
            "RETURN_IGNORES_UPSTREAM_VALUE",
            "STATE_FABRICATED_FIELD",
            "STATE_QUERY_MUTATION_CONFUSION",
        )
    ):
        caps.append(49)
        blockers.extend(
            code
            for code in fidelity_findings
            if code
            in {
                "FIDELITY_SIGNATURE_MISSING_REQUIRED",
                "FIDELITY_SIGNATURE_INVENTED_PARAMETER",
                "RETURN_FABRICATED_ATTRIBUTE",
                "RETURN_IGNORES_UPSTREAM_VALUE",
                "STATE_FABRICATED_FIELD",
                "STATE_QUERY_MUTATION_CONFUSION",
            }
        )
    if "TRUST_PROVENANCE_MISSING" in trust_findings:
        caps.append(69)
    if "RUNTIME_NO_PROBE_EVIDENCE" in runtime_findings:
        caps.append(84)

    if caps:
        score = min(score, min(caps))
    score = max(0, min(100, score))
    band = _band_for_score(score)
    if score <= 19:
        verdict = "broken"
    elif score <= 49:
        verdict = "misleading"
    else:
        verdict = "acceptable_with_limits"

    required_actions: list[str] = []
    action_map = {
        "STRUCT_WITNESS_PLACEHOLDER": "replace placeholder witness with a typed witness binding",
        "STRUCT_WITNESS_FILE_MISSING": "add or reconcile the companion witnesses module",
        "STRUCT_CDG_MISSING": "add a CDG artifact for this atom family",
        "RUNTIME_NO_PROBE_EVIDENCE": "add parity fixtures or a safe runtime probe",
        "RUNTIME_IMPORT_FAIL": "repair the wrapper import path or document why runtime probing is not applicable",
        "RUNTIME_PROBE_FAIL": "repair the wrapper so a safe positive runtime probe succeeds",
        "RUNTIME_CONTRACT_NEGATIVE_FAIL": "repair input validation so invalid inputs are rejected",
        "FIDELITY_UPSTREAM_UNMAPPED": "map this atom in scripts/atom_manifest.yml",
        "FIDELITY_UPSTREAM_SIGNATURE_UNAVAILABLE": "add vendored or importable upstream signature evidence",
        "FIDELITY_SIGNATURE_MISSING_REQUIRED": "align wrapper parameters with the upstream required parameters",
        "FIDELITY_SIGNATURE_INVENTED_PARAMETER": "remove or document invented wrapper parameters",
        "RETURN_FABRICATED_ATTRIBUTE": "return values directly anchored to upstream outputs or documented wrapper state",
        "RETURN_IGNORES_UPSTREAM_VALUE": "thread the upstream return value into the wrapper output or document the derivation",
        "RETURN_DERIVED_ARTIFACT_UNDOCUMENTED": "document or simplify derived return artifacts",
        "STATE_FABRICATED_FIELD": "align state updates with the declared state model fields",
        "STATE_REHYDRATION_MISSING": "rehydrate the required state fields or downgrade the wrapper semantics",
        "STATE_QUERY_MUTATION_CONFUSION": "separate query and mutation semantics or rename the wrapper",
        "NOUN_UNDOCUMENTED_OUTPUT": "document generated output nouns or align them with upstream terminology",
        "NOUN_UNDOCUMENTED_STATE": "document generated state nouns or align them with declared state fields",
        "NOUN_LOW_UPSTREAM_ALIGNMENT": "review the wrapper terminology against the upstream/source terms",
        "TRUST_PROVENANCE_MISSING": "pin source revision or upstream version in the manifest",
    }
    for code in structural_findings + runtime_findings + fidelity_findings + semantics_findings + trust_findings:
        action = action_map.get(code)
        if action and action not in required_actions:
            required_actions.append(action)

    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "acceptability_score": score,
        "acceptability_band": band,
        "max_reviewable_verdict": verdict,
        "overall_verdict": verdict,
        "dimension_scores": {
            "structural": structural_score,
            "runtime": runtime_score,
            "upstream_fidelity": fidelity_score,
            "developer_semantics": semantics_score,
            "trust_support": trust_score,
        },
        "dimension_evidence": {
            "structural_status": structural_status,
            "runtime_status": runtime_status,
            "semantic_status": semantic_status,
            "developer_semantics_status": developer_status,
            "references_status": "pass" if record.get("has_references") else "unknown",
            "parity_test_status": "pass" if record.get("has_parity_tests") else "unknown",
            "trust_support_status": trust_status,
        },
        "hard_blockers": sorted(set(blockers)),
        "major_penalties": sorted(set(structural_findings + runtime_findings + fidelity_findings + semantics_findings + trust_findings)),
        "required_actions": required_actions,
    }


def write_acceptability_result(result: dict[str, Any]) -> None:
    """Write one per-atom acceptability result."""
    write_json(AUDIT_RESULTS_DIR / f"{safe_atom_stem(result['atom_id'])}.json", result)


def write_scores_csv(rows: list[dict[str, Any]]) -> None:
    """Write a portfolio-level score CSV."""
    fieldnames = [
        "atom_id",
        "atom_name",
        "acceptability_score",
        "acceptability_band",
        "max_reviewable_verdict",
        "structural",
        "runtime",
        "upstream_fidelity",
        "developer_semantics",
        "trust_support",
        "hard_blockers",
        "major_penalties",
    ]
    with AUDIT_SCORES_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "atom_id": row["atom_id"],
                    "atom_name": row["atom_name"],
                    "acceptability_score": row["acceptability_score"],
                    "acceptability_band": row["acceptability_band"],
                    "max_reviewable_verdict": row["max_reviewable_verdict"],
                    "structural": row["dimension_scores"]["structural"],
                    "runtime": row["dimension_scores"]["runtime"],
                    "upstream_fidelity": row["dimension_scores"]["upstream_fidelity"],
                    "developer_semantics": row["dimension_scores"]["developer_semantics"],
                    "trust_support": row["dimension_scores"]["trust_support"],
                    "hard_blockers": ";".join(row["hard_blockers"]),
                    "major_penalties": ";".join(row["major_penalties"]),
                }
            )
