"""Deterministic Phase 3 risk triage."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, write_json
from .paths import (
    AUDIT_EVIDENCE_DIR,
    AUDIT_MANIFEST_PATH,
    AUDIT_RESULTS_DIR,
    AUDIT_REVIEW_QUEUE_CSV_PATH,
    AUDIT_RISK_REPORT_PATH,
    AUDIT_SCORES_PATH,
    AUDIT_STRUCTURAL_REPORT_PATH,
)

HIGH_RISK_THRESHOLD = 60
MEDIUM_RISK_THRESHOLD = 30

REVIEW_PRIORITY_ORDER = {
    "review_now": 0,
    "review_soon": 1,
    "review_later": 2,
}

PLACEHOLDER_DOC_PATTERNS = (
    "derived deterministically from inputs",
    "skeleton for future ingestion",
    "stateless wrapper",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.split(r"[^A-Za-z0-9]+", expanded.lower())
    return {token for token in tokens if token}


def _load_acceptability_rows() -> dict[str, dict[str, str]]:
    if not AUDIT_SCORES_PATH.exists():
        return {}
    with AUDIT_SCORES_PATH.open() as handle:
        reader = csv.DictReader(handle)
        return {row["atom_id"]: row for row in reader}


def _load_signature_evidence() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if not AUDIT_EVIDENCE_DIR.exists():
        return evidence
    for path in AUDIT_EVIDENCE_DIR.glob("*.json"):
        payload = read_json(path)
        atom_id = payload.get("atom_id")
        if atom_id:
            evidence[atom_id] = payload
    return evidence


def _load_structural_findings() -> dict[str, list[dict[str, Any]]]:
    if not AUDIT_STRUCTURAL_REPORT_PATH.exists():
        return {}
    payload = read_json(AUDIT_STRUCTURAL_REPORT_PATH)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in payload.get("findings", []):
        grouped[finding["atom_id"]].append(finding)
    return dict(grouped)


def _reason(points: int, *reasons: str) -> tuple[int, list[str]]:
    return points, [reason for reason in reasons if reason]


def _score_structural(record: dict[str, Any], findings: list[dict[str, Any]]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    codes = {finding["code"] for finding in findings}
    status = record.get("structural_status")
    if status == "fail":
        score = max(score, 25)
        reasons.append("RISK_STRUCTURAL_FAIL")
    elif status == "partial":
        score = max(score, 12)
        reasons.append("RISK_STRUCTURAL_PARTIAL")
    if "STRUCT_STUB_PUBLIC_API" in codes:
        score = max(score, 25)
        reasons.append("RISK_STUB_PUBLIC_API")
    if {"STRUCT_CDG_INVALID", "STRUCT_CDG_TYPE_HEALTH_LOW"} & codes:
        score = min(25, score + 5)
        reasons.append("RISK_CDG_ISSUES")
    return score, sorted(set(reasons))


def _score_fidelity(record: dict[str, Any], signature: dict[str, Any] | None) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if not signature or not signature.get("mapping_found"):
        score += 15
        reasons.append("RISK_UPSTREAM_UNMAPPED")
    elif signature.get("upstream_signature") is None:
        score += 8
        reasons.append("RISK_WEAK_UPSTREAM_ANCHOR")

    findings = set((signature or {}).get("findings", []))
    if {
        "FIDELITY_SIGNATURE_MISSING_REQUIRED",
        "FIDELITY_SIGNATURE_INVENTED_PARAMETER",
        "FIDELITY_SIGNATURE_ORDER_MISMATCH",
        "FIDELITY_REQUIREDNESS_MISMATCH",
    } & findings:
        score += 12
        reasons.append("RISK_SIGNATURE_MISMATCH")
    if record.get("has_weak_types") or "FIDELITY_WEAK_TYPES" in findings:
        score += 6
        reasons.append("RISK_WEAK_TYPES")
    return min(25, score), sorted(set(reasons))


def _score_evidence_gap(record: dict[str, Any], acceptability: dict[str, str] | None) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if not record.get("has_parity_tests"):
        score += 8
        reasons.append("RISK_MISSING_PARITY")
    if not (record.get("source_revision") or record.get("upstream_version")):
        score += 6
        reasons.append("RISK_MISSING_PROVENANCE")
    if not record.get("review_basis_at"):
        score += 3
        reasons.append("RISK_MISSING_REVIEW_BASIS")
    runtime_status = record.get("runtime_status")
    if runtime_status in {"unknown", "partial"}:
        score += 3
        reasons.append("RISK_NO_RUNTIME_EVIDENCE")
    if acceptability:
        penalties = set(filter(None, (acceptability.get("major_penalties") or "").split(";")))
        if "RUNTIME_NO_PROBE_EVIDENCE" in penalties and "RISK_NO_RUNTIME_EVIDENCE" not in reasons:
            score += 3
            reasons.append("RISK_NO_RUNTIME_EVIDENCE")
    return min(20, score), sorted(set(reasons))


def _score_statefulness(record: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if record.get("stateful"):
        score += 4
        reasons.append("RISK_STATEFUL_API")
    kind = record.get("stateful_kind", "none")
    if kind == "explicit_state_model":
        score = max(score, 8)
        reasons.append("RISK_STATE_MODEL_PRESENT")
    elif kind in {"argument_state", "return_state"}:
        score = max(score, 5)
        reasons.append("RISK_STATEFUL_API")
    if record.get("procedural"):
        score += 4
        reasons.append("RISK_PROCEDURAL_WRAPPER")
    return min(10, score), sorted(set(reasons))


def _score_origin_platform(record: dict[str, Any]) -> tuple[tuple[int, list[str]], tuple[int, list[str]]]:
    generation_score = 0
    generation_reasons: list[str] = []
    ffi_score = 0
    ffi_reasons: list[str] = []

    source_kind = record.get("source_kind")
    if source_kind == "generated_ingest":
        generation_score += 10
        generation_reasons.append("RISK_GENERATED_INGEST")
    elif source_kind == "refined_ingest":
        generation_score += 5
        generation_reasons.append("RISK_REFINED_INGEST")
    if record.get("domain_family") == "sklearn":
        generation_score += 6
        generation_reasons.append("RISK_SKLEARN_GENERATED_FAMILY")
    if record.get("stochastic"):
        generation_score += 4
        generation_reasons.append("RISK_STOCHASTIC")
    if record.get("ffi"):
        ffi_score += 5
        ffi_reasons.append("RISK_FFI_BACKED")
    return (
        min(10, generation_score),
        sorted(set(generation_reasons)),
    ), (
        min(5, ffi_score),
        sorted(set(ffi_reasons)),
    )


def _score_semantics_proxy(
    record: dict[str, Any],
    signature: dict[str, Any] | None,
    acceptability: dict[str, str] | None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    summary = (record.get("docstring_summary") or "").lower()
    if any(pattern in summary for pattern in PLACEHOLDER_DOC_PATTERNS):
        score += 3
        reasons.append("RISK_PLACEHOLDER_DOCSTRING")
    penalties = set(filter(None, (acceptability or {}).get("major_penalties", "").split(";")))
    if "SEMANTICS_GENERATED_ABSTRACTION" in penalties:
        score += 2
        reasons.append("RISK_GENERATED_ABSTRACTION_LANGUAGE")
    if "SEMANTICS_LOW_NAME_OVERLAP" in penalties:
        score += 2
        reasons.append("RISK_LOW_NAME_ALIGNMENT")
    elif signature and signature.get("upstream_mapping"):
        wrapper_tokens = _tokenize(record.get("wrapper_symbol"))
        mapping = signature["upstream_mapping"]
        upstream_tokens = _tokenize(mapping.get("function")) | _tokenize(mapping.get("module"))
        if wrapper_tokens and upstream_tokens and not (wrapper_tokens & upstream_tokens):
            score += 2
            reasons.append("RISK_LOW_NAME_ALIGNMENT")
    return min(5, score), sorted(set(reasons))


def compute_atom_risk(
    record: dict[str, Any],
    structural_findings: list[dict[str, Any]],
    signature: dict[str, Any] | None,
    acceptability: dict[str, str] | None,
) -> dict[str, Any]:
    """Compute deterministic Phase 3 risk for one atom."""
    structural_score, structural_reasons = _score_structural(record, structural_findings)
    fidelity_score, fidelity_reasons = _score_fidelity(record, signature)
    evidence_gap_score, evidence_gap_reasons = _score_evidence_gap(record, acceptability)
    statefulness_score, statefulness_reasons = _score_statefulness(record)
    (generation_score, generation_reasons), (ffi_score, ffi_reasons) = _score_origin_platform(record)
    semantics_score, semantics_reasons = _score_semantics_proxy(record, signature, acceptability)

    risk_score = min(
        100,
        structural_score
        + fidelity_score
        + evidence_gap_score
        + statefulness_score
        + generation_score
        + ffi_score
        + semantics_score,
    )
    risk_reasons = sorted(
        set(
            structural_reasons
            + fidelity_reasons
            + evidence_gap_reasons
            + statefulness_reasons
            + generation_reasons
            + ffi_reasons
            + semantics_reasons
        )
    )
    if (
        "RISK_STRUCTURAL_FAIL" in risk_reasons
        or "RISK_STUB_PUBLIC_API" in risk_reasons
        or risk_score >= HIGH_RISK_THRESHOLD
    ):
        risk_tier = "high"
    elif risk_score >= MEDIUM_RISK_THRESHOLD:
        risk_tier = "medium"
    else:
        risk_tier = "low"
    if (
        risk_tier == "high"
        or "RISK_STUB_PUBLIC_API" in risk_reasons
        or (
            "RISK_SIGNATURE_MISMATCH" in risk_reasons
            and "RISK_STATEFUL_API" in risk_reasons
        )
    ):
        review_priority = "review_now"
    elif risk_tier == "medium" and (
        {"RISK_SIGNATURE_MISMATCH", "RISK_STATEFUL_API", "RISK_STATE_MODEL_PRESENT"} & set(risk_reasons)
    ):
        review_priority = "review_soon"
    else:
        review_priority = "review_later"

    return {
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "domain_family": record["domain_family"],
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "risk_reasons": risk_reasons,
        "risk_dimensions": {
            "structural_risk": {"score": structural_score, "reasons": structural_reasons},
            "fidelity_risk": {"score": fidelity_score, "reasons": fidelity_reasons},
            "evidence_gap_risk": {"score": evidence_gap_score, "reasons": evidence_gap_reasons},
            "statefulness_risk": {"score": statefulness_score, "reasons": statefulness_reasons},
            "generation_risk": {"score": generation_score, "reasons": generation_reasons},
            "ffi_risk": {"score": ffi_score, "reasons": ffi_reasons},
            "semantics_proxy_risk": {"score": semantics_score, "reasons": semantics_reasons},
        },
        "review_priority": review_priority,
    }


def _write_review_queue(rows: list[dict[str, Any]], manifest_map: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "atom_id",
        "atom_name",
        "domain_family",
        "risk_score",
        "risk_tier",
        "review_priority",
        "risk_reasons",
        "structural_status",
        "semantic_status",
        "max_reviewable_verdict",
    ]
    ensure_dir(AUDIT_REVIEW_QUEUE_CSV_PATH.parent)
    with AUDIT_REVIEW_QUEUE_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            manifest_row = manifest_map[row["atom_id"]]
            writer.writerow(
                {
                    "atom_id": row["atom_id"],
                    "atom_name": row["atom_name"],
                    "domain_family": row["domain_family"],
                    "risk_score": row["risk_score"],
                    "risk_tier": row["risk_tier"],
                    "review_priority": row["review_priority"],
                    "risk_reasons": ";".join(row["risk_reasons"]),
                    "structural_status": manifest_row.get("structural_status", "unknown"),
                    "semantic_status": manifest_row.get("semantic_status", "unknown"),
                    "max_reviewable_verdict": manifest_row.get("max_reviewable_verdict", "unknown"),
                }
            )


def _sorted_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            REVIEW_PRIORITY_ORDER[row["review_priority"]],
            -row["risk_score"],
            row["atom_id"],
        ),
    )


def run_risk_triage(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update a manifest with Phase 3 risk fields and build the risk report."""
    signature_map = _load_signature_evidence()
    structural_map = _load_structural_findings()
    acceptability_map = _load_acceptability_rows()

    updated_atoms: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    by_reason: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()

    for record in manifest.get("atoms", []):
        atom_id = record["atom_id"]
        result = compute_atom_risk(
            record=record,
            structural_findings=structural_map.get(atom_id, []),
            signature=signature_map.get(atom_id),
            acceptability=acceptability_map.get(atom_id),
        )
        record["risk_score"] = result["risk_score"]
        record["risk_tier"] = result["risk_tier"]
        record["risk_reasons"] = result["risk_reasons"]
        record["risk_dimensions"] = result["risk_dimensions"]
        record["review_priority"] = result["review_priority"]
        record.setdefault("status_basis", {})
        record["status_basis"]["risk"] = [
            "manifest",
            "structural_report",
            "signature_evidence",
            "acceptability_scores",
        ]
        updated_atoms.append(record)
        queue_rows.append(result)
        by_tier[result["risk_tier"]] += 1
        by_family[result["domain_family"]] += 1
        for reason in result["risk_reasons"]:
            by_reason[reason] += 1

    queue_rows = _sorted_queue(queue_rows)
    manifest["atoms"] = updated_atoms
    summary = manifest.get("summary", {})
    summary["risk_tier_counts"] = dict(sorted(by_tier.items()))
    summary["review_priority_counts"] = dict(sorted(Counter(row["review_priority"] for row in queue_rows).items()))
    summary["high_risk_count"] = by_tier.get("high", 0)
    summary["medium_risk_count"] = by_tier.get("medium", 0)
    summary["low_risk_count"] = by_tier.get("low", 0)
    manifest["summary"] = summary

    manifest_map = {row["atom_id"]: row for row in updated_atoms}
    _write_review_queue(queue_rows, manifest_map)

    source_artifacts = {
        "manifest": {"path": str(AUDIT_MANIFEST_PATH), "available": AUDIT_MANIFEST_PATH.exists()},
        "structural_report": {
            "path": str(AUDIT_STRUCTURAL_REPORT_PATH),
            "available": AUDIT_STRUCTURAL_REPORT_PATH.exists(),
        },
        "signature_evidence": {
            "path": str(AUDIT_EVIDENCE_DIR),
            "available": AUDIT_EVIDENCE_DIR.exists(),
            "count": len(signature_map),
        },
        "acceptability_scores": {
            "path": str(AUDIT_SCORES_PATH),
            "available": AUDIT_SCORES_PATH.exists(),
            "count": len(acceptability_map),
        },
    }
    risk_report = {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "summary": {
            "atom_count": len(updated_atoms),
            "high_risk_count": by_tier.get("high", 0),
            "medium_risk_count": by_tier.get("medium", 0),
            "low_risk_count": by_tier.get("low", 0),
            "review_now_count": sum(1 for row in queue_rows if row["review_priority"] == "review_now"),
            "review_soon_count": sum(1 for row in queue_rows if row["review_priority"] == "review_soon"),
            "review_later_count": sum(1 for row in queue_rows if row["review_priority"] == "review_later"),
            "top_reason_codes": dict(by_reason.most_common(10)),
            "top_risky_families": dict(by_family.most_common(10)),
        },
        "thresholds": {
            "high": HIGH_RISK_THRESHOLD,
            "medium": MEDIUM_RISK_THRESHOLD,
        },
        "review_queue": queue_rows[:50],
        "by_tier": dict(sorted(by_tier.items())),
        "by_reason": dict(by_reason.most_common()),
        "by_family": dict(by_family.most_common()),
        "source_artifacts": source_artifacts,
    }
    write_json(AUDIT_RISK_REPORT_PATH, risk_report)
    write_json(AUDIT_MANIFEST_PATH, manifest)
    return manifest, risk_report
