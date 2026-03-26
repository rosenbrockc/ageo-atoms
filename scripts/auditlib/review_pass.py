"""Phase 7 review-pass workflow and trust-readiness helpers."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, write_json
from .paths import (
    AUDIT_MANIFEST_PATH,
    AUDIT_REVIEW_BACKLOG_CSV_PATH,
    AUDIT_REVIEW_INDEX_PATH,
    AUDIT_REVIEW_PROGRESS_PATH,
    AUDIT_REVIEW_VALIDATION_PATH,
    AUDIT_REVIEWS_DIR,
)
from .reviews import load_manifest, load_review_records, manifest_index, validate_review_record

SCHEMA_VERSION = "1.0"
TRUST_READY_LEVELS = {
    "positive_and_negative",
    "parity_or_usage_equivalent",
    "not_applicable",
}
VERDICT_ORDER = {
    "broken": 0,
    "misleading": 1,
    "acceptable_with_limits": 2,
    "trusted": 3,
    "unknown": 4,
}
REVIEW_PRIORITY_ORDER = {
    "review_now": 0,
    "review_soon": 1,
    "review_later": 2,
    None: 3,
}
RISK_TIER_ORDER = {
    "high": 0,
    "medium": 1,
    "low": 2,
    "unknown": 3,
    None: 3,
}
TRUST_READINESS_ORDER = {
    "not_reviewed": 0,
    "review_in_progress": 1,
    "reviewed_not_trust_ready": 2,
    "eligible_for_trusted_promotion": 3,
}
BACKLOG_REASON_ORDER = {
    "REVIEW_HIGH_RISK_UNREVIEWED": 0,
    "REVIEW_HIGH_RISK_DRAFT_ONLY": 1,
    "REVIEW_MEDIUM_RISK_BROKEN": 2,
    "REVIEW_MEDIUM_RISK_MISLEADING": 3,
    "REVIEW_PROVENANCE_GAP": 4,
    "REVIEW_LIMITATIONS_UNDOCUMENTED": 5,
    "REVIEW_TRUST_PROMOTION_CANDIDATE": 6,
    "REVIEW_OTHER": 7,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_reviewed_at(value: Any) -> tuple[int, str]:
    if not value or not isinstance(value, str):
        return (0, "")
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return (0, value)
    return (1, dt.astimezone(timezone.utc).isoformat())


def _is_provenance_complete(review: dict[str, Any]) -> bool:
    source_basis = review.get("source_basis") or {}
    return bool(
        (source_basis.get("upstream_version") or source_basis.get("source_revision"))
        and source_basis.get("review_basis_at")
    )


def _review_sort_key(review: dict[str, Any]) -> tuple[int, tuple[int, str], str]:
    status = review.get("review_status")
    status_rank = 0 if status == "completed" else 1 if status == "draft" else 2
    return (
        status_rank,
        _parse_reviewed_at(review.get("reviewed_at")),
        str(review.get("_path", "")),
    )


def _review_precedence_key(review: dict[str, Any]) -> tuple[int, tuple[int, str]]:
    status = review.get("review_status")
    status_rank = 0 if status == "completed" else 1 if status == "draft" else 2
    return (
        status_rank,
        _parse_reviewed_at(review.get("reviewed_at")),
    )


def _reviews_conflict(reviews: list[dict[str, Any]]) -> bool:
    if len(reviews) <= 1:
        return False
    latest_key = max(_review_precedence_key(review) for review in reviews)
    latest_reviews = [review for review in reviews if _review_precedence_key(review) == latest_key]
    if len(latest_reviews) <= 1:
        return False
    signatures = {
        (
            review.get("semantic_verdict"),
            review.get("developer_semantics_verdict"),
            tuple(review.get("limitations", [])),
            tuple(review.get("required_actions", [])),
        )
        for review in latest_reviews
    }
    return len(signatures) > 1 or len(latest_reviews) > 1


def load_validated_reviews(
    *,
    reviews_dir: Path = AUDIT_REVIEWS_DIR,
    manifest_path: Path = AUDIT_MANIFEST_PATH,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load review records and partition them into valid/invalid groups."""
    manifest_rows = manifest_index(load_manifest(manifest_path))
    valid_reviews: list[dict[str, Any]] = []
    invalid_reviews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for review in load_review_records(reviews_dir):
        review_errors, review_warnings = validate_review_record(review, manifest_rows)
        if review_errors:
            invalid_reviews.append(review)
            for error in review_errors:
                errors.append({"path": review.get("_path"), "atom_id": review.get("atom_id"), "error": error})
        else:
            valid_reviews.append(review)
        for warning in review_warnings:
            warnings.append({"path": review.get("_path"), "atom_id": review.get("atom_id"), "warning": warning})
    return valid_reviews, manifest_rows, errors, warnings


def resolve_active_reviews(reviews: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Resolve one active review per atom and report conflicts."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        grouped.setdefault(review["atom_id"], []).append(review)

    resolved: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for atom_id, group in grouped.items():
        active = [review for review in group if review.get("review_status") != "superseded"]
        if not active:
            continue
        completed = [review for review in active if review.get("review_status") == "completed"]
        pool = completed or active
        pool = sorted(pool, key=_review_sort_key, reverse=True)
        selected = dict(pool[0])
        selected["_conflict"] = _reviews_conflict(pool if completed else active)
        selected["_review_count"] = len(active)
        selected["_completed_review_count"] = len(completed)
        resolved[atom_id] = selected
        if selected["_conflict"]:
            conflicts.append(
                {
                    "atom_id": atom_id,
                    "paths": [review.get("_path") for review in pool],
                    "reason": "multiple_active_reviews_share_latest_precedence",
                }
            )
    return resolved, conflicts


def compute_trust_state(
    record: dict[str, Any],
    review: dict[str, Any] | None,
    *,
    review_conflict: bool = False,
) -> dict[str, Any]:
    """Compute per-atom review state, trust readiness, and blockers."""
    blockers: list[str] = []
    recommended_verdict: str | None = None

    if review is None:
        blockers.append("TRUST_BLOCK_REVIEW_MISSING")
        return {
            "review_status": "missing",
            "review_record_path": None,
            "reviewer_type": None,
            "reviewed_at": None,
            "review_semantic_verdict": "unknown",
            "review_developer_semantics_verdict": "unknown",
            "review_limitations": [],
            "review_required_actions": [],
            "trust_readiness": "not_reviewed",
            "trust_blockers": blockers,
            "recommended_overall_verdict": None,
        }

    review_status = review.get("review_status", "missing")
    state = {
        "review_status": review_status,
        "review_record_path": review.get("_path"),
        "reviewer_type": review.get("reviewer_type"),
        "reviewed_at": review.get("reviewed_at"),
        "review_semantic_verdict": review.get("semantic_verdict", "unknown"),
        "review_developer_semantics_verdict": review.get("developer_semantics_verdict", "unknown"),
        "review_limitations": list(review.get("limitations", [])),
        "review_required_actions": list(review.get("required_actions", [])),
    }

    if review_status == "draft":
        blockers.append("TRUST_BLOCK_REVIEW_DRAFT")
        state.update(
            {
                "trust_readiness": "review_in_progress",
                "trust_blockers": blockers,
                "recommended_overall_verdict": None,
            }
        )
        return state

    if review_conflict:
        blockers.append("TRUST_BLOCK_REVIEW_CONFLICT")
    if review.get("semantic_verdict") != "pass":
        blockers.append("TRUST_BLOCK_SEMANTIC_REVIEW_NOT_PASS")
    if review.get("developer_semantics_verdict") not in {"pass", "not_applicable"}:
        blockers.append("TRUST_BLOCK_DEVELOPER_REVIEW_NOT_PASS")
    if record.get("structural_status") != "pass":
        blockers.append("TRUST_BLOCK_STRUCTURAL_STATUS")
    if record.get("runtime_status") not in {"pass", "not_applicable"}:
        blockers.append("TRUST_BLOCK_RUNTIME_STATUS")
    if record.get("parity_coverage_level") not in TRUST_READY_LEVELS:
        blockers.append("TRUST_BLOCK_PARITY_COVERAGE")
    if not _is_provenance_complete(review):
        blockers.append("TRUST_BLOCK_PROVENANCE_MISSING")
    if review.get("required_actions"):
        blockers.append("TRUST_BLOCK_REQUIRED_ACTIONS_OPEN")
    if review.get("limitations"):
        blockers.append("TRUST_BLOCK_LIMITATIONS_PRESENT")
    if review.get("semantic_verdict") == "pass" and record.get("semantic_status") == "fail":
        blockers.append("TRUST_BLOCK_DETERMINISTIC_CONFLICT")
    if (
        review.get("developer_semantics_verdict") in {"pass", "not_applicable"}
        and record.get("developer_semantics_status") == "fail"
    ):
        blockers.append("TRUST_BLOCK_DETERMINISTIC_CONFLICT")

    if not blockers:
        readiness = "eligible_for_trusted_promotion"
        recommended_verdict = "trusted"
    else:
        readiness = "reviewed_not_trust_ready"

    state.update(
        {
            "trust_readiness": readiness,
            "trust_blockers": sorted(set(blockers)),
            "recommended_overall_verdict": recommended_verdict,
        }
    )
    return state


def apply_review_state_to_manifest(
    manifest: dict[str, Any],
    resolved_reviews: dict[str, dict[str, Any]],
    *,
    only_completed: bool = False,
    fail_on_conflict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sync review-derived fields into the manifest without clobbering deterministic status."""
    updated_atoms: list[dict[str, Any]] = []
    synced_count = 0
    skipped_count = 0
    conflict_count = 0

    for record in manifest.get("atoms", []):
        review = resolved_reviews.get(record["atom_id"])
        review_conflict = bool(review and review.get("_conflict"))
        if fail_on_conflict and review_conflict:
            raise ValueError(f"review conflict for {record['atom_id']}")

        if review is None:
            if not only_completed:
                state = compute_trust_state(record, None)
                record.update(state)
                record.setdefault("status_basis", {})
                record["status_basis"]["review_sync"] = ["review_records", "review_validation"]
            else:
                skipped_count += 1
            updated_atoms.append(record)
            continue

        if only_completed and review.get("review_status") != "completed":
            skipped_count += 1
            updated_atoms.append(record)
            continue

        state = compute_trust_state(record, review, review_conflict=review_conflict)
        if review_conflict:
            conflict_count += 1
        record.update(state)
        record.setdefault("status_basis", {})
        record["status_basis"]["review_sync"] = ["review_records", "review_validation"]
        updated_atoms.append(record)
        synced_count += 1

    manifest["atoms"] = updated_atoms
    summary = manifest.get("summary", {})
    summary["trust_readiness_counts"] = dict(
        sorted(Counter(atom.get("trust_readiness", "not_reviewed") for atom in updated_atoms).items())
    )
    summary["review_status_counts"] = dict(
        sorted(Counter(atom.get("review_status", "missing") for atom in updated_atoms).items())
    )
    manifest["summary"] = summary
    return manifest, {
        "synced_count": synced_count,
        "skipped_count": skipped_count,
        "conflict_count": conflict_count,
    }


def _review_batch_reason(record: dict[str, Any]) -> str:
    if record.get("review_status") == "missing" and record.get("risk_tier") == "high":
        return "REVIEW_HIGH_RISK_UNREVIEWED"
    if record.get("review_status") == "draft" and record.get("risk_tier") == "high":
        return "REVIEW_HIGH_RISK_DRAFT_ONLY"
    if record.get("risk_tier") == "medium" and record.get("overall_verdict") == "broken":
        return "REVIEW_MEDIUM_RISK_BROKEN"
    if record.get("risk_tier") == "medium" and record.get("overall_verdict") == "misleading":
        return "REVIEW_MEDIUM_RISK_MISLEADING"
    if record.get("review_status") == "completed" and record.get("trust_readiness") == "eligible_for_trusted_promotion":
        return "REVIEW_TRUST_PROMOTION_CANDIDATE"
    if record.get("review_status") == "completed" and not (
        record.get("upstream_version") or record.get("source_revision")
    ):
        return "REVIEW_PROVENANCE_GAP"
    if record.get("overall_verdict") == "acceptable_with_limits" and not record.get("review_limitations"):
        return "REVIEW_LIMITATIONS_UNDOCUMENTED"
    return "REVIEW_OTHER"


def _recommended_review_action(record: dict[str, Any]) -> str:
    reason = _review_batch_reason(record)
    if reason == "REVIEW_HIGH_RISK_UNREVIEWED":
        return "start a structured review immediately"
    if reason == "REVIEW_HIGH_RISK_DRAFT_ONLY":
        return "complete the existing draft review"
    if reason in {"REVIEW_MEDIUM_RISK_BROKEN", "REVIEW_MEDIUM_RISK_MISLEADING"}:
        return "review before remediation or promotion"
    if reason == "REVIEW_PROVENANCE_GAP":
        return "pin upstream version or source revision in the review record"
    if reason == "REVIEW_TRUST_PROMOTION_CANDIDATE":
        return "manual promotion decision may proceed"
    if reason == "REVIEW_LIMITATIONS_UNDOCUMENTED":
        return "document limitations before broader reuse"
    return "review as capacity allows"


def build_review_backlog_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in manifest.get("atoms", []):
        row = {
            "atom_id": record["atom_id"],
            "atom_name": record["atom_name"],
            "review_priority": record.get("review_priority"),
            "risk_tier": record.get("risk_tier"),
            "risk_score": record.get("risk_score"),
            "overall_verdict": record.get("overall_verdict"),
            "semantic_status": record.get("semantic_status"),
            "review_status": record.get("review_status", "missing"),
            "has_completed_review": record.get("review_status") == "completed",
            "trust_readiness": record.get("trust_readiness", "not_reviewed"),
            "review_batch_reason": _review_batch_reason(record),
            "recommended_review_action": _recommended_review_action(record),
            "trust_blockers": list(record.get("trust_blockers", [])),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            REVIEW_PRIORITY_ORDER.get(row["review_priority"], 3),
            BACKLOG_REASON_ORDER.get(row["review_batch_reason"], 99),
            RISK_TIER_ORDER.get(row["risk_tier"], 3),
            -int(row["risk_score"] or 0),
            VERDICT_ORDER.get(row["overall_verdict"], 99),
            row["atom_id"],
        )
    )
    return rows


def write_review_backlog_csv(
    rows: list[dict[str, Any]],
    output_path: Path = AUDIT_REVIEW_BACKLOG_CSV_PATH,
) -> None:
    fieldnames = [
        "atom_id",
        "atom_name",
        "review_priority",
        "risk_tier",
        "risk_score",
        "overall_verdict",
        "semantic_status",
        "review_status",
        "has_completed_review",
        "trust_readiness",
        "review_batch_reason",
        "recommended_review_action",
        "trust_blockers",
    ]
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["trust_blockers"] = ";".join(row["trust_blockers"])
            writer.writerow(payload)


def build_review_progress_payload(
    manifest: dict[str, Any],
    backlog_rows: list[dict[str, Any]],
    *,
    review_conflicts: list[dict[str, Any]],
    validation_errors: list[dict[str, Any]],
    validation_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    atoms = manifest.get("atoms", [])
    by_review_status = Counter(atom.get("review_status", "missing") for atom in atoms)
    by_reviewer_type = Counter((atom.get("reviewer_type") or "none") for atom in atoms)
    by_risk_tier = Counter(atom.get("risk_tier", "unknown") for atom in atoms)
    by_trust_readiness = Counter(atom.get("trust_readiness", "not_reviewed") for atom in atoms)
    high_priority_backlog = [row for row in backlog_rows if row["review_priority"] == "review_now"][:50]
    promotion_candidates = [
        {
            "atom_id": atom["atom_id"],
            "atom_name": atom["atom_name"],
            "review_record_path": atom.get("review_record_path"),
            "recommended_overall_verdict": atom.get("recommended_overall_verdict"),
        }
        for atom in atoms
        if atom.get("trust_readiness") == "eligible_for_trusted_promotion"
    ]
    blocked_candidates = [
        {
            "atom_id": atom["atom_id"],
            "atom_name": atom["atom_name"],
            "review_status": atom.get("review_status"),
            "trust_blockers": atom.get("trust_blockers", []),
        }
        for atom in atoms
        if atom.get("review_status") == "completed" and atom.get("trust_readiness") != "eligible_for_trusted_promotion"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "summary": {
            "atom_count": len(atoms),
            "completed_review_count": by_review_status.get("completed", 0),
            "draft_review_count": by_review_status.get("draft", 0),
            "missing_review_count": by_review_status.get("missing", 0),
            "high_risk_reviewed_count": sum(
                1 for atom in atoms if atom.get("risk_tier") == "high" and atom.get("review_status") == "completed"
            ),
            "high_risk_unreviewed_count": sum(
                1 for atom in atoms if atom.get("risk_tier") == "high" and atom.get("review_status") == "missing"
            ),
            "trust_ready_count": by_trust_readiness.get("eligible_for_trusted_promotion", 0),
            "trust_blocked_count": by_trust_readiness.get("reviewed_not_trust_ready", 0),
            "review_conflict_count": len(review_conflicts),
            "validation_error_count": len(validation_errors),
            "validation_warning_count": len(validation_warnings),
        },
        "by_review_status": dict(sorted(by_review_status.items())),
        "by_reviewer_type": dict(sorted(by_reviewer_type.items())),
        "by_risk_tier": dict(sorted(by_risk_tier.items())),
        "by_trust_readiness": dict(
            sorted(by_trust_readiness.items(), key=lambda item: TRUST_READINESS_ORDER.get(item[0], 99))
        ),
        "high_priority_backlog": high_priority_backlog,
        "promotion_candidates": promotion_candidates,
        "blocked_candidates": blocked_candidates,
        "review_conflicts": review_conflicts,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
    }


def build_review_pass(
    *,
    manifest_path: Path = AUDIT_MANIFEST_PATH,
    reviews_dir: Path = AUDIT_REVIEWS_DIR,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Join manifest and reviews into a review-progress and backlog view."""
    manifest = read_json(manifest_path)
    valid_reviews, _, validation_errors, validation_warnings = load_validated_reviews(
        reviews_dir=reviews_dir,
        manifest_path=manifest_path,
    )
    resolved_reviews, review_conflicts = resolve_active_reviews(valid_reviews)
    updated_manifest, sync_summary = apply_review_state_to_manifest(manifest, resolved_reviews, only_completed=False)
    backlog_rows = build_review_backlog_rows(updated_manifest)
    progress = build_review_progress_payload(
        updated_manifest,
        backlog_rows,
        review_conflicts=review_conflicts,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )
    return updated_manifest, progress, backlog_rows, sync_summary


def write_review_progress(
    progress: dict[str, Any],
    backlog_rows: list[dict[str, Any]],
    *,
    progress_path: Path = AUDIT_REVIEW_PROGRESS_PATH,
    backlog_path: Path = AUDIT_REVIEW_BACKLOG_CSV_PATH,
) -> None:
    write_json(progress_path, progress)
    write_review_backlog_csv(backlog_rows, backlog_path)


def refresh_review_index(
    reviews: list[dict[str, Any]],
    *,
    index_path: Path = AUDIT_REVIEW_INDEX_PATH,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "summary": {
            "review_count": len(reviews),
            "by_status": dict(sorted(Counter(review.get("review_status", "missing") for review in reviews).items())),
            "by_reviewer_type": dict(
                sorted(Counter(review.get("reviewer_type", "unknown") for review in reviews).items())
            ),
        },
        "reviews": [
            {
                "atom_id": review.get("atom_id"),
                "path": review.get("_path"),
                "review_status": review.get("review_status"),
                "reviewer_type": review.get("reviewer_type"),
                "semantic_verdict": review.get("semantic_verdict"),
                "developer_semantics_verdict": review.get("developer_semantics_verdict"),
                "reviewed_at": review.get("reviewed_at"),
                "required_actions_count": len(review.get("required_actions", [])),
            }
            for review in reviews
        ],
    }
    write_json(index_path, payload)


def write_validation_payload(
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    valid_reviews: list[dict[str, Any]],
    *,
    validation_path: Path = AUDIT_REVIEW_VALIDATION_PATH,
) -> dict[str, Any]:
    review_paths = {review.get("_path") for review in valid_reviews if review.get("_path")}
    review_paths.update(error.get("path") for error in errors if error.get("path"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "ok": not errors,
        "summary": {
            "review_count": len(review_paths),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(validation_path, payload)
    return payload
