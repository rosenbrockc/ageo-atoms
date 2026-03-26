"""Helpers for Phase 4 semantic review records."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, safe_atom_stem, write_json
from .models import ReviewRecord
from .paths import (
    AUDIT_EVIDENCE_DIR,
    AUDIT_MANIFEST_PATH,
    AUDIT_REVIEW_INDEX_PATH,
    AUDIT_REVIEW_QUEUE_CSV_PATH,
    AUDIT_REVIEW_VALIDATION_PATH,
    AUDIT_RESULTS_DIR,
    AUDIT_REVIEWS_DIR,
)

SCHEMA_VERSION = "1.0"
REVIEW_STATUS_VALUES = {"draft", "completed", "superseded"}
REVIEWER_TYPE_VALUES = {"human", "model", "human_verified_model"}
VERDICT_VALUES = {"pass", "partial", "fail", "not_applicable", "unknown"}
REQUIRED_KEYS = {
    "schema_version",
    "atom_id",
    "atom_name",
    "review_status",
    "reviewer_type",
    "reviewed_at",
    "upstream_symbols",
    "authoritative_sources",
    "source_basis",
    "line_references",
    "wrapper_truth",
    "api_truth",
    "state_truth",
    "output_truth",
    "decomposition_truth",
    "semantic_verdict",
    "developer_semantics_verdict",
    "limitations",
    "required_actions",
    "notes",
}
REVIEW_PRIORITY_VALUES = {"review_now", "review_soon", "review_later"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_manifest(manifest_path: Path = AUDIT_MANIFEST_PATH) -> dict[str, Any]:
    """Load the current audit manifest."""
    return read_json(manifest_path)


def manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build an atom_id keyed index of manifest rows."""
    return {row["atom_id"]: row for row in manifest.get("atoms", [])}


def review_path_for_atom(atom_id: str, reviews_dir: Path = AUDIT_REVIEWS_DIR) -> Path:
    """Return the canonical review path for one atom."""
    return reviews_dir / f"{safe_atom_stem(atom_id)}.json"


def load_review_records(reviews_dir: Path = AUDIT_REVIEWS_DIR) -> list[dict[str, Any]]:
    """Load all review JSON files, preserving source path metadata."""
    ensure_dir(reviews_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(reviews_dir.glob("*.json")):
        record = read_json(path)
        if isinstance(record, dict):
            record = dict(record)
            record["_path"] = str(path)
            records.append(record)
    return records


def _supporting_artifacts(atom_id: str) -> dict[str, Any]:
    stem = safe_atom_stem(atom_id)
    payload: dict[str, Any] = {}
    evidence_path = AUDIT_EVIDENCE_DIR / f"{stem}.json"
    result_path = AUDIT_RESULTS_DIR / f"{stem}.json"
    if evidence_path.exists():
        payload["signature_evidence"] = str(evidence_path)
    if result_path.exists():
        payload["acceptability_result"] = str(result_path)
    return payload


def _canonical_authoritative_sources(record: dict[str, Any]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for item in record.get("authoritative_sources", []):
        kind = item.get("kind", "repository")
        if kind == "local_wrapper":
            canonical.append(
                {
                    "kind": "local_wrapper",
                    "label": "Local wrapper",
                    "reference": item.get("path", record.get("module_path")),
                    "relevance": "Wrapper implementation under review.",
                }
            )
        elif kind == "vendored_repo":
            ref = item.get("module") or item.get("repo")
            canonical.append(
                {
                    "kind": "vendored_source",
                    "label": item.get("repo", "Vendored source"),
                    "reference": ref,
                    "relevance": "Vendored upstream source anchor.",
                }
            )
        elif kind == "local_references":
            canonical.append(
                {
                    "kind": "repository",
                    "label": "Local references",
                    "reference": item.get("path"),
                    "relevance": "Repository-local reference metadata.",
                }
            )
        else:
            canonical.append(
                {
                    "kind": kind,
                    "label": item.get("label", kind),
                    "reference": item.get("reference") or item.get("path") or item.get("module") or item.get("repo"),
                    "relevance": item.get("relevance", "Imported from manifest authoritative_sources."),
                }
            )
    return canonical


def _draft_line_references(record: dict[str, Any]) -> list[dict[str, Any]]:
    refs = [
        {
            "scope": "wrapper",
            "path": record["module_path"],
            "line": record["wrapper_line"],
            "note": "Wrapper entry point for semantic review.",
        }
    ]
    for source in record.get("authoritative_sources", []):
        if source.get("kind") != "vendored_repo":
            continue
        module = source.get("module")
        if not module:
            continue
        refs.append(
            {
                "scope": "vendored_source",
                "path": module,
                "line": 1,
                "note": "Vendored upstream module reference; exact line should be refined by reviewer.",
            }
        )
        break
    return refs


def make_draft_review(record: dict[str, Any], review_priority: str | None = None) -> ReviewRecord:
    """Seed a draft review from a manifest row."""
    notes = []
    if record.get("blocking_findings"):
        notes.append("Seeded from manifest blocking_findings.")
    review = ReviewRecord(
        schema_version=SCHEMA_VERSION,
        atom_id=record["atom_id"],
        atom_name=record["atom_name"],
        review_status="draft",
        reviewer_type="human",
        reviewed_at=None,
        upstream_symbols=record.get("upstream_symbols", {}),
        authoritative_sources=_canonical_authoritative_sources(record),
        source_basis={
            "upstream_version": record.get("upstream_version"),
            "source_revision": record.get("source_revision"),
            "review_basis_at": record.get("review_basis_at"),
        },
        line_references=_draft_line_references(record),
        limitations=[],
        required_actions=list(record.get("required_actions", [])),
        notes=notes,
        supporting_artifacts=_supporting_artifacts(record["atom_id"]),
        seed_context={
            "risk_tier": record.get("risk_tier"),
            "risk_score": record.get("risk_score"),
            "review_priority": review_priority or record.get("review_priority"),
            "risk_reasons": list(record.get("risk_reasons", [])),
            "blocking_findings": list(record.get("blocking_findings", [])),
            "structural_findings": list(record.get("structural_findings", [])),
        },
    )
    return review


def _queue_rows(queue_path: Path = AUDIT_REVIEW_QUEUE_CSV_PATH) -> list[dict[str, str]]:
    if not queue_path.exists():
        return []
    with queue_path.open() as handle:
        return list(csv.DictReader(handle))


def _select_atom_ids(
    *,
    manifest: dict[str, Any],
    atom_id: str | None,
    priority: str | None,
    limit: int | None,
    queue_path: Path,
) -> list[tuple[str, str | None]]:
    if atom_id:
        return [(atom_id, None)]
    rows = _queue_rows(queue_path)
    selected: list[tuple[str, str | None]] = []
    for row in rows:
        if priority and row.get("review_priority") != priority:
            continue
        selected.append((row["atom_id"], row.get("review_priority")))
        if limit is not None and len(selected) >= limit:
            break
    if selected:
        return selected
    manifest_rows = manifest.get("atoms", [])
    return [(row["atom_id"], None) for row in manifest_rows[: limit or 10]]


def seed_review_records(
    *,
    manifest_path: Path = AUDIT_MANIFEST_PATH,
    queue_path: Path = AUDIT_REVIEW_QUEUE_CSV_PATH,
    reviews_dir: Path = AUDIT_REVIEWS_DIR,
    atom_id: str | None = None,
    priority: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Seed draft review records from the manifest and review queue."""
    manifest = load_manifest(manifest_path)
    by_atom = manifest_index(manifest)
    ensure_dir(reviews_dir)
    selected = _select_atom_ids(
        manifest=manifest,
        atom_id=atom_id,
        priority=priority,
        limit=limit,
        queue_path=queue_path,
    )
    seeded: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []
    for current_atom_id, review_priority in selected:
        record = by_atom.get(current_atom_id)
        if record is None:
            missing.append(current_atom_id)
            continue
        target_path = review_path_for_atom(current_atom_id, reviews_dir)
        if target_path.exists() and not force:
            existing = read_json(target_path)
            if existing.get("review_status") == "completed":
                skipped.append(current_atom_id)
                continue
            skipped.append(current_atom_id)
            continue
        review = make_draft_review(record, review_priority=review_priority)
        write_json(target_path, asdict(review))
        seeded.append(current_atom_id)
    return {
        "seeded_count": len(seeded),
        "skipped_count": len(skipped),
        "missing_count": len(missing),
        "seeded_atom_ids": seeded,
        "skipped_atom_ids": skipped,
        "missing_atom_ids": missing,
    }


def _validate_line_reference(item: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["line_references entry must be an object"]
    for key in ("scope", "path", "line", "note"):
        if key not in item:
            errors.append(f"line_references entry missing {key}")
    if item.get("scope") not in {"wrapper", "vendored_source", "official_docs"}:
        errors.append("line_references scope must be wrapper, vendored_source, or official_docs")
    if "line" in item and (not isinstance(item["line"], int) or item["line"] <= 0):
        errors.append("line_references line must be a positive integer")
    return errors


def _validate_authoritative_source(item: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["authoritative_sources entry must be an object"]
    for key in ("kind", "label", "reference", "relevance"):
        if key not in item:
            errors.append(f"authoritative_sources entry missing {key}")
    return errors


def validate_review_record(record: dict[str, Any], manifest_rows: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Validate one review record."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_keys = sorted(REQUIRED_KEYS - set(record))
    for key in missing_keys:
        errors.append(f"missing required key: {key}")

    if record.get("review_status") not in REVIEW_STATUS_VALUES:
        errors.append("review_status must be draft, completed, or superseded")
    if record.get("reviewer_type") not in REVIEWER_TYPE_VALUES:
        errors.append("reviewer_type must be human, model, or human_verified_model")
    for key in (
        "wrapper_truth",
        "api_truth",
        "state_truth",
        "output_truth",
        "decomposition_truth",
        "semantic_verdict",
        "developer_semantics_verdict",
    ):
        if record.get(key) not in VERDICT_VALUES:
            errors.append(f"{key} must be one of {sorted(VERDICT_VALUES)}")

    atom_id = record.get("atom_id")
    manifest_row = manifest_rows.get(atom_id)
    if manifest_row is None:
        errors.append("atom_id not present in manifest")
    else:
        if record.get("atom_name") != manifest_row.get("atom_name"):
            errors.append("atom_name does not match manifest")

    source_basis = record.get("source_basis")
    if not isinstance(source_basis, dict):
        errors.append("source_basis must be an object")
        source_basis = {}
    else:
        for key in ("upstream_version", "source_revision", "review_basis_at"):
            if key not in source_basis:
                errors.append(f"source_basis missing {key}")

    line_refs = record.get("line_references", [])
    if not isinstance(line_refs, list):
        errors.append("line_references must be a list")
        line_refs = []
    else:
        for item in line_refs:
            errors.extend(_validate_line_reference(item))

    sources = record.get("authoritative_sources", [])
    if not isinstance(sources, list):
        errors.append("authoritative_sources must be a list")
        sources = []
    else:
        for item in sources:
            errors.extend(_validate_authoritative_source(item))

    if record.get("review_status") == "completed":
        if not (source_basis.get("upstream_version") or source_basis.get("source_revision")):
            errors.append("completed review must include upstream_version or source_revision")
        if not source_basis.get("review_basis_at"):
            errors.append("completed review must include review_basis_at")
        if not sources:
            errors.append("completed review must include authoritative_sources")
        if not line_refs:
            errors.append("completed review must include line_references")
        if record.get("semantic_verdict") == "unknown":
            errors.append("completed review must set semantic_verdict")
        if record.get("developer_semantics_verdict") == "unknown":
            errors.append("completed review must set developer_semantics_verdict")
        if (
            record.get("semantic_verdict") in {"partial", "fail"}
            or record.get("developer_semantics_verdict") in {"partial", "fail"}
        ) and not (record.get("limitations") or record.get("required_actions")):
            errors.append("completed review with partial/fail verdict must include limitations or required_actions")
    else:
        if not (source_basis.get("upstream_version") or source_basis.get("source_revision")):
            warnings.append("draft review is missing reproducible provenance")
        if not source_basis.get("review_basis_at"):
            warnings.append("draft review is missing review_basis_at")
    return errors, warnings


def _review_index_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "atom_id": record["atom_id"],
        "path": record.get("_path"),
        "review_status": record["review_status"],
        "reviewer_type": record["reviewer_type"],
        "semantic_verdict": record["semantic_verdict"],
        "developer_semantics_verdict": record["developer_semantics_verdict"],
        "reviewed_at": record["reviewed_at"],
        "required_actions_count": len(record.get("required_actions", [])),
    }


def validate_review_directory(
    *,
    reviews_dir: Path = AUDIT_REVIEWS_DIR,
    manifest_path: Path = AUDIT_MANIFEST_PATH,
    validation_path: Path = AUDIT_REVIEW_VALIDATION_PATH,
    index_path: Path = AUDIT_REVIEW_INDEX_PATH,
) -> dict[str, Any]:
    """Validate all review records and emit validation/index artifacts."""
    manifest_rows = manifest_index(load_manifest(manifest_path))
    ensure_dir(reviews_dir)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    review_index_entries: list[dict[str, Any]] = []

    for path in sorted(reviews_dir.glob("*.json")):
        try:
            record = read_json(path)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"invalid json: {exc}"})
            continue
        record = dict(record)
        record["_path"] = str(path)
        review_entries.append({"path": str(path), "atom_id": record.get("atom_id")})
        record_errors, record_warnings = validate_review_record(record, manifest_rows)
        for error in record_errors:
            errors.append({"path": str(path), "atom_id": record.get("atom_id"), "error": error})
        for warning in record_warnings:
            warnings.append({"path": str(path), "atom_id": record.get("atom_id"), "warning": warning})
        review_index_entries.append(_review_index_entry(record))

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "summary": {
            "review_count": len(review_index_entries),
            "by_status": _counter(review["review_status"] for review in review_index_entries),
            "by_reviewer_type": _counter(review["reviewer_type"] for review in review_index_entries),
            "by_semantic_verdict": _counter(review["semantic_verdict"] for review in review_index_entries),
        },
        "reviews": review_index_entries,
    }
    validation_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "ok": not errors,
        "summary": {
            "review_count": len(review_index_entries),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(index_path, index_payload)
    write_json(validation_path, validation_payload)
    return validation_payload


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))
