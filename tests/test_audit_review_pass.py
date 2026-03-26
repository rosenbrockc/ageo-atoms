from __future__ import annotations

import csv
import json
from pathlib import Path

from auditlib.io import read_json
from auditlib.review_pass import (
    apply_review_state_to_manifest,
    build_review_backlog_rows,
    build_review_pass,
    compute_trust_state,
    resolve_active_reviews,
    write_review_progress,
)


def _atom(
    *,
    atom_id: str,
    risk_tier: str = "high",
    review_priority: str = "review_now",
    structural_status: str = "pass",
    runtime_status: str = "pass",
    parity_coverage_level: str = "parity_or_usage_equivalent",
    semantic_status: str = "pass",
    developer_semantics_status: str = "pass",
    overall_verdict: str = "acceptable_with_limits",
    risk_score: int = 80,
) -> dict:
    return {
        "atom_id": atom_id,
        "atom_name": atom_id.split("@", 1)[0],
        "review_priority": review_priority,
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "structural_status": structural_status,
        "runtime_status": runtime_status,
        "parity_coverage_level": parity_coverage_level,
        "semantic_status": semantic_status,
        "developer_semantics_status": developer_semantics_status,
        "overall_verdict": overall_verdict,
        "upstream_version": "1.0.0",
        "source_revision": "abc123",
        "status_basis": {},
    }


def _review(
    atom_id: str,
    *,
    path: str,
    review_status: str = "completed",
    reviewed_at: str = "2026-03-26T00:00:00+00:00",
    semantic_verdict: str = "pass",
    developer_semantics_verdict: str = "pass",
    limitations: list[str] | None = None,
    required_actions: list[str] | None = None,
    source_revision: str | None = "abc123",
    upstream_version: str | None = None,
    review_basis_at: str | None = "2026-03-26T00:00:00+00:00",
) -> dict:
    return {
        "_path": path,
        "schema_version": "1.0",
        "atom_id": atom_id,
        "atom_name": atom_id.split("@", 1)[0],
        "review_status": review_status,
        "reviewer_type": "human",
        "reviewed_at": reviewed_at,
        "upstream_symbols": {"repo": "Repo"},
        "authoritative_sources": [
            {
                "kind": "local_wrapper",
                "label": "wrapper",
                "reference": "ageoa/example.py",
                "relevance": "implementation",
            }
        ],
        "source_basis": {
            "upstream_version": upstream_version,
            "source_revision": source_revision,
            "review_basis_at": review_basis_at,
        },
        "line_references": [
            {
                "scope": "wrapper",
                "path": "ageoa/example.py",
                "line": 10,
                "note": "wrapper",
            }
        ],
        "wrapper_truth": "pass",
        "api_truth": "pass",
        "state_truth": "not_applicable",
        "output_truth": "pass",
        "decomposition_truth": "pass",
        "semantic_verdict": semantic_verdict,
        "developer_semantics_verdict": developer_semantics_verdict,
        "limitations": limitations or [],
        "required_actions": required_actions or [],
        "notes": [],
    }


def test_compute_trust_state_marks_completed_review_ready() -> None:
    state = compute_trust_state(
        _atom(atom_id="ageoa.example.ready@ageoa/example.py:10"),
        _review("ageoa.example.ready@ageoa/example.py:10", path="review.json"),
    )
    assert state["trust_readiness"] == "eligible_for_trusted_promotion"
    assert state["trust_blockers"] == []
    assert state["recommended_overall_verdict"] == "trusted"


def test_compute_trust_state_marks_draft_in_progress() -> None:
    state = compute_trust_state(
        _atom(atom_id="ageoa.example.draft@ageoa/example.py:10"),
        _review(
            "ageoa.example.draft@ageoa/example.py:10",
            path="review.json",
            review_status="draft",
            reviewed_at=None,
        ),
    )
    assert state["trust_readiness"] == "review_in_progress"
    assert state["trust_blockers"] == ["TRUST_BLOCK_REVIEW_DRAFT"]


def test_compute_trust_state_marks_missing_review() -> None:
    state = compute_trust_state(_atom(atom_id="ageoa.example.missing@ageoa/example.py:10"), None)
    assert state["trust_readiness"] == "not_reviewed"
    assert state["trust_blockers"] == ["TRUST_BLOCK_REVIEW_MISSING"]


def test_compute_trust_state_blocks_incomplete_completed_review() -> None:
    state = compute_trust_state(
        _atom(
            atom_id="ageoa.example.blocked@ageoa/example.py:10",
            structural_status="partial",
            runtime_status="unknown",
            parity_coverage_level="positive_path",
        ),
        _review(
            "ageoa.example.blocked@ageoa/example.py:10",
            path="review.json",
            semantic_verdict="partial",
            developer_semantics_verdict="partial",
            limitations=["behavior differs under edge cases"],
            required_actions=["collect upstream examples"],
            review_basis_at=None,
        ),
    )
    assert state["trust_readiness"] == "reviewed_not_trust_ready"
    assert "TRUST_BLOCK_SEMANTIC_REVIEW_NOT_PASS" in state["trust_blockers"]
    assert "TRUST_BLOCK_DEVELOPER_REVIEW_NOT_PASS" in state["trust_blockers"]
    assert "TRUST_BLOCK_STRUCTURAL_STATUS" in state["trust_blockers"]
    assert "TRUST_BLOCK_RUNTIME_STATUS" in state["trust_blockers"]
    assert "TRUST_BLOCK_PARITY_COVERAGE" in state["trust_blockers"]
    assert "TRUST_BLOCK_PROVENANCE_MISSING" in state["trust_blockers"]
    assert "TRUST_BLOCK_LIMITATIONS_PRESENT" in state["trust_blockers"]
    assert "TRUST_BLOCK_REQUIRED_ACTIONS_OPEN" in state["trust_blockers"]


def test_resolve_active_reviews_flags_conflicting_completed_reviews() -> None:
    atom_id = "ageoa.example.conflict@ageoa/example.py:10"
    resolved, conflicts = resolve_active_reviews(
        [
            _review(atom_id, path="a.json", semantic_verdict="pass"),
            _review(atom_id, path="b.json", semantic_verdict="fail"),
        ]
    )
    assert resolved[atom_id]["_conflict"] is True
    assert conflicts[0]["atom_id"] == atom_id


def test_compute_trust_state_blocks_deterministic_conflict() -> None:
    state = compute_trust_state(
        _atom(
            atom_id="ageoa.example.conflicted@ageoa/example.py:10",
            semantic_status="fail",
            developer_semantics_status="fail",
        ),
        _review("ageoa.example.conflicted@ageoa/example.py:10", path="review.json"),
    )
    assert "TRUST_BLOCK_DETERMINISTIC_CONFLICT" in state["trust_blockers"]


def test_backlog_ordering_prioritizes_high_risk_unreviewed() -> None:
    manifest = {
        "atoms": [
            {
                **_atom(atom_id="ageoa.example.high@ageoa/example.py:10", risk_tier="high", risk_score=90),
                "review_status": "missing",
                "trust_readiness": "not_reviewed",
                "review_limitations": [],
                "trust_blockers": ["TRUST_BLOCK_REVIEW_MISSING"],
            },
            {
                **_atom(atom_id="ageoa.example.medium@ageoa/example.py:20", risk_tier="medium", risk_score=70),
                "review_status": "draft",
                "trust_readiness": "review_in_progress",
                "review_limitations": [],
                "trust_blockers": ["TRUST_BLOCK_REVIEW_DRAFT"],
            },
        ]
    }
    rows = build_review_backlog_rows(manifest)
    assert rows[0]["atom_id"] == "ageoa.example.high@ageoa/example.py:10"
    assert rows[0]["review_batch_reason"] == "REVIEW_HIGH_RISK_UNREVIEWED"


def test_apply_review_state_updates_review_fields_without_clobbering_deterministic_status() -> None:
    atom_id = "ageoa.example.sync@ageoa/example.py:10"
    manifest = {"atoms": [_atom(atom_id=atom_id, structural_status="fail", runtime_status="pass")], "summary": {}}
    updated_manifest, summary = apply_review_state_to_manifest(
        manifest,
        {atom_id: _review(atom_id, path="review.json", required_actions=["fix output drift"])},
        only_completed=True,
    )
    record = updated_manifest["atoms"][0]
    assert record["review_status"] == "completed"
    assert record["review_record_path"] == "review.json"
    assert record["review_required_actions"] == ["fix output drift"]
    assert record["structural_status"] == "fail"
    assert record["runtime_status"] == "pass"
    assert record["trust_readiness"] == "reviewed_not_trust_ready"
    assert summary["synced_count"] == 1


def test_build_review_pass_and_write_artifacts(tmp_path: Path) -> None:
    atom_id = "ageoa.example.artifact@ageoa/example.py:10"
    manifest_path = tmp_path / "audit_manifest.json"
    reviews_dir = tmp_path / "audit_reviews"
    progress_path = tmp_path / "review_progress.json"
    backlog_path = tmp_path / "review_backlog.csv"
    reviews_dir.mkdir()
    manifest = {"atoms": [_atom(atom_id=atom_id)], "summary": {}}
    manifest_path.write_text(json.dumps(manifest))
    (reviews_dir / "review.json").write_text(json.dumps(_review(atom_id, path=str(reviews_dir / "review.json"))))

    updated_manifest, progress, backlog_rows, _ = build_review_pass(
        manifest_path=manifest_path,
        reviews_dir=reviews_dir,
    )
    write_review_progress(progress, backlog_rows, progress_path=progress_path, backlog_path=backlog_path)

    assert updated_manifest["atoms"][0]["trust_readiness"] == "eligible_for_trusted_promotion"
    assert progress_path.exists()
    assert backlog_path.exists()
    progress_payload = read_json(progress_path)
    assert progress_payload["summary"]["trust_ready_count"] == 1
    with backlog_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["atom_id"] == atom_id
