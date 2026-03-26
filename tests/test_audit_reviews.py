from __future__ import annotations

import json
from pathlib import Path

from auditlib.io import safe_atom_stem
from auditlib.reviews import (
    cleanup_orphan_review_records,
    make_draft_review,
    seed_review_records,
    validate_review_directory,
    validate_review_record,
)


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    manifest = {
        "atoms": [
            {
                "atom_id": "ageoa.example.atom@ageoa/example.py:10",
                "atom_name": "ageoa.example.atom",
                "atom_key": "example:atom",
                "module_path": "ageoa/example.py",
                "wrapper_line": 10,
                "upstream_symbols": {"repo": "Repo", "module": "pkg.mod", "function": "atom", "language": "python"},
                "authoritative_sources": [
                    {
                        "kind": "local_wrapper",
                        "label": "wrapper",
                        "reference": "ageoa/example.py",
                        "relevance": "implementation",
                    }
                ],
                "upstream_version": None,
                "source_revision": "abc123",
                "review_basis_at": "2026-03-25T00:00:00+00:00",
                "required_actions": ["inspect upstream state handling"],
                "risk_tier": "high",
                "risk_score": 80,
                "review_priority": "review_now",
                "risk_reasons": ["RISK_STATEFUL_API"],
                "blocking_findings": ["STRUCT_CDG_INVALID"],
                "structural_findings": ["STRUCT_CDG_INVALID"],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, manifest


def _queue(tmp_path: Path, atom_id: str) -> Path:
    queue_path = tmp_path / "review_queue.csv"
    queue_path.write_text(
        "atom_id,atom_name,domain_family,risk_score,risk_tier,review_priority,risk_reasons,structural_status,semantic_status,max_reviewable_verdict\n"
        f"{atom_id},ageoa.example.atom,example,80,high,review_now,RISK_STATEFUL_API,partial,unknown,\n"
    )
    return queue_path


def test_make_draft_review_seeds_manifest_fields() -> None:
    record = {
        "atom_id": "ageoa.example.atom@ageoa/example.py:10",
        "atom_name": "ageoa.example.atom",
        "module_path": "ageoa/example.py",
        "wrapper_line": 10,
        "upstream_symbols": {"repo": "Repo"},
        "authoritative_sources": [],
        "upstream_version": None,
        "source_revision": "abc123",
        "review_basis_at": None,
        "required_actions": ["follow up"],
        "risk_tier": "high",
        "risk_score": 75,
        "review_priority": "review_now",
        "risk_reasons": ["RISK_UPSTREAM_UNMAPPED"],
        "blocking_findings": [],
        "structural_findings": [],
    }
    review = make_draft_review(record)
    assert review.review_status == "draft"
    assert review.source_basis["source_revision"] == "abc123"
    assert review.seed_context["review_priority"] == "review_now"


def test_seed_review_records_from_priority_queue(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    queue_path = _queue(tmp_path, manifest["atoms"][0]["atom_id"])
    reviews_dir = tmp_path / "reviews"
    result = seed_review_records(
        manifest_path=manifest_path,
        queue_path=queue_path,
        reviews_dir=reviews_dir,
        priority="review_now",
        limit=1,
    )
    assert result["seeded_count"] == 1
    review_path = reviews_dir / f"{safe_atom_stem(manifest['atoms'][0]['atom_id'])}.json"
    assert review_path.exists()


def test_validate_review_record_requires_provenance_for_completed(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    record = make_draft_review(manifest["atoms"][0]).to_dict()
    record["review_status"] = "completed"
    record["reviewer_type"] = "human"
    record["semantic_verdict"] = "pass"
    record["developer_semantics_verdict"] = "pass"
    record["source_basis"]["source_revision"] = None
    record["source_basis"]["upstream_version"] = None
    errors, warnings = validate_review_record(record, {manifest["atoms"][0]["atom_id"]: manifest["atoms"][0]})
    assert any("upstream_version or source_revision" in error for error in errors)
    assert warnings == []


def test_validate_review_directory_builds_index(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    record = make_draft_review(manifest["atoms"][0]).to_dict()
    record["review_status"] = "completed"
    record["reviewer_type"] = "human"
    record["reviewed_at"] = "2026-03-25T00:00:00+00:00"
    record["semantic_verdict"] = "pass"
    record["developer_semantics_verdict"] = "pass"
    record["authoritative_sources"] = [
        {
            "kind": "local_wrapper",
            "label": "wrapper",
            "reference": "ageoa/example.py",
            "relevance": "implementation",
        }
    ]
    review_path = reviews_dir / f"{safe_atom_stem(record['atom_id'])}.json"
    review_path.write_text(json.dumps(record))
    validation_path = tmp_path / "review_validation.json"
    index_path = tmp_path / "review_index.json"
    payload = validate_review_directory(
        reviews_dir=reviews_dir,
        manifest_path=manifest_path,
        validation_path=validation_path,
        index_path=index_path,
    )
    assert payload["ok"] is True
    assert validation_path.exists()
    assert index_path.exists()


def test_cleanup_orphan_review_records_deletes_missing_manifest_rows(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    valid_record = make_draft_review(manifest["atoms"][0]).to_dict()
    valid_path = reviews_dir / f"{safe_atom_stem(valid_record['atom_id'])}.json"
    valid_path.write_text(json.dumps(valid_record))

    orphan_record = dict(valid_record)
    orphan_record["atom_id"] = "ageoa.sklearn.stale.atom@ageoa/sklearn/stale/atoms.py:10"
    orphan_record["atom_name"] = "ageoa.sklearn.stale.atom"
    orphan_path = reviews_dir / f"{safe_atom_stem(orphan_record['atom_id'])}.json"
    orphan_path.write_text(json.dumps(orphan_record))

    payload = cleanup_orphan_review_records(
        reviews_dir=reviews_dir,
        manifest_path=manifest_path,
        atom_prefix="ageoa.sklearn.",
    )
    assert payload["deleted_count"] == 1
    assert valid_path.exists()
    assert not orphan_path.exists()
