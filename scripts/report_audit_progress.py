#!/usr/bin/env python3
"""Build Phase 7 review progress and backlog artifacts."""

from __future__ import annotations

from auditlib.review_pass import (
    build_review_pass,
    load_validated_reviews,
    refresh_review_index,
    write_review_progress,
    write_validation_payload,
)


def main() -> None:
    valid_reviews, _, errors, warnings = load_validated_reviews()
    write_validation_payload(errors, warnings, valid_reviews)
    refresh_review_index(valid_reviews)
    _, progress, backlog_rows, _ = build_review_pass()
    write_review_progress(progress, backlog_rows)
    summary = progress["summary"]
    print(
        "Review progress:",
        f"atoms={summary['atom_count']}",
        f"completed={summary['completed_review_count']}",
        f"draft={summary['draft_review_count']}",
        f"missing={summary['missing_review_count']}",
        f"trust_ready={summary['trust_ready_count']}",
    )


if __name__ == "__main__":
    main()
