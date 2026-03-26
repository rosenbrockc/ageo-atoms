#!/usr/bin/env python3
"""Sync validated review records into the audit manifest."""

from __future__ import annotations

import argparse

from auditlib.io import read_json, write_json
from auditlib.paths import AUDIT_MANIFEST_PATH
from auditlib.review_pass import (
    apply_review_state_to_manifest,
    load_validated_reviews,
    refresh_review_index,
    resolve_active_reviews,
    write_validation_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-id", help="Sync only one atom_id", default=None)
    parser.add_argument(
        "--only-completed",
        action="store_true",
        help="Sync only completed reviews into the manifest.",
    )
    parser.add_argument(
        "--fail-on-conflict",
        action="store_true",
        help="Exit on conflicting active review records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_reviews, _, errors, warnings = load_validated_reviews()
    write_validation_payload(errors, warnings, valid_reviews)
    refresh_review_index(valid_reviews)
    manifest = read_json(AUDIT_MANIFEST_PATH)
    resolved_reviews, _ = resolve_active_reviews(valid_reviews)

    if args.atom_id:
        resolved_reviews = {
            atom_id: review
            for atom_id, review in resolved_reviews.items()
            if atom_id == args.atom_id
        }

    updated_manifest, summary = apply_review_state_to_manifest(
        manifest,
        resolved_reviews,
        only_completed=args.only_completed,
        fail_on_conflict=args.fail_on_conflict,
    )
    write_json(AUDIT_MANIFEST_PATH, updated_manifest)
    print(
        "Review sync:",
        f"synced={summary['synced_count']}",
        f"skipped={summary['skipped_count']}",
        f"conflicts={summary['conflict_count']}",
    )


if __name__ == "__main__":
    main()
