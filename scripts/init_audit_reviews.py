#!/usr/bin/env python3
"""Seed draft semantic review records from the manifest and review queue."""

from __future__ import annotations

import argparse

from auditlib.reviews import REVIEW_PRIORITY_VALUES, seed_review_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-id")
    parser.add_argument("--priority", choices=sorted(REVIEW_PRIORITY_VALUES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = seed_review_records(
        atom_id=args.atom_id,
        priority=args.priority,
        limit=args.limit,
        force=args.force,
    )
    print(
        "Seeded review drafts: "
        f"{result['seeded_count']} seeded, {result['skipped_count']} skipped, {result['missing_count']} missing"
    )


if __name__ == "__main__":
    main()
