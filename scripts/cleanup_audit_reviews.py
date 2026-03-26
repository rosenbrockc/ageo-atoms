#!/usr/bin/env python3
"""Remove stale review records that point at missing manifest atoms."""

from __future__ import annotations

import argparse

from auditlib.reviews import cleanup_orphan_review_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = cleanup_orphan_review_records(atom_prefix=args.atom_prefix, dry_run=args.dry_run)
    print(
        "Review cleanup:",
        f"deleted={payload['deleted_count']}",
        f"kept={payload['kept_count']}",
        f"invalid={payload['invalid_count']}",
        f"dry_run={payload['dry_run']}",
    )


if __name__ == "__main__":
    main()
