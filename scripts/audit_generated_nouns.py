#!/usr/bin/env python3
"""Run deterministic generated-noun analysis for the audit manifest."""

from __future__ import annotations

import argparse

from auditlib.generated_nouns import write_generated_nouns
from auditlib.io import read_json
from auditlib.paths import AUDIT_MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-id", help="Only process one atom id")
    args = parser.parse_args()

    manifest = read_json(AUDIT_MANIFEST_PATH)
    processed = 0
    for record in manifest.get("atoms", []):
        if args.atom_id and record["atom_id"] != args.atom_id:
            continue
        write_generated_nouns(record)
        processed += 1
    print(f"Wrote generated-noun evidence for {processed} atom(s)")


if __name__ == "__main__":
    main()
