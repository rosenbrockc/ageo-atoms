#!/usr/bin/env python3
"""Build deterministic signature fidelity evidence for the audit manifest."""

from __future__ import annotations

import argparse

from auditlib.fidelity import write_signature_evidence
from auditlib.io import read_json
from auditlib.paths import AUDIT_MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-id", help="Only process one atom id")
    args = parser.parse_args()

    manifest = read_json(AUDIT_MANIFEST_PATH)
    atoms = manifest.get("atoms", [])
    processed = 0
    for record in atoms:
        if args.atom_id and record["atom_id"] != args.atom_id:
            continue
        write_signature_evidence(record)
        processed += 1
    print(f"Wrote signature evidence for {processed} atom(s)")


if __name__ == "__main__":
    main()
