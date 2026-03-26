#!/usr/bin/env python3
"""Run deterministic state-fidelity analysis for the audit manifest."""

from __future__ import annotations

import argparse

from auditlib.io import read_json
from auditlib.paths import AUDIT_MANIFEST_PATH
from auditlib.state_fidelity import write_state_fidelity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom-id", help="Only process one atom id")
    args = parser.parse_args()

    manifest = read_json(AUDIT_MANIFEST_PATH)
    processed = 0
    for record in manifest.get("atoms", []):
        if args.atom_id and record["atom_id"] != args.atom_id:
            continue
        write_state_fidelity(record)
        processed += 1
    print(f"Wrote state-fidelity evidence for {processed} atom(s)")


if __name__ == "__main__":
    main()
