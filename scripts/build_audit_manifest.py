#!/usr/bin/env python3
"""Build the deterministic audit manifest for all registered atoms."""

from __future__ import annotations

import argparse

from auditlib.inventory import write_manifest
from auditlib.manifest_validation import validate_manifest, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    payload = write_manifest()
    if not args.skip_validation:
        report = validate_manifest(payload)
        write_validation_report(report)
    atoms = payload.get("atoms", [])
    errors = payload.get("inventory_errors", [])
    print(f"Wrote data/audit_manifest.json with {len(atoms)} atoms")
    if errors:
        print(f"Inventory errors: {len(errors)}")
    if not args.skip_validation:
        print(
            "Manifest validation:",
            "ok" if report["ok"] else "failed",
            f"(errors={report['summary']['error_count']}, warnings={report['summary']['warning_count']})",
        )
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
