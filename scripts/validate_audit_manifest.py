#!/usr/bin/env python3
"""Validate the deterministic audit manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from auditlib.io import read_json
from auditlib.manifest_validation import validate_manifest, write_validation_report
from auditlib.paths import AUDIT_MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(AUDIT_MANIFEST_PATH))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = read_json(manifest_path)
    report = validate_manifest(payload)
    write_validation_report(report)
    summary = report["summary"]
    print(
        "Manifest validation:",
        "ok" if report["ok"] else "failed",
        f"(errors={report['summary']['error_count']}, warnings={report['summary']['warning_count']})",
        f"(heuristic_atoms={summary.get('heuristic_atom_count', 0)}, heuristic_errors={summary.get('heuristic_error_count', 0)}, heuristic_warnings={summary.get('heuristic_warning_count', 0)})",
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
