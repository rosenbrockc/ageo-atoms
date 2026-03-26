#!/usr/bin/env python3
"""Run deterministic Phase 6 parity coverage accounting."""

from __future__ import annotations

from auditlib.io import read_json
from auditlib.parity import run_parity_coverage
from auditlib.paths import AUDIT_MANIFEST_PATH


def main() -> None:
    manifest = read_json(AUDIT_MANIFEST_PATH)
    updated_manifest, report, backlog = run_parity_coverage(manifest)
    summary = report["summary"]
    print(
        "Parity coverage complete:",
        f"atoms={len(updated_manifest.get('atoms', []))}",
        f"fixture_files={summary['fixture_file_count']}",
        f"fixture_cases={summary['total_fixture_case_count']}",
        f"missing={backlog['summary']['missing_count']}",
    )


if __name__ == "__main__":
    main()
