#!/usr/bin/env python3
"""Aggregate Phase 5 semantic evidence back into the manifest."""

from __future__ import annotations

from auditlib.io import read_json, write_json
from auditlib.paths import AUDIT_MANIFEST_PATH, AUDIT_STRUCTURAL_REPORT_PATH
from auditlib.semantics import build_semantic_report, write_semantic_report


def main() -> None:
    manifest = read_json(AUDIT_MANIFEST_PATH)
    structural_report = read_json(AUDIT_STRUCTURAL_REPORT_PATH) if AUDIT_STRUCTURAL_REPORT_PATH.exists() else {}
    manifest, report = build_semantic_report(manifest, structural_report)
    write_json(AUDIT_MANIFEST_PATH, manifest)
    write_semantic_report(report)
    print(
        f"Wrote semantic report for {report['summary']['atom_count']} atom(s); "
        f"runtime_pass={report['summary']['runtime_status_counts'].get('pass', 0)}"
    )


if __name__ == "__main__":
    main()
