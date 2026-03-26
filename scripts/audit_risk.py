#!/usr/bin/env python3
"""Run deterministic Phase 3 risk triage."""

from __future__ import annotations

from auditlib.io import read_json
from auditlib.paths import AUDIT_MANIFEST_PATH
from auditlib.risk import run_risk_triage


def main() -> None:
    manifest = read_json(AUDIT_MANIFEST_PATH)
    updated_manifest, report = run_risk_triage(manifest)
    print(
        "Risk triage complete:",
        f"atoms={len(updated_manifest.get('atoms', []))}",
        f"high={report['summary']['high_risk_count']}",
        f"medium={report['summary']['medium_risk_count']}",
        f"low={report['summary']['low_risk_count']}",
    )


if __name__ == "__main__":
    main()
