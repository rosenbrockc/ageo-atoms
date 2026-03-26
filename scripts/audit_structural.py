#!/usr/bin/env python3
"""Run the unified structural audit and update the manifest."""

from __future__ import annotations

import argparse

from auditlib.io import read_json, write_json
from auditlib.paths import AUDIT_MANIFEST_PATH
from auditlib.structural import build_structural_report, integrate_structural_results, write_structural_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    manifest = read_json(AUDIT_MANIFEST_PATH)
    report = build_structural_report(manifest, include_verify=not args.skip_verify)
    write_structural_report(report)
    manifest = integrate_structural_results(manifest, report)
    write_json(AUDIT_MANIFEST_PATH, manifest)
    print(
        f"Wrote structural report with {report['summary']['finding_count']} finding(s); "
        f"atoms with findings={report['summary']['atoms_with_findings']}"
    )


if __name__ == "__main__":
    main()
