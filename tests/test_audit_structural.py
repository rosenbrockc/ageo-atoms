from __future__ import annotations

from auditlib.inventory import build_manifest
from auditlib.structural import build_structural_report, integrate_structural_results


def _subset_manifest(manifest: dict, atom_keys: set[str]) -> dict:
    rows = [row for row in manifest["atoms"] if row["atom_key"] in atom_keys]
    return {
        "schema_version": manifest["schema_version"],
        "metadata": manifest["metadata"],
        "summary": manifest["summary"].copy(),
        "atoms": rows,
        "inventory_errors": [],
    }


def _record_for(manifest: dict, atom_key: str) -> dict:
    for record in manifest["atoms"]:
        if record["atom_key"] == atom_key:
            return record
    raise AssertionError(atom_key)


def test_structural_report_flags_known_stubbed_atom() -> None:
    manifest = build_manifest()
    subset = _subset_manifest(manifest, {"tempo:offset_tt2tdb"})
    report = build_structural_report(subset, include_verify=False)
    codes = {finding["code"] for finding in report["findings"]}
    assert "STRUCT_STUB_PUBLIC_API" in codes


def test_structural_integration_updates_manifest_status() -> None:
    manifest = build_manifest()
    subset = _subset_manifest(manifest, {"tempo:offset_tt2tdb", "algorithms/graph:bfs"})
    report = build_structural_report(subset, include_verify=False)
    integrated = integrate_structural_results(subset, report)
    tempo_record = _record_for(integrated, "tempo:offset_tt2tdb")
    bfs_record = _record_for(integrated, "algorithms/graph:bfs")
    assert tempo_record["structural_status"] == "fail"
    assert "STRUCT_STUB_PUBLIC_API" in tempo_record["structural_findings"]
    assert "STRUCT_WITNESS_FILE_MISSING" not in bfs_record["structural_findings"]
