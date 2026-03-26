from __future__ import annotations

from auditlib.inventory import build_manifest
from auditlib.manifest_validation import validate_manifest


def test_validate_manifest_accepts_current_payload() -> None:
    manifest = build_manifest()
    report = validate_manifest(manifest)
    assert report["ok"] is True
    assert report["summary"]["error_count"] == 0


def test_validate_manifest_detects_duplicate_atom_ids() -> None:
    manifest = build_manifest()
    manifest["atoms"][1]["atom_id"] = manifest["atoms"][0]["atom_id"]
    report = validate_manifest(manifest)
    codes = {finding["code"] for finding in report["findings"]}
    assert report["ok"] is False
    assert "MANIFEST_DUPLICATE_ATOM_ID" in codes
