"""Repository-local path helpers for audit tooling."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGEOA_DIR = ROOT / "ageoa"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"
THIRD_PARTY_DIR = ROOT / "third_party"
DATA_DIR = ROOT / "data"
AUDIT_DIR = DATA_DIR / "audit"
AUDIT_EVIDENCE_DIR = AUDIT_DIR / "evidence"
AUDIT_RESULTS_DIR = AUDIT_DIR / "results"
AUDIT_PROBES_DIR = AUDIT_DIR / "probes"
AUDIT_MANIFEST_PATH = ROOT / "data" / "audit_manifest.json"
AUDIT_SCORES_PATH = ROOT / "data" / "audit_scores.csv"
AUDIT_MANIFEST_VALIDATION_PATH = AUDIT_DIR / "manifest_validation.json"
AUDIT_STRUCTURAL_REPORT_PATH = AUDIT_DIR / "structural_report.json"
AUDIT_STRUCTURAL_FINDINGS_CSV_PATH = AUDIT_DIR / "structural_findings.csv"
ATOM_MANIFEST_PATH = SCRIPTS_DIR / "atom_manifest.yml"
