"""Deterministic Phase 6 parity and usage coverage accounting."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, write_json
from .paths import (
    AUDIT_MANIFEST_PATH,
    AUDIT_PARITY_BACKLOG_PATH,
    AUDIT_PARITY_COVERAGE_CSV_PATH,
    AUDIT_PARITY_COVERAGE_PATH,
    AUDIT_PROBES_DIR,
    FIXTURES_DIR,
)

PARITY_LEVEL_ORDER = {
    "none": 0,
    "fixture_only": 1,
    "positive_path": 2,
    "positive_and_negative": 3,
    "parity_or_usage_equivalent": 4,
    "not_applicable": 5,
}

USAGE_TEST_COVERAGE_ORDER = {
    "none": 0,
    "positive_path": 1,
    "positive_and_negative": 2,
    "usage_equivalent": 3,
    "not_applicable": 4,
}

UNSUPPORTED_FFI_FAMILIES = {
    "bayes_rs",
    "rust_robotics",
    "tempo",
    "tempo_jl",
}

POSITIVE_LEVELS = {
    "positive_path",
    "positive_and_negative",
    "parity_or_usage_equivalent",
}

BACKLOG_LEVELS = {
    "none",
    "fixture_only",
    "positive_path",
    "positive_and_negative",
}

PARTIAL_STATUS_LEVELS = {
    "fixture_only",
    "positive_path",
}

PASS_STATUS_LEVELS = {
    "positive_and_negative",
    "parity_or_usage_equivalent",
}


def utc_now() -> str:
    """Return a stable UTC timestamp for generated artifacts."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _case_is_negative(case: dict[str, Any]) -> bool:
    if case.get("valid") is False:
        return True
    if case.get("expect_exception") or case.get("expected_exception"):
        return True
    if any(key in case for key in ("error", "exception")):
        return True
    status = case.get("status")
    return status in {"error", "fail", "invalid"}


def inventory_parity_fixtures(fixtures_dir: Path | None = None) -> dict[str, Any]:
    """Inventory fixture files and map them to explicit atom keys."""
    fixtures_dir = fixtures_dir or FIXTURES_DIR
    atoms: dict[str, dict[str, Any]] = {}
    orphan_fixtures: list[dict[str, Any]] = []
    fixture_file_count = 0
    total_case_count = 0

    if not fixtures_dir.exists():
        return {
            "atoms": {},
            "fixture_file_count": 0,
            "total_case_count": 0,
            "orphan_fixtures": [],
        }

    for fixture_path in sorted(fixtures_dir.rglob("*.json")):
        fixture_file_count += 1
        rel_path = str(fixture_path.relative_to(fixtures_dir))
        payload = read_json(fixture_path)
        if not isinstance(payload, list):
            orphan_fixtures.append(
                {
                    "path": rel_path,
                    "reason": "non_list_fixture_payload",
                }
            )
            continue

        total_case_count += len(payload)
        atom_keys = sorted(
            {
                case.get("atom")
                for case in payload
                if isinstance(case, dict) and isinstance(case.get("atom"), str) and case.get("atom")
            }
        )
        if len(atom_keys) != 1:
            orphan_fixtures.append(
                {
                    "path": rel_path,
                    "reason": "missing_or_ambiguous_atom_key",
                    "case_count": len(payload),
                    "atom_keys": atom_keys,
                }
            )
            continue

        atom_key = atom_keys[0]
        case_count = len(payload)
        negative_case_count = sum(
            1 for case in payload if isinstance(case, dict) and _case_is_negative(case)
        )
        positive_case_count = max(0, case_count - negative_case_count)

        fixture_entry = atoms.setdefault(
            atom_key,
            {
                "atom_key": atom_key,
                "fixture_paths": [],
                "fixture_file_count": 0,
                "case_count": 0,
                "positive_case_count": 0,
                "negative_case_count": 0,
                "empty_fixture_count": 0,
                "has_usage_equivalence": False,
            },
        )
        fixture_entry["fixture_paths"].append(rel_path)
        fixture_entry["fixture_file_count"] += 1
        fixture_entry["case_count"] += case_count
        fixture_entry["positive_case_count"] += positive_case_count
        fixture_entry["negative_case_count"] += negative_case_count
        if case_count == 0:
            fixture_entry["empty_fixture_count"] += 1
        if positive_case_count > 0:
            # Fixtures are captured against upstream outputs, so non-empty parity
            # fixtures count as explicit equivalence evidence.
            fixture_entry["has_usage_equivalence"] = True

    return {
        "atoms": atoms,
        "fixture_file_count": fixture_file_count,
        "total_case_count": total_case_count,
        "orphan_fixtures": orphan_fixtures,
    }


def load_runtime_probe_index(probes_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Index runtime probe artifacts by atom id."""
    probes_dir = probes_dir or AUDIT_PROBES_DIR
    index: dict[str, dict[str, Any]] = {}
    if not probes_dir.exists():
        return index

    for probe_path in sorted(probes_dir.glob("*.json")):
        payload = read_json(probe_path)
        atom_id = payload.get("atom_id")
        if not atom_id:
            continue
        positive_probe = payload.get("positive_probe") or {}
        negative_probe = payload.get("negative_probe") or {}
        index[atom_id] = {
            "atom_id": atom_id,
            "path": str(probe_path),
            "status": payload.get("status", "unknown"),
            "probe_status": payload.get("probe_status", "unknown"),
            "positive_pass": positive_probe.get("status") == "pass",
            "negative_pass": negative_probe.get("status") == "pass",
            "parity_used": bool(payload.get("parity_used")),
            "findings": payload.get("findings", []),
        }
    return index


def _usage_test_coverage(
    *,
    stub_not_applicable: bool,
    ffi_not_applicable: bool,
    probe: dict[str, Any] | None,
) -> str:
    if stub_not_applicable or ffi_not_applicable:
        return "not_applicable"
    if probe is None:
        return "none"
    if probe.get("parity_used"):
        return "usage_equivalent"
    if probe.get("positive_pass") and probe.get("negative_pass"):
        return "positive_and_negative"
    if probe.get("positive_pass"):
        return "positive_path"
    return "none"


def derive_parity_entry(
    record: dict[str, Any],
    fixture_entry: dict[str, Any] | None,
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive deterministic parity coverage for one atom."""
    fixture_entry = fixture_entry or {}
    probe = probe or {}

    fixture_count = int(fixture_entry.get("fixture_file_count", 0))
    case_count = int(fixture_entry.get("case_count", 0))
    empty_fixture_count = int(fixture_entry.get("empty_fixture_count", 0))
    positive_fixture_cases = int(fixture_entry.get("positive_case_count", 0))
    negative_fixture_cases = int(fixture_entry.get("negative_case_count", 0))
    usage_equivalence = bool(fixture_entry.get("has_usage_equivalence")) or bool(probe.get("parity_used"))

    positive_probe = bool(probe.get("positive_pass"))
    negative_probe = bool(probe.get("negative_pass"))
    has_positive_evidence = positive_fixture_cases > 0 or positive_probe
    has_negative_evidence = negative_fixture_cases > 0 or negative_probe

    structural_findings = set(record.get("structural_findings", []))
    stub_not_applicable = (
        record.get("source_kind") == "skeleton"
        or "STRUCT_STUB_PUBLIC_API" in structural_findings
    )
    ffi_not_applicable = (
        not stub_not_applicable
        and record.get("ffi") is True
        and record.get("domain_family") in UNSUPPORTED_FFI_FAMILIES
        and fixture_count == 0
        and not positive_probe
        and probe.get("status") in {"not_applicable", "unknown", None}
    )

    reasons: list[str] = []
    if fixture_count > 0:
        reasons.append("PARITY_FIXTURE_PRESENT")
    if empty_fixture_count > 0:
        reasons.append("PARITY_FIXTURE_EMPTY")
    if has_positive_evidence:
        reasons.append("PARITY_POSITIVE_CASES_PRESENT")
    if has_negative_evidence:
        reasons.append("PARITY_NEGATIVE_CASES_PRESENT")
    if positive_probe:
        reasons.append("PARITY_RUNTIME_PROBE_SUPPORT")
    if usage_equivalence:
        reasons.append("PARITY_USAGE_EQUIVALENCE_PRESENT")

    if stub_not_applicable:
        reasons.append("PARITY_NOT_APPLICABLE_STUB")
        level = "not_applicable"
    elif ffi_not_applicable:
        reasons.append("PARITY_NOT_APPLICABLE_FFI")
        level = "not_applicable"
    elif fixture_count > 0 and case_count == 0 and not positive_probe:
        level = "fixture_only"
    elif usage_equivalence:
        level = "parity_or_usage_equivalent"
    elif has_positive_evidence and has_negative_evidence:
        level = "positive_and_negative"
    elif has_positive_evidence:
        level = "positive_path"
    elif fixture_count > 0:
        level = "fixture_only"
    else:
        level = "none"

    if level != "not_applicable" and fixture_count == 0:
        reasons.append("PARITY_MISSING_FIXTURE")
    if level != "not_applicable" and has_positive_evidence and not has_negative_evidence:
        reasons.append("PARITY_MISSING_NEGATIVE_CASES")
    if level != "not_applicable" and not usage_equivalence:
        reasons.append("PARITY_MISSING_USAGE_EQUIVALENCE")

    usage_test_coverage = _usage_test_coverage(
        stub_not_applicable=stub_not_applicable,
        ffi_not_applicable=ffi_not_applicable,
        probe=probe if probe else None,
    )

    return {
        "atom_id": record["atom_id"],
        "atom_name": record["atom_name"],
        "atom_key": record["atom_key"],
        "domain_family": record["domain_family"],
        "risk_tier": record.get("risk_tier", "unknown"),
        "review_priority": record.get("review_priority", "unknown"),
        "parity_coverage_level": level,
        "parity_coverage_reasons": sorted(set(reasons)),
        "parity_fixture_count": fixture_count,
        "parity_case_count": case_count,
        "usage_test_coverage": usage_test_coverage,
        "runtime_probe_status": probe.get("status", record.get("runtime_status", "unknown")),
        "stateful": bool(record.get("stateful")),
        "ffi": bool(record.get("ffi")),
        "fixture_paths": fixture_entry.get("fixture_paths", []),
    }


def _coverage_sort_key(level: str) -> int:
    return PARITY_LEVEL_ORDER.get(level, len(PARITY_LEVEL_ORDER))


def _recommended_action(entry: dict[str, Any]) -> str:
    reasons = set(entry.get("parity_coverage_reasons", []))
    if "PARITY_NOT_APPLICABLE_STUB" in reasons:
        return "no parity action until stub is implemented"
    if "PARITY_NOT_APPLICABLE_FFI" in reasons:
        return "document safe FFI parity strategy before adding fixtures"
    if "PARITY_MISSING_FIXTURE" in reasons:
        return "capture an explicit upstream parity fixture"
    if "PARITY_MISSING_USAGE_EQUIVALENCE" in reasons:
        return "add upstream-equivalent fixture or explicit usage probe"
    if "PARITY_MISSING_NEGATIVE_CASES" in reasons:
        return "add a deterministic negative-contract probe"
    return "review current parity evidence"


def _parity_test_status(level: str) -> str:
    if level in PASS_STATUS_LEVELS:
        return "pass"
    if level in PARTIAL_STATUS_LEVELS:
        return "partial"
    if level == "not_applicable":
        return "not_applicable"
    return "unknown"


def write_parity_csv(rows: list[dict[str, Any]], output_path: Path | None = None) -> None:
    """Write the Phase 6 parity CSV view."""
    output_path = output_path or AUDIT_PARITY_COVERAGE_CSV_PATH
    fieldnames = [
        "atom_id",
        "atom_name",
        "domain_family",
        "parity_coverage_level",
        "parity_fixture_count",
        "parity_case_count",
        "runtime_probe_status",
        "risk_tier",
        "review_priority",
        "parity_coverage_reasons",
    ]
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "atom_id": row["atom_id"],
                    "atom_name": row["atom_name"],
                    "domain_family": row["domain_family"],
                    "parity_coverage_level": row["parity_coverage_level"],
                    "parity_fixture_count": row["parity_fixture_count"],
                    "parity_case_count": row["parity_case_count"],
                    "runtime_probe_status": row["runtime_probe_status"],
                    "risk_tier": row["risk_tier"],
                    "review_priority": row["review_priority"],
                    "parity_coverage_reasons": ";".join(row["parity_coverage_reasons"]),
                }
            )


def build_parity_backlog(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Group missing parity work into a durable backlog artifact."""
    missing = [entry for entry in entries if entry["parity_coverage_level"] in BACKLOG_LEVELS]
    high_priority_missing = [
        entry for entry in missing if entry["review_priority"] == "review_now" or entry["risk_tier"] == "high"
    ]
    medium_priority_missing = [
        entry
        for entry in missing
        if entry not in high_priority_missing
        and (entry["review_priority"] == "review_soon" or entry["risk_tier"] == "medium")
    ]
    low_priority_missing = [
        entry for entry in missing if entry not in high_priority_missing and entry not in medium_priority_missing
    ]
    not_applicable = [entry for entry in entries if entry["parity_coverage_level"] == "not_applicable"]

    def _serialize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "atom_id": row["atom_id"],
                "atom_name": row["atom_name"],
                "domain_family": row["domain_family"],
                "risk_tier": row["risk_tier"],
                "review_priority": row["review_priority"],
                "parity_coverage_level": row["parity_coverage_level"],
                "parity_coverage_reasons": row["parity_coverage_reasons"],
                "recommended_action": _recommended_action(row),
            }
            for row in rows
        ]

    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "summary": {
            "atom_count": len(entries),
            "missing_count": len(missing),
            "high_priority_missing_count": len(high_priority_missing),
            "medium_priority_missing_count": len(medium_priority_missing),
            "low_priority_missing_count": len(low_priority_missing),
            "not_applicable_count": len(not_applicable),
        },
        "high_priority_missing": _serialize(high_priority_missing),
        "medium_priority_missing": _serialize(medium_priority_missing),
        "low_priority_missing": _serialize(low_priority_missing),
        "not_applicable": _serialize(not_applicable),
    }


def run_parity_coverage(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Update the manifest with Phase 6 parity coverage and write report artifacts."""
    fixture_inventory = inventory_parity_fixtures()
    probe_index = load_runtime_probe_index()

    entries: list[dict[str, Any]] = []
    updated_atoms: list[dict[str, Any]] = []
    by_level: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for record in manifest.get("atoms", []):
        entry = derive_parity_entry(
            record=record,
            fixture_entry=fixture_inventory["atoms"].get(record["atom_key"]),
            probe=probe_index.get(record["atom_id"]),
        )
        record["parity_coverage_level"] = entry["parity_coverage_level"]
        record["parity_coverage_reasons"] = entry["parity_coverage_reasons"]
        record["parity_fixture_count"] = entry["parity_fixture_count"]
        record["parity_case_count"] = entry["parity_case_count"]
        record["usage_test_coverage"] = entry["usage_test_coverage"]
        record["has_parity_tests"] = entry["parity_coverage_level"] in POSITIVE_LEVELS
        record["parity_test_status"] = _parity_test_status(entry["parity_coverage_level"])
        record.setdefault("status_basis", {})
        record["status_basis"]["parity"] = [
            "fixtures",
            "runtime_probes",
            "manifest",
        ]
        updated_atoms.append(record)
        entries.append(entry)
        by_level[entry["parity_coverage_level"]] += 1
        by_family[entry["domain_family"]]["atom_count"] += 1
        by_family[entry["domain_family"]][entry["parity_coverage_level"]] += 1
        if entry["parity_coverage_level"] in {"none", "fixture_only"}:
            by_family[entry["domain_family"]]["lacking_any_parity_evidence"] += 1
        for reason in entry["parity_coverage_reasons"]:
            by_reason[reason] += 1

    entries.sort(
        key=lambda row: (
            _coverage_sort_key(row["parity_coverage_level"]),
            -row["parity_fixture_count"],
            row["atom_id"],
        )
    )
    manifest["atoms"] = updated_atoms
    summary = manifest.get("summary", {})
    summary["parity_coverage_count"] = sum(1 for entry in entries if entry["parity_coverage_level"] in POSITIVE_LEVELS)
    summary["parity_coverage_level_counts"] = dict(sorted(by_level.items()))
    summary["parity_fixture_file_count"] = fixture_inventory["fixture_file_count"]
    summary["parity_fixture_case_count"] = fixture_inventory["total_case_count"]
    summary["families_missing_parity_evidence"] = {
        family: counts["lacking_any_parity_evidence"]
        for family, counts in sorted(by_family.items())
        if counts["lacking_any_parity_evidence"] > 0
    }
    manifest["summary"] = summary

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "summary": {
            "atom_count": len(entries),
            "fixture_file_count": fixture_inventory["fixture_file_count"],
            "total_fixture_case_count": fixture_inventory["total_case_count"],
            "family_counts_lacking_any_parity_evidence": summary["families_missing_parity_evidence"],
            "orphan_fixture_count": len(fixture_inventory["orphan_fixtures"]),
        },
        "by_level": dict(sorted(by_level.items())),
        "by_family": {
            family: {
                "atom_count": counts["atom_count"],
                "lacking_any_parity_evidence": counts["lacking_any_parity_evidence"],
                "levels": {
                    level: counts[level]
                    for level in PARITY_LEVEL_ORDER
                    if counts[level] > 0
                },
            }
            for family, counts in sorted(by_family.items())
        },
        "by_reason": dict(by_reason.most_common()),
        "fixture_inventory": {
            "fixture_file_count": fixture_inventory["fixture_file_count"],
            "total_case_count": fixture_inventory["total_case_count"],
            "orphan_fixtures": fixture_inventory["orphan_fixtures"],
        },
        "atoms": entries,
    }
    backlog = build_parity_backlog(entries)

    write_json(AUDIT_PARITY_COVERAGE_PATH, report)
    write_parity_csv(entries)
    write_json(AUDIT_PARITY_BACKLOG_PATH, backlog)
    write_json(AUDIT_MANIFEST_PATH, manifest)
    return manifest, report, backlog
