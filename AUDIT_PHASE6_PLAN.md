# Phase 6 Parity And Usage Tests Plan

This document is the concrete implementation plan for Phase 6 from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md):
parity and usage tests.

It is written to survive process restarts. A planning or coding agent should be
able to resume from this file plus the existing audit artifacts without relying
on prior chat context.

## Goal

Turn parity and usage evidence into a measurable, durable coverage layer for the
audit system.

Phase 6 should:

- inventory existing parity fixtures and tests
- map parity coverage back to `atom_id`
- classify coverage quality per atom
- generate durable parity coverage reports
- expose a backlog of missing parity evidence

Phase 6 is about coverage accounting and safe expansion of parity evidence. It
is not yet about CI gating or semantic review decisions.

## Current Baseline

Available inputs:

- manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- semantic report:
  [semantic_report.json](/Users/conrad/personal/ageo-atoms/data/audit/semantic_report.json)
- risk report:
  [risk_report.json](/Users/conrad/personal/ageo-atoms/data/audit/risk_report.json)
- review queue:
  [review_queue.csv](/Users/conrad/personal/ageo-atoms/data/audit/review_queue.csv)
- fixtures:
  [tests/fixtures](/Users/conrad/personal/ageo-atoms/tests/fixtures)
- parity runner:
  [test_parity.py](/Users/conrad/personal/ageo-atoms/tests/test_parity.py)
- parity generator:
  [generate_parity_tests.py](/Users/conrad/personal/ageo-atoms/scripts/generate_parity_tests.py)
- runtime probe artifacts:
  [data/audit/probes](/Users/conrad/personal/ageo-atoms/data/audit/probes)

Current limitation:

- the manifest only tracks `has_parity_tests` / coarse `parity_test_status`
- there is no repo-wide parity coverage report
- fixture presence is not normalized into a coverage level
- there is no backlog artifact for missing parity evidence

## Deliverables

Required code:

- `scripts/auditlib/parity.py`
- `scripts/report_parity_coverage.py`
- optionally `scripts/seed_parity_backlog.py`

Files likely to modify:

- `scripts/generate_parity_tests.py`
- `scripts/auditlib/models.py`
- `scripts/auditlib/paths.py`
- possibly `scripts/audit_risk.py`
  if parity coverage should affect risk after Phase 6 reports land

Required generated artifacts:

- `data/audit/parity_coverage.json`
- `data/audit/parity_coverage.csv`
- `data/audit/parity_backlog.json`
- updated `data/audit_manifest.json`

Optional generated artifacts:

- family-specific backlog CSVs
- summary markdown for humans

## Coverage Model

Phase 6 should classify parity evidence per atom using an explicit coverage
level, not just a boolean.

Required per-atom fields:

- `parity_coverage_level`
- `parity_coverage_reasons`
- `parity_fixture_count`
- `parity_case_count`
- `usage_test_coverage`

Suggested `parity_coverage_level` enum:

- `none`
- `fixture_only`
- `positive_path`
- `positive_and_negative`
- `parity_or_usage_equivalent`
- `not_applicable`

Definitions:

- `none`: no parity fixtures or usage-equivalence evidence
- `fixture_only`: fixture exists but no usable execution signal yet
- `positive_path`: parity fixture/test covers a basic valid-path execution
- `positive_and_negative`: positive path plus contract violation coverage
- `parity_or_usage_equivalent`: strong parity or usage-equivalence evidence
- `not_applicable`: parity is not meaningful for this atom family in current
  scope and should be explicitly justified

## Evidence Sources

Use only persisted or deterministic local evidence.

Primary sources:

1. `tests/fixtures/**/*.json`
2. `tests/test_parity.py`
3. generated `tests/test_parity_<domain>.py`
4. runtime probe artifacts
5. manifest facts

Do not infer parity coverage from unrelated unit tests unless the mapping is
explicit.

## Required Reason Codes

Use stable coverage reason codes:

- `PARITY_FIXTURE_PRESENT`
- `PARITY_FIXTURE_EMPTY`
- `PARITY_POSITIVE_CASES_PRESENT`
- `PARITY_NEGATIVE_CASES_PRESENT`
- `PARITY_USAGE_EQUIVALENCE_PRESENT`
- `PARITY_RUNTIME_PROBE_SUPPORT`
- `PARITY_NOT_APPLICABLE_FFI`
- `PARITY_NOT_APPLICABLE_STUB`
- `PARITY_MISSING_FIXTURE`
- `PARITY_MISSING_NEGATIVE_CASES`
- `PARITY_MISSING_USAGE_EQUIVALENCE`

## Coverage Rules

### Fixture Inventory

For each fixture file:

- parse JSON cases
- extract the atom key
- count cases
- classify whether the file is:
  - empty
  - positive-only
  - mixed

If negative cases cannot be reliably inferred from current fixture structure,
record that explicitly and keep the model conservative.

### Positive Coverage

An atom gets at least `positive_path` if:

- one or more non-empty parity fixtures exist for it, or
- a runtime probe passed and the atom is in a parity-eligible family

### Negative Coverage

An atom gets `positive_and_negative` only if:

- explicit negative parity/usage evidence exists, or
- a negative contract probe passed and there is also positive-path evidence

Initially, runtime probe negative-contract passes may be used as negative
evidence, but they should be labeled separately in the reasons.

### Usage-Equivalence Coverage

An atom gets `parity_or_usage_equivalent` if:

- parity fixtures exist and compare against upstream outputs, or
- the audit system has explicit usage-equivalence evidence for a stateful API

Phase 6 v1 may mostly populate this level from existing parity fixtures.

### Not Applicable

Use `not_applicable` sparingly.

Allowed early reasons:

- stub/not implemented public atom
- currently unsupported FFI family where parity fixtures are unsafe or absent
- explicitly documented “approximate atom” category once that exists in the
  manifest/review record

## Coverage Report Schema

`data/audit/parity_coverage.json` should contain:

- `schema_version`
- `generated_at`
- `summary`
- `by_level`
- `by_family`
- `by_reason`
- `atoms`

`summary` should include:

- atom count
- counts by coverage level
- fixture file count
- total fixture case count
- family counts lacking any parity evidence

Each per-atom entry should contain:

- `atom_id`
- `atom_name`
- `parity_coverage_level`
- `parity_coverage_reasons`
- `parity_fixture_count`
- `parity_case_count`
- `runtime_probe_status`
- `stateful`
- `ffi`

## CSV Output

`data/audit/parity_coverage.csv` should contain:

- `atom_id`
- `atom_name`
- `domain_family`
- `parity_coverage_level`
- `parity_fixture_count`
- `parity_case_count`
- `runtime_probe_status`
- `risk_tier`
- `review_priority`
- `parity_coverage_reasons`

Sort by:

1. coverage level
2. descending fixture count
3. `atom_id`

## Backlog Artifact

`data/audit/parity_backlog.json` should identify where parity work is missing.

Suggested grouping:

- `high_priority_missing`
- `medium_priority_missing`
- `low_priority_missing`

High priority should include:

- `review_now` atoms with `parity_coverage_level=none`
- stateful atoms with no usage-equivalence evidence
- high-risk atoms lacking any positive-path parity evidence

Each backlog entry should include:

- `atom_id`
- `atom_name`
- `risk_tier`
- `review_priority`
- `stateful`
- `ffi`
- `recommended_parity_action`

## Manifest Integration Rules

After Phase 6 reporting, each manifest row should be updated with:

- `parity_coverage_level`
- `parity_coverage_reasons`
- `parity_fixture_count`
- `parity_case_count`
- `usage_test_coverage`
- refined `parity_test_status`

Suggested `parity_test_status` mapping:

- `pass` for `positive_and_negative` or `parity_or_usage_equivalent`
- `partial` for `fixture_only` or `positive_path`
- `unknown` for `none`
- `not_applicable` for `not_applicable`

## Risk Integration

Phase 6 should not directly change risk scoring logic, but after parity coverage
is updated, rerunning [audit_risk.py](/Users/conrad/personal/ageo-atoms/scripts/audit_risk.py)
should naturally reduce some `RISK_MISSING_PARITY` cases.

Recommended sequence:

1. generate parity coverage artifacts
2. update manifest
3. rerun `python scripts/audit_risk.py`

## Test Plan

Add focused tests for:

1. fixture inventory and case counting
2. atom-key to `atom_id` mapping
3. coverage-level derivation from fixture/runtime evidence
4. backlog generation
5. manifest integration

Recommended test file:

- `tests/test_audit_parity.py`

Prefer compact synthetic fixture trees and temporary manifests over depending on
the full repo state when possible.

## Execution Order

Implement in this order:

1. add path constants for parity coverage outputs
2. implement parity inventory helpers in `scripts/auditlib/parity.py`
3. implement coverage-level derivation
4. implement report/backlog builders
5. implement CLI `scripts/report_parity_coverage.py`
6. update manifest with parity coverage fields
7. add tests
8. regenerate parity coverage artifacts
9. rerun risk triage

## Idempotent Command Set

An agent resuming this phase should rerun:

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_semantics.py
python scripts/audit_acceptability.py
python scripts/report_parity_coverage.py
python scripts/audit_risk.py
pytest -q tests/test_audit_parity.py
```

If a prerequisite artifact is stale or missing, regenerate it first.

## Exit Criteria

Phase 6 is complete when:

1. parity coverage is measurable per atom
2. `data/audit/parity_coverage.json` exists
3. `data/audit/parity_coverage.csv` exists
4. `data/audit/parity_backlog.json` exists
5. manifest rows include parity coverage fields
6. targeted tests pass

## Known Non-Goals

This phase does not:

- add broad new parity fixtures for every family
- replace runtime probes
- complete human review records
- enforce CI gating

Those belong to later phases.

## Agent Handoff Template

If an implementing agent stops mid-phase, it should leave:

- current commit and branch
- files created or modified
- whether parity coverage artifacts are fresh
- whether risk was rerun after parity updates
- tests run
- current blockers

If no handoff is available, the next agent should assume artifacts may be stale
and rerun the idempotent command set.

