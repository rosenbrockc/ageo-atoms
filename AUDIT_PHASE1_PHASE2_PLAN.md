# Phase 1 And Phase 2 Delivery Plan

This document is the concrete implementation plan for:

- finishing Phase 1 from
  [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md)
- implementing Phase 2 on top of the current audit tooling

It is intentionally specific enough for a planning agent to resume mid-stream
after a restart.

## Current Repo State

Available now:

- manifest builder:
  [build_audit_manifest.py](/Users/conrad/personal/ageo-atoms/scripts/build_audit_manifest.py)
- manifest library:
  [inventory.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/inventory.py)
- current manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- structural inputs:
  [audit.py](/Users/conrad/personal/ageo-atoms/scripts/audit.py),
  [type_and_isomorphism_audit.py](/Users/conrad/personal/ageo-atoms/scripts/type_and_isomorphism_audit.py)

Missing for this milestone:

- manifest validation
- explicit Phase 1 completion criteria in code
- unified structural orchestrator
- normalized structural finding codes
- structural report artifact

## Milestone Definition

This milestone is complete when all of the following are true:

1. Phase 1 manifest is complete, validated, and self-describing.
2. A new structural orchestrator produces a normalized repo-wide report.
3. Every atom row in the manifest has a derived `structural_status`.
4. The structural report can be regenerated from scratch with stable outputs.

## Work Breakdown

### Workstream A: Finish Phase 1 Schema

Owner:

- `scripts/auditlib/inventory.py`
- `scripts/build_audit_manifest.py`

Required changes:

1. Separate manifest sections:
   - `metadata`
   - `summary`
   - `atoms`
   - `inventory_errors`
2. Add missing atom fields:
   - `stochastic`
   - `procedural`
   - `authoritative_sources`
   - `risk_reasons`
   - `status_basis`
3. Convert vague booleans into explicit enums where useful:
   - `source_kind`
   - `stateful_kind`
4. Add manifest summary block:
   - atom count
   - family counts
   - source kind counts
   - risk tier counts
   - unmapped upstream count
5. Keep all fields derivable without importing `ageoa`

Acceptance criteria:

- manifest schema version increments
- manifest contains top-level summary data
- every atom row contains the required Phase 1 fields

### Workstream B: Add Manifest Validation

Owner:

- new file:
  `scripts/auditlib/manifest_validation.py`
- CLI entry:
  `scripts/validate_audit_manifest.py`

Required validation rules:

1. `atom_id` is unique
2. `atom_name` is unique unless multiple line-specific wrappers are intended
3. `module_path` exists
4. `wrapper_line` points to a function definition
5. enum fields contain only allowed values
6. every `atom_key` is stable and non-empty
7. every row has all required fields

Output artifacts:

- `data/audit/manifest_validation.json`

Acceptance criteria:

- validator exits nonzero on schema errors
- validator output is machine-readable
- `build_audit_manifest.py` can optionally invoke validation automatically

### Workstream C: Finish Phase 1 Classification Quality

Owner:

- `scripts/auditlib/inventory.py`

Required changes:

1. improve `source_kind` classification
2. distinguish:
   - hand-written
   - generated ingest
   - refined ingest
   - skeleton
3. add deterministic `stochastic` heuristics:
   - RNG argument names
   - probabilistic abstract types
   - random/nuts/hmc family markers
4. add deterministic `procedural` heuristics:
   - state-machine style APIs
   - orchestration wrappers
   - pipeline/multi-step helpers
5. add `authoritative_sources` stubs from:
   - `scripts/atom_manifest.yml`
   - vendored repo path
   - references.json existence

Acceptance criteria:

- manifest supports the fields described in Phase 1 of `AUDIT_INGEST.md`
- classification logic is testable and deterministic

### Workstream D: Implement Phase 2 Orchestrator

Owner:

- new file:
  `scripts/audit_structural.py`
- new library module:
  `scripts/auditlib/structural.py`

Purpose:

- turn existing structural checks into one normalized per-atom report

Input sources:

1. manifest rows
2. `scripts/audit.py`
3. `scripts/type_and_isomorphism_audit.py`
4. `../ageo-matcher/scripts/verify_atoms_repo.py`

Required outputs:

- `data/audit/structural_report.json`
- `data/audit/structural_findings.csv`

Normalized finding codes:

- `STRUCT_PARSE_FAIL`
- `STRUCT_REGISTER_MISSING`
- `STRUCT_REGISTER_NOT_OUTERMOST`
- `STRUCT_REQUIRE_MISSING`
- `STRUCT_ENSURE_MISSING`
- `STRUCT_WITNESS_FILE_MISSING`
- `STRUCT_WITNESS_PLACEHOLDER`
- `STRUCT_WITNESS_TYPE_MISSING`
- `STRUCT_CDG_MISSING`
- `STRUCT_CDG_INVALID`
- `STRUCT_IMPORT_HEAVY`
- `STRUCT_DOCSTRING_MISSING`
- `STRUCT_DOCSTRING_PLACEHOLDER`
- `STRUCT_PUBLIC_VARARGS`
- `STRUCT_PUBLIC_KWARGS`
- `STRUCT_WEAK_TYPES`
- `STRUCT_STUB_PUBLIC_API`
- `STRUCT_EXPORT_OR_REGISTRY_ISSUE`

Mapping strategy:

1. create adapters for each input tool
2. normalize raw findings into the stable code set
3. attach severity:
   - `error`
   - `warning`
   - `info`
4. attach source:
   - `audit.py`
   - `type_and_isomorphism_audit.py`
   - `verify_atoms_repo.py`
   - `manifest_heuristics`

Acceptance criteria:

- every manifest row has zero or more structural findings
- every finding has a normalized code and source
- report summary counts are stable across reruns

### Workstream E: Integrate Structural Status Into Manifest

Owner:

- `scripts/audit_structural.py`
- `scripts/auditlib/inventory.py`

Required derivation rules:

- `fail` if any structural error code in:
  - parse
  - registration
  - witness integrity
  - CDG validity
  - stub public API
- `partial` if only warning-grade issues remain
- `pass` if no structural findings remain

Required side effects:

- update `structural_status`
- append structural finding codes to `blocking_findings`
- append structural repair recommendations to `required_actions`
- preserve non-structural fields already present in the manifest

Acceptance criteria:

- manifest and structural report agree
- rerunning the structural pass does not drift statuses

## File-Level Plan

Files to create:

- [manifest_validation.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/manifest_validation.py)
- [validate_audit_manifest.py](/Users/conrad/personal/ageo-atoms/scripts/validate_audit_manifest.py)
- [structural.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/structural.py)
- [audit_structural.py](/Users/conrad/personal/ageo-atoms/scripts/audit_structural.py)

Files to modify:

- [inventory.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/inventory.py)
- [build_audit_manifest.py](/Users/conrad/personal/ageo-atoms/scripts/build_audit_manifest.py)
- possibly [acceptability.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/acceptability.py)
  so it consumes structural results instead of duplicating the logic
- [test_audit_tools.py](/Users/conrad/personal/ageo-atoms/tests/test_audit_tools.py)

## Test Plan

Add or update tests for:

1. manifest schema contains new required fields
2. manifest validator catches duplicates or missing fields
3. structural mapping converts raw `audit.py` violations to normalized finding
   codes
4. structural status derivation behaves correctly for:
   - passing atom
   - placeholder witness atom
   - stub atom
5. generated structural report joins correctly against `atom_id`

Recommended test file additions:

- `tests/test_audit_manifest_validation.py`
- `tests/test_audit_structural.py`

## Execution Order

Execute in this order:

1. extend manifest schema in `inventory.py`
2. add manifest validation module and CLI
3. regenerate manifest
4. add structural normalization module
5. add structural CLI
6. update manifest integration
7. add tests
8. regenerate outputs

Do not start runtime probes or review-record work until this milestone passes.

## Idempotent Command Set

This is the command set an agent should rerun after any restart:

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
pytest -q tests/test_audit_tools.py tests/test_audit_manifest_validation.py tests/test_audit_structural.py
```

If any command fails, fix that layer before continuing.

## Exit Criteria

Phase 1 is finished when:

- manifest validation passes
- manifest fields match the intended schema
- summary metadata exists
- every public atom has one validated row

Phase 2 is implemented when:

- `scripts/audit_structural.py` exists
- `data/audit/structural_report.json` is generated
- `data/audit/structural_findings.csv` is generated
- manifest rows include derived structural status from normalized findings

## Agent Handoff Template

An agent stopping mid-work should leave this information in its final note:

- current branch and commit
- completed workstreams
- files created or modified
- commands last run
- command outputs that matter
- remaining failing tests or blockers
- whether `data/audit_manifest.json` and `data/audit/structural_report.json`
  are fresh

Without that handoff, the next agent should assume generated artifacts may be
stale and rerun the idempotent command set above.

