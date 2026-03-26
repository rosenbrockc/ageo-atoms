# Phase 4 Semantic Review Plan

This document is the concrete implementation plan for Phase 4 from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md):
the semantic review rubric and durable review-record workflow.

It is written to survive process restarts. A planning or coding agent should be
able to resume from this file plus the existing audit artifacts without relying
on prior chat context.

## Goal

Create a durable, validated review-record system for per-atom semantic review
that:

- records authoritative sources and pinned provenance
- captures structured reviewer judgments
- can be joined back to `data/audit_manifest.json`
- can support both human and model reviewers
- does not grant `trusted` automatically

Phase 4 is about review records and validation, not about running the reviews at
scale yet. That comes in Phase 7.

## Current Baseline

Available inputs:

- manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- structural report:
  [structural_report.json](/Users/conrad/personal/ageo-atoms/data/audit/structural_report.json)
- risk report:
  [risk_report.json](/Users/conrad/personal/ageo-atoms/data/audit/risk_report.json)
- review queue:
  [review_queue.csv](/Users/conrad/personal/ageo-atoms/data/audit/review_queue.csv)
- signature evidence:
  [data/audit/evidence/](/Users/conrad/personal/ageo-atoms/data/audit/evidence)
- per-atom acceptability results:
  [data/audit/results/](/Users/conrad/personal/ageo-atoms/data/audit/results)

Current limitation:

- there is no review schema
- there is no review template
- there is no validator for review records
- there is no join/update flow from review records back into the manifest

## Deliverables

Required code:

- `scripts/auditlib/reviews.py`
- `scripts/init_audit_reviews.py`
- `scripts/validate_audit_reviews.py`
- optionally `scripts/report_audit_reviews.py`

Files likely to modify:

- `scripts/auditlib/models.py`
- `scripts/auditlib/paths.py`
- possibly `scripts/auditlib/risk.py`
  only if shared helpers are useful for selecting seed review atoms

Required generated artifacts:

- `data/audit_reviews/`
- `data/audit/review_validation.json`
- optionally `data/audit/review_index.json`

## Review Record Format

Use JSON as the canonical review format. Markdown may be generated later for
humans, but JSON should be the validation target.

Canonical location:

- `data/audit_reviews/<safe_atom_stem>.json`

One review record per atom.

## Required Review Schema

Each review record must contain:

- `schema_version`
- `atom_id`
- `atom_name`
- `review_status`
- `reviewer_type`
- `reviewed_at`
- `upstream_symbols`
- `authoritative_sources`
- `source_basis`
- `line_references`
- `wrapper_truth`
- `api_truth`
- `state_truth`
- `output_truth`
- `decomposition_truth`
- `semantic_verdict`
- `developer_semantics_verdict`
- `limitations`
- `required_actions`
- `notes`

### Enum Fields

Allowed `review_status`:

- `draft`
- `completed`
- `superseded`

Allowed `reviewer_type`:

- `human`
- `model`
- `human_verified_model`

Allowed verdict values for:

- `wrapper_truth`
- `api_truth`
- `state_truth`
- `output_truth`
- `decomposition_truth`
- `semantic_verdict`
- `developer_semantics_verdict`

Allowed values:

- `pass`
- `partial`
- `fail`
- `not_applicable`
- `unknown`

### Source Basis

`source_basis` must capture the provenance that makes the review reproducible.

Required fields:

- `upstream_version`
- `source_revision`
- `review_basis_at`

At least one of `upstream_version` or `source_revision` must be populated.

### Authoritative Sources

Each entry in `authoritative_sources` must contain:

- `kind`
- `label`
- `reference`
- `relevance`

Allowed `kind` examples:

- `local_wrapper`
- `vendored_source`
- `official_docs`
- `repository`
- `paper`

### Line References

Each entry in `line_references` must contain:

- `scope`
- `path`
- `line`
- `note`

Allowed `scope` values:

- `wrapper`
- `vendored_source`
- `official_docs`

## Review Workflow

Phase 4 should implement the workflow skeleton, not full review throughput.

Workflow:

1. seed a review file from existing manifest/evidence
2. let a reviewer fill in semantic judgments
3. validate the review record
4. build a review index for downstream consumers
5. optionally sync review verdicts back into the manifest

## CLI Requirements

### `scripts/init_audit_reviews.py`

Purpose:

- initialize draft review records from existing evidence

Required behavior:

- accept `--atom-id`
- accept `--priority review_now|review_soon|review_later`
- accept `--limit N`
- if no specific atom is given, seed from `data/audit/review_queue.csv`
- do not overwrite completed review records unless `--force` is given

Output:

- draft review JSON files under `data/audit_reviews/`

### `scripts/validate_audit_reviews.py`

Purpose:

- validate existing review records

Required behavior:

- scan all review JSON files
- validate schema and enums
- validate `atom_id` maps to the manifest
- validate source-basis requirements
- validate required sections are present when `review_status=completed`
- emit machine-readable results
- exit nonzero if validation errors exist

Output:

- `data/audit/review_validation.json`

### Optional `scripts/report_audit_reviews.py`

Purpose:

- summarize review coverage and backlog

Output:

- counts by review status
- counts by reviewer type
- counts by semantic verdict
- list of missing completed reviews for `review_now`

## Review Seeding Rules

When initializing a draft review:

1. copy identity fields from the manifest
2. copy `upstream_symbols`
3. copy `authoritative_sources` from the manifest when present
4. copy `review_basis_at`, `upstream_version`, and `source_revision`
5. include placeholders for the judgment sections
6. include links or paths to supporting artifacts:
   - structural findings
   - fidelity evidence
   - risk reasons
   - acceptability results

This should reduce reviewer setup work and improve consistency.

## Validation Rules

Validation must be strict enough that later phases can trust the records.

Required checks:

1. review file is valid JSON
2. required top-level keys exist
3. enum fields use allowed values
4. `atom_id` exists in the manifest
5. `atom_name` matches the manifest row
6. `source_basis` includes reproducible provenance
7. completed reviews have non-empty:
   - `authoritative_sources`
   - `line_references`
   - `semantic_verdict`
   - `developer_semantics_verdict`
8. completed reviews with `fail` or `partial` verdicts must include either
   `limitations` or `required_actions`

## Review Index

Phase 4 should also build a normalized index of reviews so later phases do not
need to parse individual files repeatedly.

Suggested artifact:

- `data/audit/review_index.json`

Suggested contents:

- `schema_version`
- `generated_at`
- `summary`
- `reviews`

Each `reviews` entry should contain:

- `atom_id`
- `review_status`
- `reviewer_type`
- `semantic_verdict`
- `developer_semantics_verdict`
- `reviewed_at`
- `required_actions_count`

## Manifest Integration Rules

Phase 4 may optionally sync review-derived fields back into the manifest, but
it must not overwrite deterministic evidence.

If manifest sync is implemented, update only:

- `review_basis_at`
- `semantic_status`
- `developer_semantics_status`
- `required_actions`

Only completed and valid review records may affect the manifest.

## Test Plan

Add focused tests for:

1. draft review initialization from a manifest atom
2. validation success for a minimal completed review
3. validation failure for missing provenance
4. validation failure for missing required sections on completed reviews
5. review index generation
6. `--priority` seeding behavior using the review queue

Recommended test files:

- `tests/test_audit_reviews.py`

Prefer compact synthetic fixtures over depending on the full repo state when
possible.

## Execution Order

Implement in this order:

1. add path constants for review artifacts
2. add review schema helpers in `scripts/auditlib/reviews.py`
3. implement draft review initializer
4. implement validator
5. implement review index builder
6. add tests
7. seed a small set of draft review files for `review_now`

## Idempotent Command Set

An agent resuming this phase should rerun:

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_risk.py
python scripts/init_audit_reviews.py --priority review_now --limit 10
python scripts/validate_audit_reviews.py
pytest -q tests/test_audit_reviews.py
```

If a prerequisite artifact is stale or missing, regenerate it before debugging
Phase 4 logic.

## Exit Criteria

Phase 4 is complete when:

1. canonical review record schema exists
2. draft review initialization works
3. review validation works and emits machine-readable output
4. review index exists
5. a starter set of seeded review drafts exists for `review_now`
6. targeted tests pass

## Known Non-Goals

This phase does not:

- complete large-scale review throughput
- run runtime probes
- grant `trusted`
- replace the deterministic risk or structural pipeline

Those belong to later phases.

## Agent Handoff Template

If an implementing agent stops mid-phase, it should leave:

- current commit and branch
- files created or modified
- whether `data/audit_reviews/` contains drafts
- whether `data/audit/review_validation.json` is fresh
- whether `data/audit/review_index.json` is fresh
- tests run
- current blockers

If no handoff is available, the next agent should assume the review artifacts
may be stale and rerun the idempotent command set.

