# Phase 7 Human / Model Review Pass Plan

This document is the concrete implementation plan for Phase 7 from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md):
the actual human / model review pass over the audited atom portfolio.

It is written to survive process restarts. A planning or coding agent should be
able to resume from this file plus the existing audit artifacts without relying
on prior chat context.

## Goal

Turn the review-record system from Phase 4 into an operational review workflow
that:

- selects atoms for review in a deterministic priority order
- tracks review progress and backlog
- validates completed review records before they affect portfolio state
- syncs validated review judgments back into `data/audit_manifest.json`
- promotes atoms only within the trust rules from
  [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md)

Phase 7 is about executing and integrating structured reviews. It is not yet
about repository-wide code remediation or CI gating.

## Current Baseline

Available inputs:

- manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- risk report:
  [risk_report.json](/Users/conrad/personal/ageo-atoms/data/audit/risk_report.json)
- review queue:
  [review_queue.csv](/Users/conrad/personal/ageo-atoms/data/audit/review_queue.csv)
- review directory:
  [data/audit_reviews/](/Users/conrad/personal/ageo-atoms/data/audit_reviews)
- review validation:
  [review_validation.json](/Users/conrad/personal/ageo-atoms/data/audit/review_validation.json)
- review index:
  [review_index.json](/Users/conrad/personal/ageo-atoms/data/audit/review_index.json)
- semantic report:
  [semantic_report.json](/Users/conrad/personal/ageo-atoms/data/audit/semantic_report.json)
- parity coverage:
  [parity_coverage.json](/Users/conrad/personal/ageo-atoms/data/audit/parity_coverage.json)

Current limitation:

- review files exist, but there is no deterministic review-pass runner
- there is no durable batching artifact for who should be reviewed next
- there is no validated manifest sync from review records
- there is no progress report for review throughput / backlog / trust readiness
- `trusted` promotion rules are documented but not operationalized

## Deliverables

Required code:

- `scripts/auditlib/review_pass.py`
- `scripts/apply_audit_reviews.py`
- `scripts/report_audit_progress.py`

Files likely to modify:

- `scripts/auditlib/reviews.py`
- `scripts/auditlib/models.py`
- `scripts/auditlib/paths.py`
- `scripts/validate_audit_reviews.py`

Optional supporting code:

- `scripts/select_audit_review_batch.py`
- `scripts/export_review_batch.py`

Required generated artifacts:

- `data/audit/review_progress.json`
- `data/audit/review_backlog.csv`
- updated [review_index.json](/Users/conrad/personal/ageo-atoms/data/audit/review_index.json)
- updated [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)

Optional generated artifacts:

- `data/audit/review_batch.json`
- `data/audit/review_batch.md`

## Phase 7 Scope

Phase 7 should implement the workflow around review records, not the reviews
themselves as free-form human work.

Required capabilities:

1. choose the next atoms to review using deterministic priority rules
2. report current review coverage and backlog
3. merge validated review results into the manifest
4. compute review-derived portfolio state without overriding deterministic
   evidence improperly

Out of scope for Phase 7:

- rewriting atoms
- changing ingest prompts
- CI enforcement
- broad parity expansion
- automatic promotion of any atom to `trusted` without a valid review record

## Review Priority Model

Phase 7 should not invent a new queue from scratch. It should consume the
existing risk and review artifacts.

Primary ordering inputs:

1. `review_priority` from the manifest / review queue
2. `risk_tier`
3. `overall_verdict`
4. `semantic_status`
5. presence or absence of any review record

Deterministic review order:

1. high-risk atoms with no review record
2. high-risk atoms with only draft review records
3. medium-risk atoms with `broken` or `misleading` verdicts
4. medium-risk atoms lacking provenance or parity evidence
5. low-risk atoms that are candidates for later promotion

Stable tie-breakers:

1. higher `risk_score`
2. worse `overall_verdict`
3. lexicographic `atom_id`

## Review State Model

Phase 7 needs a portfolio-level view of review completion.

Required per-atom derived fields in the manifest:

- `review_status`
- `review_record_path`
- `reviewer_type`
- `reviewed_at`
- `review_semantic_verdict`
- `review_developer_semantics_verdict`
- `review_limitations`
- `review_required_actions`
- `trust_readiness`
- `trust_blockers`

If these fields do not already exist in the manifest schema, Phase 7 should add
them conservatively through shared model helpers.

Allowed `trust_readiness` values:

- `not_reviewed`
- `review_in_progress`
- `reviewed_not_trust_ready`
- `eligible_for_trusted_promotion`

`trust_readiness` is not the same as `overall_verdict`. It expresses whether
the atom has satisfied the documented prerequisites for a later promotion
decision.

## Trust Promotion Rules

Phase 7 must encode the trust rules from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md), but
without auto-promoting the whole portfolio.

An atom may be considered `eligible_for_trusted_promotion` only if:

- a validated completed review record exists
- `semantic_verdict` from review is `pass`
- `developer_semantics_verdict` from review is `pass` or `not_applicable`
- structural status is `pass`
- runtime status is `pass` or `not_applicable`
- parity coverage is `positive_and_negative`,
  `parity_or_usage_equivalent`, or explicitly `not_applicable`
- provenance is pinned in the review record
- no blocking structural or semantic findings remain

An atom must stay `reviewed_not_trust_ready` if:

- review completed but verdict is `partial` or `fail`
- provenance is missing
- parity coverage is weak
- runtime / structural state is not sufficient
- review limitations or required actions indicate unresolved concerns

Important:

- deterministic tools may cap or deny trust
- Phase 7 may mark trust readiness
- Phase 7 must not auto-assign `overall_verdict=trusted` across the portfolio
  just because readiness is satisfied

At most, Phase 7 may add an explicit manifest field like
`recommended_overall_verdict` if useful. If added, it must remain conservative
and never bypass the review gating rules.

## Review Batch Artifact

Phase 7 should produce a deterministic batch artifact for the next review pass.

Suggested artifact:

- `data/audit/review_backlog.csv`

Each row should include:

- `atom_id`
- `atom_name`
- `review_priority`
- `risk_tier`
- `risk_score`
- `overall_verdict`
- `semantic_status`
- `review_status`
- `has_completed_review`
- `trust_readiness`
- `review_batch_reason`
- `recommended_review_action`

Stable `review_batch_reason` codes:

- `REVIEW_HIGH_RISK_UNREVIEWED`
- `REVIEW_HIGH_RISK_DRAFT_ONLY`
- `REVIEW_MEDIUM_RISK_BROKEN`
- `REVIEW_MEDIUM_RISK_MISLEADING`
- `REVIEW_PROVENANCE_GAP`
- `REVIEW_TRUST_PROMOTION_CANDIDATE`
- `REVIEW_LIMITATIONS_UNDOCUMENTED`

## Manifest Sync Rules

Phase 7 should sync only validated review records into the manifest.

Required behavior:

- ignore `draft` records for promotion/readiness purposes
- allow `draft` records to affect progress reporting only
- accept only one active review record per atom for sync
- if multiple completed records exist, prefer the newest non-superseded record
- if conflicts remain, mark the atom as blocked and report the conflict

Fields synced from review records:

- review metadata
- review semantic verdicts
- limitations
- required actions
- authoritative source summary
- provenance summary

Fields that must not be overwritten by Phase 7 sync:

- structural findings
- runtime findings
- deterministic signature findings
- parity coverage details
- risk score inputs other than review completion status

If a review verdict contradicts deterministic evidence, keep both:

- deterministic findings remain in the manifest
- review results are added alongside them
- trust readiness should remain blocked until the contradiction is resolved

## Progress Report Schema

`data/audit/review_progress.json` should contain:

- `schema_version`
- `generated_at`
- `summary`
- `by_review_status`
- `by_reviewer_type`
- `by_risk_tier`
- `by_trust_readiness`
- `high_priority_backlog`
- `promotion_candidates`
- `blocked_candidates`

`summary` should include:

- atom count
- completed review count
- draft review count
- missing review count
- high-risk reviewed count
- high-risk unreviewed count
- trust-ready count
- trust-blocked count

`promotion_candidates` should list atoms that satisfy all documented
prerequisites except the final manual promotion step.

`blocked_candidates` should list atoms with completed reviews that still fail
trust readiness, including blocker codes.

## Required Blocker Codes

Use stable blocker codes in `trust_blockers`:

- `TRUST_BLOCK_REVIEW_MISSING`
- `TRUST_BLOCK_REVIEW_DRAFT`
- `TRUST_BLOCK_SEMANTIC_REVIEW_NOT_PASS`
- `TRUST_BLOCK_DEVELOPER_REVIEW_NOT_PASS`
- `TRUST_BLOCK_STRUCTURAL_STATUS`
- `TRUST_BLOCK_RUNTIME_STATUS`
- `TRUST_BLOCK_PARITY_COVERAGE`
- `TRUST_BLOCK_PROVENANCE_MISSING`
- `TRUST_BLOCK_REQUIRED_ACTIONS_OPEN`
- `TRUST_BLOCK_LIMITATIONS_PRESENT`
- `TRUST_BLOCK_DETERMINISTIC_CONFLICT`
- `TRUST_BLOCK_REVIEW_CONFLICT`

## CLI Requirements

### `scripts/report_audit_progress.py`

Purpose:

- join manifest and review records into a durable progress/backlog report

Required behavior:

- read the manifest
- read all review records
- read current review validation / index artifacts
- compute per-atom trust readiness and blockers
- write `data/audit/review_progress.json`
- write `data/audit/review_backlog.csv`

Output:

- a stable machine-readable progress report
- a CSV backlog ordered by review priority

### `scripts/apply_audit_reviews.py`

Purpose:

- sync validated review records into the manifest

Required behavior:

- read the manifest
- read review records
- validate or consume validation output
- apply only `completed` non-superseded records
- update manifest review fields and trust-readiness fields
- write updated manifest
- refresh review index if needed

Required options:

- `--atom-id`
- `--only-completed`
- `--fail-on-conflict`

### Optional `scripts/select_audit_review_batch.py`

Purpose:

- emit a smaller deterministic review batch for human/model reviewers

Required behavior:

- accept `--priority`
- accept `--limit`
- select from backlog, not from raw manifest rows

This CLI is optional if `report_audit_progress.py` already produces a
sufficiently ordered backlog artifact.

## Shared Library Responsibilities

`scripts/auditlib/review_pass.py` should own:

- review-record selection / resolution
- review conflict detection
- trust-readiness computation
- blocker derivation
- progress report assembly
- backlog row generation

`scripts/auditlib/reviews.py` may need to absorb:

- helper to resolve latest active review record per atom
- helper to summarize provenance completeness

## Tests

Required test file:

- `tests/test_audit_review_pass.py`

Minimum test cases:

1. completed review with full prerequisites yields
   `eligible_for_trusted_promotion`
2. draft review yields `review_in_progress`
3. missing review yields `not_reviewed`
4. completed review with structural/runtime/parity gaps yields
   `reviewed_not_trust_ready`
5. conflicting completed review records produce `TRUST_BLOCK_REVIEW_CONFLICT`
6. deterministic semantic failure plus positive review produces
   `TRUST_BLOCK_DETERMINISTIC_CONFLICT`
7. backlog ordering prioritizes high-risk unreviewed atoms first
8. manifest sync updates review fields without clobbering deterministic status

Optional integration tests:

- `tests/test_audit_reviews.py`
- `tests/test_audit_risk.py`

## Implementation Order

1. add any needed path constants for progress/backlog artifacts
2. implement review resolution and trust-readiness logic in
   `scripts/auditlib/review_pass.py`
3. add manifest-sync helpers
4. implement `scripts/report_audit_progress.py`
5. implement `scripts/apply_audit_reviews.py`
6. update review index / validation integration if needed
7. add tests
8. generate fresh progress/backlog artifacts

## Idempotent Commands

From repo root:

```bash
python scripts/validate_audit_reviews.py
python scripts/report_audit_progress.py
python scripts/apply_audit_reviews.py --only-completed
pytest -q tests/test_audit_review_pass.py tests/test_audit_reviews.py
```

Recommended broader verification:

```bash
pytest -q tests/test_audit_tools.py tests/test_audit_manifest_validation.py tests/test_audit_structural.py tests/test_audit_risk.py tests/test_audit_reviews.py tests/test_audit_semantics.py tests/test_audit_parity.py tests/test_audit_review_pass.py
```

## Exit Criteria

Phase 7 is complete when:

1. the repo can compute a deterministic review backlog
2. the repo can compute portfolio review progress
3. validated completed reviews can be synced into the manifest
4. every high-risk atom is visible as reviewed, draft, or unreviewed
5. trust readiness is explicit per atom
6. promotion blockers are explicit per atom
7. tests cover review resolution and trust-readiness logic

## Non-Goals And Guardrails

Do not:

- auto-promote every eligible atom to `trusted`
- overwrite deterministic evidence with review opinions
- treat draft reviews as completed evidence
- resolve review conflicts silently
- broaden trust exemptions without explicit provenance-backed logic

## Restart Checklist

On restart, first verify:

1. whether `data/audit_reviews/` contains completed reviews beyond the initial
   drafts
2. whether `data/audit/review_validation.json` is fresh
3. whether `data/audit/review_progress.json` exists and is fresh
4. whether the manifest already contains synced review fields
5. whether review conflicts have already been surfaced

Then rerun the idempotent commands above before making further changes.
