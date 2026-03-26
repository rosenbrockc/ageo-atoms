# Phase 3 Risk Triage Plan

This document is the concrete implementation plan for Phase 3 from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md):
semantic risk triage.

It is written to survive process restarts. A new planning or coding agent
should be able to resume from this file plus the generated audit artifacts
without reconstructing intent from prior chat history.

## Goal

Replace the current coarse `risk_tier` heuristic in
[audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
with a deterministic, explainable, evidence-backed risk triage system that:

- ranks atoms by likelihood of semantic failure
- records explicit `risk_reasons`
- records a numeric `risk_score`
- produces a durable review queue
- persists a machine-readable portfolio report

Phase 3 does not make semantic trust decisions. It only prioritizes where
semantic review effort should go first.

## Current Baseline

Available inputs:

- validated manifest:
  [audit_manifest.json](/Users/conrad/personal/ageo-atoms/data/audit_manifest.json)
- manifest validation:
  [manifest_validation.json](/Users/conrad/personal/ageo-atoms/data/audit/manifest_validation.json)
- structural report:
  [structural_report.json](/Users/conrad/personal/ageo-atoms/data/audit/structural_report.json)
- signature fidelity evidence:
  [data/audit/evidence/](/Users/conrad/personal/ageo-atoms/data/audit/evidence)
- acceptability scores:
  [audit_scores.csv](/Users/conrad/personal/ageo-atoms/data/audit_scores.csv)

Current limitation:

- `risk_tier` is still derived from coarse heuristics in
  [inventory.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/inventory.py)
- `risk_reasons` are inventory-oriented but not yet joined to structural and
  fidelity evidence
- there is no repo-wide risk report or review queue artifact

## Deliverables

Required code:

- `scripts/auditlib/risk.py`
- `scripts/audit_risk.py`

Files likely to modify:

- `scripts/auditlib/inventory.py`
- `scripts/auditlib/paths.py`
- `scripts/audit_acceptability.py`
  only if needed for shared helpers, not for new risk logic
- tests under `tests/`

Required generated artifacts:

- `data/audit/risk_report.json`
- `data/audit/review_queue.csv`
- updated `data/audit_manifest.json`

## Risk Model

Phase 3 should compute a deterministic `risk_score` in the range `0..100`,
where higher means greater probability that semantic review will uncover a
meaningful problem.

The score should be evidence-backed, not opinion-based.

### Required Output Fields Per Atom

Every manifest row should gain:

- `risk_score`
- `risk_tier`
- `risk_reasons`
- `risk_dimensions`
- `review_priority`

Definitions:

- `risk_score`: numeric triage score
- `risk_tier`: `high` / `medium` / `low`
- `risk_reasons`: stable machine-readable reason codes
- `risk_dimensions`: structured breakdown by category
- `review_priority`: deterministic sortable label such as
  `review_now`, `review_soon`, `review_later`

### Risk Dimensions

Use these dimensions:

- structural risk
- fidelity risk
- evidence-gap risk
- statefulness risk
- generation/refinement risk
- FFI risk
- naming/semantics proxy risk

### Initial Weighting

Start with this conservative weighting:

- structural risk: 25
- fidelity risk: 25
- evidence-gap risk: 20
- statefulness risk: 10
- generation/refinement risk: 10
- FFI risk: 5
- naming/semantics proxy risk: 5

Base formula:

`risk_score = structural + fidelity + evidence_gap + statefulness + generation + ffi + semantics_proxy`

### Tier Thresholds

- `high`: `risk_score >= 60`
- `medium`: `30 <= risk_score < 60`
- `low`: `risk_score < 30`

These are implementation defaults, not policy. They should be easy to tune in
one place.

## Required Reason Codes

Use stable reason codes so downstream review tools can group them.

Structural-driven:

- `RISK_STRUCTURAL_FAIL`
- `RISK_STRUCTURAL_PARTIAL`
- `RISK_CDG_ISSUES`
- `RISK_STUB_PUBLIC_API`

Fidelity-driven:

- `RISK_UPSTREAM_UNMAPPED`
- `RISK_SIGNATURE_MISMATCH`
- `RISK_WEAK_TYPES`
- `RISK_WEAK_UPSTREAM_ANCHOR`

Evidence-gap driven:

- `RISK_MISSING_PARITY`
- `RISK_MISSING_PROVENANCE`
- `RISK_MISSING_REVIEW_BASIS`
- `RISK_NO_RUNTIME_EVIDENCE`

Statefulness / decomposition:

- `RISK_STATEFUL_API`
- `RISK_STATE_MODEL_PRESENT`
- `RISK_PROCEDURAL_WRAPPER`

Origin-driven:

- `RISK_GENERATED_INGEST`
- `RISK_REFINED_INGEST`
- `RISK_SKLEARN_GENERATED_FAMILY`
- `RISK_FFI_BACKED`
- `RISK_STOCHASTIC`

Naming / semantics proxies:

- `RISK_PLACEHOLDER_DOCSTRING`
- `RISK_GENERATED_ABSTRACTION_LANGUAGE`
- `RISK_LOW_NAME_ALIGNMENT`

## Evidence Sources

The Phase 3 risk computation should only consume persisted evidence or stable
manifest facts.

Priority sources:

1. `data/audit_manifest.json`
2. `data/audit/structural_report.json`
3. `data/audit/evidence/*.json`
4. `data/audit_scores.csv`

Do not re-run expensive audits inside the core scoring function. The CLI may
validate that prerequisite artifacts exist, but risk scoring should mostly be a
join-and-score pass.

## Scoring Rules

### Structural Risk

Inputs:

- `structural_status`
- `structural_findings`

Rules:

- `fail` adds 25
- `partial` adds 12
- `pass` adds 0
- `STRUCT_STUB_PUBLIC_API` forces `RISK_STUB_PUBLIC_API`
- `STRUCT_CDG_INVALID` or `STRUCT_CDG_TYPE_HEALTH_LOW` add extra points

### Fidelity Risk

Inputs:

- signature evidence file
- manifest weak type fields

Rules:

- unmapped upstream adds 15
- missing upstream signature adds 8
- invented parameter mismatch adds 12
- missing required parameter mismatch adds 12
- weak types add 6

### Evidence-Gap Risk

Inputs:

- `has_parity_tests`
- `review_basis_at`
- `source_revision`
- `upstream_version`
- `runtime_status`

Rules:

- no parity adds 8
- no provenance adds 6
- no review basis adds 3
- runtime unknown adds 3

### Stateful / Procedural Risk

Inputs:

- `stateful`
- `stateful_kind`
- `procedural`

Rules:

- explicit state model adds 8
- argument or return state adds 5
- procedural wrapper adds 4

### Origin / Platform Risk

Inputs:

- `source_kind`
- `ffi`
- `stochastic`
- family/module markers

Rules:

- generated ingest adds 10
- refined ingest adds 5
- FFI adds 5
- stochastic adds 4
- sklearn generated family adds 6

### Naming / Semantics Proxy Risk

Inputs:

- `docstring_summary`
- `inventory_notes`
- wrapper/upstream token overlap

Rules:

- placeholder docstring adds 3
- generated abstraction wording adds 2
- low wrapper/upstream token alignment adds 2

## Review Priority Rules

Review priority should be derived from both tier and reasons.

Suggested mapping:

- `review_now`:
  - all `high` risk atoms
  - any atom with `RISK_STUB_PUBLIC_API`
  - any atom with both `RISK_SIGNATURE_MISMATCH` and `RISK_STATEFUL_API`
- `review_soon`:
  - `medium` risk atoms with fidelity or statefulness reasons
- `review_later`:
  - remaining `medium`
  - all `low`

## Risk Report Schema

`data/audit/risk_report.json` should contain:

- `schema_version`
- `generated_at`
- `summary`
- `thresholds`
- `review_queue`
- `by_tier`
- `by_reason`
- `by_family`
- `source_artifacts`

`summary` should include:

- atom count
- high/medium/low counts
- review_now/review_soon/review_later counts
- top reason codes
- top risky families

## Review Queue CSV

`data/audit/review_queue.csv` should contain at least:

- `atom_id`
- `atom_name`
- `domain_family`
- `risk_score`
- `risk_tier`
- `review_priority`
- `risk_reasons`
- `structural_status`
- `semantic_status`
- `max_reviewable_verdict`

The CSV should be sorted by:

1. `review_priority`
2. descending `risk_score`
3. `atom_id`

## Manifest Integration Rules

`scripts/audit_risk.py` should:

1. load the current manifest
2. load structural and fidelity evidence
3. compute risk fields for every atom
4. update manifest rows in place
5. write `data/audit/risk_report.json`
6. write `data/audit/review_queue.csv`
7. write the updated manifest back to disk

It must not silently drop unrelated manifest fields.

## Test Plan

Add tests for:

1. risk scoring for a structurally failing atom
2. risk scoring for a mapped, hand-written, parity-covered atom
3. risk scoring for an unmapped generated wrapper
4. review priority derivation
5. risk report generation
6. manifest update round-trip

Recommended files:

- `tests/test_audit_risk.py`

The tests should avoid depending on the full repository artifact set when
possible. Prefer compact synthetic manifest rows and evidence snippets.

## Execution Order

Implement in this order:

1. add shared helpers and constants in `scripts/auditlib/risk.py`
2. implement per-atom risk scoring
3. implement report builders
4. implement CLI `scripts/audit_risk.py`
5. update manifest with new risk fields
6. add tests
7. regenerate Phase 3 artifacts

## Idempotent Command Set

An agent resuming this phase should rerun:

```bash
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_risk.py
pytest -q tests/test_audit_tools.py tests/test_audit_manifest_validation.py tests/test_audit_structural.py tests/test_audit_risk.py
```

If a prerequisite artifact is stale or missing, regenerate it before debugging
Phase 3 logic.

## Exit Criteria

Phase 3 is complete when:

1. every atom has `risk_score`, `risk_tier`, `risk_reasons`,
   `risk_dimensions`, and `review_priority`
2. `data/audit/risk_report.json` exists and validates
3. `data/audit/review_queue.csv` exists and is sorted deterministically
4. the high-risk portfolio is smaller than the full portfolio
5. targeted tests pass

## Known Non-Goals

This phase does not:

- grant `trusted`
- create human review records
- perform runtime probes
- perform return/state/generated-noun semantic checks

Those belong to later phases.

## Agent Handoff Template

If an implementing agent stops mid-phase, it should leave:

- current commit and branch
- files created or modified
- tests run
- whether `data/audit/risk_report.json` is fresh
- whether `data/audit/review_queue.csv` is fresh
- current blockers

If no handoff is available, the next agent should assume artifacts may be stale
and rerun the idempotent command set.

