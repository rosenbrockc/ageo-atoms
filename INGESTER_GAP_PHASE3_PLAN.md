# Ingester Gap Phase 3 Plan

This document is the detailed implementation plan for **Phase 3: Regression
Corpus Expansion** from
[INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md).

It should be read together with:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Objective

Expand the matcher’s curated ingest regression corpus so that more of the real
repair history from `ageo-atoms` is protected by stable fixtures and goldens.

This phase should pay down the `partially_covered` row in the crosswalk for:

- curated regression coverage for grouped families

It should also help later phases by making structured-return and grouped-family
behavior more durable under matcher changes.

## Why This Phase Comes After Phase 2

Phase 2 expanded matcher-side smoke coverage. That protects some staged outputs
at ingest time, but it does not yet create durable regression fixtures for the
families that taught us the lessons.

Phase 3 should therefore add a small number of curated regression cases that:

- are cheap to run
- mirror real repair pressures
- make future matcher regressions obvious

## Scope

In scope:

- `../ageo-matcher/sciona/ingester/regression_harness.py`
- `../ageo-matcher/tests/test_ingest_regression_harness.py`
- new curated fixture files under:
  - `../ageo-matcher/tests/fixtures/ingest_regression/`
- new golden artifacts under:
  - `../ageo-matcher/tests/golden/ingest_regression/`

Out of scope:

- changes to smoke validation
- changes to return-shape allowlists
- changes to grouped-ingest CLI/workflow
- live integration against full external libraries

## Candidate Families

Choose compact families that already appear in the repair history and crosswalk.

### Required first slice

Add one new curated regression family beyond the existing grouped sklearn and
detector-structured-output cases.

Preferred target:

- `tempo_jl/offsets`

Why:

- it is the clearest `uncovered` grouped-family rollout example in the
  crosswalk
- it mirrors the real repo shift from fragmented singleton packages to a
  grouped family package
- it exercises grouped publication semantics without requiring live Julia
  runtime integration

### Optional second slice if it fits naturally

One additional compact family is acceptable only if it stays small and
reuses the same harness patterns.

Good options:

- a grouped helper family with simple pure-function goldens
- a detector-like fixture only if it adds a materially different regression
  shape from what already exists

Do not add a second family if it materially enlarges the phase.

## Required Evidence Sources

In `ageo-atoms`:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- grouped family examples:
  - [tempo_jl/offsets/atoms.py](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets/atoms.py)
  - [tempo_jl/offsets/cdg.json](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets/cdg.json)
  - [sklearn/images/atoms.py](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images/atoms.py)

In `../ageo-matcher`:

- [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py)
- [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py)
- existing curated fixtures:
  - [tests/fixtures/ingest_regression/sklearn_grouped_images/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/sklearn_grouped_images/source.py)
  - [tests/fixtures/ingest_regression/detector_structured_output/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/detector_structured_output/source.py)
- existing goldens under:
  - [tests/golden/ingest_regression/sklearn_grouped_images](/Users/conrad/personal/ageo-matcher/tests/golden/ingest_regression/sklearn_grouped_images)
  - [tests/golden/ingest_regression/detector_structured_output](/Users/conrad/personal/ageo-matcher/tests/golden/ingest_regression/detector_structured_output)

## Implementation Strategy

### 1. Add one compact grouped-family regression case

The new case should assert:

- grouped-family output scope
- expected grouped artifacts
- stable golden surfaces for:
  - `atoms.py`
  - `witnesses.py`
  - `cdg.json`
  - `canonical_ir.json`
  - `planning_graph.json`

The case does not need to mirror the exact real implementation. It only needs
to preserve the grouped-family pressures that mattered in the repair history.

### 2. Keep the fixture minimal and synthetic

Do not ingest live Tempo.jl or depend on Julia.

Instead:

- create a compact local source fixture that captures the same grouped helper
  shape
- name the case and comments clearly so the real-world mapping is obvious

### 3. Prefer stable harness assertions over fragile textual details

Goldens should focus on stable artifact shapes and grouped output semantics.
Avoid over-encoding incidental formatting or unstable IDs if the harness
already normalizes them.

## Questions The Coding Worker Must Resolve

1. Is a single grouped `tempo_offsets` style fixture sufficient, or does the
   harness need a second helper row to exercise publication summary semantics?
2. Which golden artifacts are strictly required for this new case based on the
   harness defaults?
3. Does the case fit best as a new default curated case, or as a focused test
   fixture used only by one test?
4. What comments or labels best preserve the link back to the `tempo_jl/offsets`
   lesson from the crosswalk?

## Expected Write Scope

The worker should own only:

- `../ageo-matcher/sciona/ingester/regression_harness.py`
- `../ageo-matcher/tests/test_ingest_regression_harness.py`
- one new fixture directory under `../ageo-matcher/tests/fixtures/ingest_regression/`
- one new golden directory under `../ageo-matcher/tests/golden/ingest_regression/`

The worker should not:

- edit smoke files
- edit return-shape files
- edit `ageo-atoms`
- touch unrelated local dirty files

## Required Deliverables

1. one new curated grouped-family regression case
2. corresponding fixture source file
3. corresponding golden artifact directory
4. harness test updates proving the new case is part of the curated matrix
5. a short worker report listing:
   - files changed
   - case id added
   - whether any second case was intentionally deferred

## Validation

Required matcher-side commands:

- `pytest -q tests/test_ingest_regression_harness.py`

No `ageo-atoms` tests are required in this phase.

## Exit Criteria

This phase is complete when:

1. the matcher regression harness covers at least one new grouped family beyond
   the existing grouped sklearn example
2. the new case is stable enough to live in the curated default matrix
3. the crosswalk’s `tempo_jl/offsets` uncovered row can be reevaluated after
   landing this phase

## Risks

- creating a fixture that is too literal to the real Tempo.jl source will make
  the test harder to maintain
- unstable goldens will create unnecessary churn
- adding too many cases at once will make review noisy and reduce confidence

## Suggested Worker Slice

Implement one new grouped-family case modeled on the `tempo_jl/offsets`
lesson. Stop after that single case unless a second case falls out almost for
free from the same harness edits.
