# Ingester Rollout Workstream 2 Plan

This document is the implementation plan for the **next rollout slice after
smoke-coverage expansion**. It targets **Workstream 2** from
[INGESTER_ROLLOUT_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_ROLLOUT_PLAN.md):
curated end-to-end ingest regression fixtures for the families that taught us
the recent hardening lessons.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Extend the existing matcher ingest regression harness with two new curated
families:

1. a **grouped sklearn image helper** case
2. a **detector-style structured-output** case

The objective is to make those learned families part of the matcher’s durable
regression surface, rather than keeping them only as historical knowledge from
manual repairs in `ageo-atoms`.

## Why This Slice Matters

The hardening phases and the first rollout slice proved that:

- grouped output is a real first-class matcher path
- deterministic smoke coverage can protect grouped families
- narrow allowlists are workable

What we do not yet have is a stable end-to-end regression corpus for the
specific family shapes that drove those design decisions. Without that, the
matcher can still regress in ways the smoke tests alone will not catch.

## Current Baseline

The matcher already has a curated regression harness:

- `../ageo-matcher/sciona/ingester/regression_harness.py`
- `../ageo-matcher/tests/test_ingest_regression_harness.py`
- fixtures under `../ageo-matcher/tests/fixtures/ingest_regression/`
- goldens under `../ageo-matcher/tests/golden/ingest_regression/`

Current coverage includes:

- sklearn-style estimator
- rolling stateful class
- bayesian/message-passing family
- DSP biosignal pipeline
- non-Python FFI
- procedural ingest

What it does not include yet:

- grouped/family output layout as a regression-harness concept
- a detector-like case where structured returns must be preserved

## Scope

In scope:

- extend the regression harness model if needed
- add one grouped-family regression case
- add one detector-style regression case
- add or update goldens for those cases
- add matcher tests proving the new cases are in the default curated matrix

Out of scope:

- broad fixture corpus expansion
- live library imports in test
- full return-shape allowlist implementation
- any changes to `ageo-atoms` runtime probes

## Target Cases

### Case 1: Grouped Sklearn Images

Purpose:

- represent the grouped-family output shape that motivated Phase 4 and the
  rollout smoke expansion

Desired properties:

- pure helper-style functions
- coherent family/module package output
- output directory basename different from the symbol name
- expected artifacts still align with the standard surface

Recommended source shape:

- a small local Python fixture containing one or more image-helper-style
  functions such as:
  - `extract_patches_2d`
  - `reconstruct_from_patches_2d`
  - or a smaller custom grouped analogue if a direct sklearn-style fixture is
    simpler and more stable

Recommended regression assertion:

- the case belongs to a grouped-family class of ingest inputs
- the harness preserves the correct artifact surface
- the golden outputs remain stable under the grouped shape

### Case 2: Detector-Style Structured Output

Purpose:

- represent the family of detector wrappers that taught the return-shape and
  result-passthrough lessons

Desired properties:

- a detector-like function or shallow class method
- returns a structured bundle rather than a trivial scalar
- still small enough to be stable in a fixture
- sufficiently realistic to catch future output-extraction drift

Recommended source shape:

- a local fixture function that computes a detector-style result such as:
  - `{"rpeaks": peaks, "quality": quality}`
  - or a small named tuple / tuple bundle with clearly documented meaning

Recommended regression assertion:

- emitted/golden output continues to model the structured return explicitly
- the case remains in the curated corpus even if later output-binding rules are
  refactored

## Design Direction

Prefer extension of the **existing regression harness** over one-off tests.

That likely means:

1. add new case directories under `tests/fixtures/ingest_regression/`
2. add matching golden directories under `tests/golden/ingest_regression/`
3. update `default_ingest_regression_cases(...)`
4. update harness tests that assert case counts and family coverage

If the grouped-family case needs explicit grouping metadata, prefer adding a
small field to `IngestRegressionCase` rather than encoding grouping only in the
directory name.

## Likely Code Changes

Primary matcher files:

- `../ageo-matcher/sciona/ingester/regression_harness.py`
- `../ageo-matcher/tests/test_ingest_regression_harness.py`

Likely new fixture directories:

- `../ageo-matcher/tests/fixtures/ingest_regression/sklearn_grouped_images/`
- `../ageo-matcher/tests/fixtures/ingest_regression/detector_structured_output/`

Likely new golden directories:

- `../ageo-matcher/tests/golden/ingest_regression/sklearn_grouped_images/`
- `../ageo-matcher/tests/golden/ingest_regression/detector_structured_output/`

Secondary matcher files only if needed:

- helper code used by the harness to stage grouped outputs
- existing ingest regression fixture utilities

## Key Implementation Questions

1. Does the harness need an explicit `output_scope` field on
   `IngestRegressionCase`?
2. If yes, where should that flow:
   - monitor start
   - output dir naming
   - summary metadata
3. How should grouped output be represented in a golden directory:
   - still one case folder per ingest run
   - but with grouped-family semantics encoded in the fixture and summary
4. What is the smallest detector-style structured return that still catches
   return-shape drift without requiring external dependencies?
5. Which artifact surfaces are stable enough to compare as goldens for the new
   cases?

## Recommended Answers

Use these unless implementation evidence forces a different choice:

- yes, add a small optional `output_scope` field to `IngestRegressionCase`
- default it to `symbol`; set the grouped sklearn case to `family`
- keep one case folder per run under the harness output root
- use a local structured-return fixture instead of a live BioSPPy import
- continue comparing canonical IR, planning graph, atoms, witnesses, and CDG
  through the existing golden machinery

## Workstreams

### Workstream A: Harness Model Extension

Objective:

- make the curated harness capable of representing grouped-output cases

Tasks:

- inspect `IngestRegressionCase` and `run_ingest_regression_case(...)`
- add an optional `output_scope` field if needed
- ensure monitor start and output summary can reflect that scope
- keep the default path backward-compatible for existing six cases

Exit criteria:

- existing cases still pass unchanged
- new grouped case can declare itself explicitly

### Workstream B: Grouped Sklearn Fixture And Goldens

Objective:

- add a grouped-family regression case

Tasks:

- create the grouped source fixture
- define the new default case entry
- create stable goldens for canonical IR / planning graph / atoms / witnesses /
  CDG
- add regression tests proving the case is in the curated matrix

Exit criteria:

- the grouped case is part of `default_ingest_regression_cases(...)`
- the harness can compare it against goldens successfully

### Workstream C: Detector-Style Structured Output Fixture And Goldens

Objective:

- add a structured-return detector-style regression case

Tasks:

- create the detector-like source fixture
- define the new default case entry
- create stable goldens
- add at least one semantic expectation that makes the case meaningful

Exit criteria:

- the detector-style case is part of the curated matrix
- the harness can compare it against goldens successfully

### Workstream D: Harness Tests

Objective:

- keep the new corpus stable

Required coverage:

- case-count and family assertions updated for the expanded matrix
- default real-world corpus test updated for the new cases
- grouped/family metadata is visible if added

Exit criteria:

- matcher tests fail if either new case drops out of the default matrix

## Verification Commands

Minimum:

- targeted matcher tests for the regression harness

Recommended:

- `pytest -q tests/test_ingest_regression_harness.py`
- if useful, a narrower focused selector while iterating

If new goldens are added with a fixture agent approach, verify:

- the default corpus count increases as expected
- golden comparison remains fully matched

## Success Criteria

This slice is complete when:

- the curated harness includes both a grouped-family case and a
  detector-style structured-output case
- both cases participate in the default matrix
- both cases compare cleanly against checked-in goldens
- existing harness cases remain green

## Suggested Worker Scope

Assign a coding worker ownership of:

- `../ageo-matcher/sciona/ingester/regression_harness.py`
- `../ageo-matcher/tests/test_ingest_regression_harness.py`
- new fixture directories under `tests/fixtures/ingest_regression/`
- new golden directories under `tests/golden/ingest_regression/`

Tell the worker:

- do not touch unrelated `.claude` files
- do not modify the Phase 5 smoke gate unless the harness absolutely requires
  a tiny compatibility hook
- keep the write set focused on the harness and its fixtures/goldens

## What Remains After This Slice

If this slice lands cleanly, the next logical step is Workstream 3 from
[INGESTER_ROLLOUT_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_ROLLOUT_PLAN.md):
a narrow return-shape knowledge layer for explicitly known structured outputs.
