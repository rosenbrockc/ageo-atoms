# Ingester Gap Phase 1 Plan

This document is the detailed implementation plan for **Phase 1: Cross-Repo
Lesson Traceability** from
[INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md).

Primary implementation repo:

- `ageo-atoms`

Referenced validation/coverage repo:

- `../ageo-matcher`

## Objective

Create a durable crosswalk that maps the highest-signal repaired
`ageo-atoms` families and trust-debt lessons to the matcher features, smoke
cases, and regression tests that now cover them.

The crosswalk should make it possible for future planner or coding agents to
answer all of these quickly:

- which repaired families already have matcher protection
- which families are only partially covered
- which learned repairs still have no matcher-side representation
- which later rollout phases should target which uncovered rows

## Why This Phase Comes First

The remaining gaps are no longer architecture-first. They are rollout-first.
That means later phases need a precise target list instead of relying on repo
memory and verbal summaries.

Without a crosswalk:

- smoke expansion risks picking arbitrary families
- regression expansion risks duplicating already-covered lessons
- structured-return expansion risks encoding the wrong cases

## Scope

In scope:

- one durable crosswalk artifact in `ageo-atoms`
- one row per **major repair family or repair theme**
- links to:
  - repaired `ageo-atoms` evidence
  - matcher implementation files
  - matcher tests, smoke cases, or regression fixtures
- normalized coverage status:
  - `covered`
  - `partially_covered`
  - `uncovered`
- a short recommendation per row describing the next useful follow-up

Out of scope:

- adding new matcher behavior
- changing audit logic or manifest schema
- creating a comprehensive row for every atom in the repo
- automatic mining of git history

## Target Artifact

Create one canonical artifact:

- `INGESTER_LESSON_CROSSWALK.md`

The file should be human-readable first. It should be optimized for planner and
reviewer use, not for machine ingestion.

Recommended structure:

1. purpose and usage notes
2. coverage-status legend
3. a flat table with one row per family/theme
4. a short “priority follow-ups” section grouped by uncovered/partial rows

## Row Granularity

Use **family-or-theme level**, not atom level.

Good rows:

- `biosppy detectors: signature fidelity`
- `biosppy detectors: structured returns`
- `sklearn grouped images: grouped family publication`
- `tempo_jl offsets: grouped package scoping`
- `cyclic recursive decomposition rejection`

Bad rows:

- one row per small atom
- one row per commit
- one row per abstract architecture concept with no repo anchor

## Minimum Required Coverage Set

The first version of the crosswalk must cover at least these rows:

1. signature fidelity for BioSPPy detector families
2. structured-return handling for detector-like APIs
3. witness/decorator correctness
4. grouped publication for grouped package families
5. ingest-time smoke rejection
6. curated regression coverage for grouped families
7. recursive decomposition / CDG cycle rejection
8. grouped-family rollout examples:
   - `sklearn/images`
   - `tempo_jl/offsets`

Optional rows if cheap:

- variadic signature/witness handling
- grouped publication monitor summaries
- narrow return-shape allowlist coverage

## Required Evidence Sources

The implementation should inspect at least these files.

In `ageo-atoms`:

- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)
- [INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md)
- [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
- repaired family examples:
  - [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py)
  - [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py)
  - [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py)
  - [images/atoms.py](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images/atoms.py)
  - [offsets/atoms.py](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets/atoms.py)

In `../ageo-matcher`:

- [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py)
- [chunker.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/chunker.py)
- [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py)
- [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py)
- [regression_harness.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/regression_harness.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py)
- [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py)

Tests/fixtures to inspect:

- [test_ingester_emitter.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_emitter.py)
- [test_ingester_chunker.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_chunker.py)
- [test_ingester_return_shapes.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_return_shapes.py)
- [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py)
- [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py)
- [test_ingest_regression_harness.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_regression_harness.py)
- [tests/fixtures/ingest_regression/sklearn_grouped_images/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/sklearn_grouped_images/source.py)
- [tests/fixtures/ingest_regression/detector_structured_output/source.py](/Users/conrad/personal/ageo-matcher/tests/fixtures/ingest_regression/detector_structured_output/source.py)

## Row Schema

Each row should include:

- `lesson_or_family`
- `ageo_atoms_evidence`
- `matcher_coverage_status`
- `matcher_implementation`
- `matcher_tests_or_fixtures`
- `remaining_gap`
- `recommended_next_phase`

Status guidance:

- `covered`
  Matcher behavior and tests directly protect the lesson in normal use.
- `partially_covered`
  The lesson is absorbed architecturally, but rollout is still narrow or
  incomplete.
- `uncovered`
  The lesson still depends entirely on manual repo repair or audit logic.

## Implementation Steps

1. Read the current audit and execution-plan docs to derive the candidate row
   set.
2. Confirm each row’s matcher evidence by reading the referenced matcher
   implementation and tests.
3. Write `INGESTER_LESSON_CROSSWALK.md`.
4. Keep the file concise and planning-oriented.
5. Do not modify matcher code in this phase.

## Validation

This phase is documentation-only. Validation should be lightweight:

- ensure every row references real files that exist
- ensure every `partially_covered` or `uncovered` row points cleanly to a
  later phase from
  [INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md)

Suggested commands:

- `rg --files | rg 'INGESTER_(RISK_LESSONS_AUDIT|GAP_CLOSURE_EXECUTION_PLAN|GAP_PHASE1_PLAN|LESSON_CROSSWALK)\.md'`
- targeted `sed -n` / `rg -n` checks against referenced matcher files

## Exit Criteria

This phase is complete when:

1. `INGESTER_LESSON_CROSSWALK.md` exists.
2. It covers the minimum required family/theme set above.
3. Every row has explicit repo evidence on both sides when coverage is partial
   or complete.
4. Later planners can use the crosswalk to choose targets for smoke,
   regression, return-shape, and grouped-ingest follow-up phases.

## Handoff Guidance For The Coding Worker

The worker should own only:

- creation of `INGESTER_LESSON_CROSSWALK.md`

The worker should not:

- edit matcher code
- edit audit code
- touch unrelated local dirty files

The worker’s final report should include:

- the file created
- the row count
- the statuses used
- the main uncovered or partially covered rows it found
