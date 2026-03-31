# Ingester Gap Phase 4 Plan

This document is the detailed implementation plan for **Phase 4:
Structured-Return Allowlist Expansion** from
[INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md).

It should be read together with:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Objective

Expand the matcher’s explicit structured-return allowlist by one additional
justified case, without reopening broad output-extraction heuristics.

This phase should pay down the `partially_covered` row in the crosswalk for:

- structured-return handling for detector-like APIs

## Why This Phase Comes After Phase 3

We now have:

- the original `PeakDetector.detect` allowlisted case
- a grouped-family regression corpus that covers more of the repair history

That makes it safe to add one more narrow structured-return case, as long as:

- the matcher still fails closed for everything else
- the new case is tied to a real repair theme
- the tests explicitly prove conservative fallback remains intact

## Scope

In scope:

- `../ageo-matcher/sciona/ingester/return_shapes.py`
- `../ageo-matcher/sciona/ingester/chunker.py` only if needed by the allowlist
  path
- `../ageo-matcher/sciona/ingester/emitter.py` only if the new binding kind
  path needs no-op confirmation
- `../ageo-matcher/tests/test_ingester_return_shapes.py`

Out of scope:

- smoke validation
- regression harness growth
- grouped-ingest workflow
- broad field-name inference
- tuple-slot heuristics based on names alone

## Candidate Case Shape

Add one additional explicit detector-like case with a stable dict-shaped return.

Recommended target shape:

- subject: `OnsetDetector`
- method: `detect_events`
- outputs:
  - `onsets`
  - `confidence`

Why this shape:

- it is clearly detector-like
- it maps cleanly to the repaired onset/event detector families in
  `ageo-atoms`, especially PPG/EMG detector semantics
- it is distinct from the existing `PeakDetector.detect` case without adding a
  new extraction mechanism

The case is intentionally synthetic. It does not need to mirror one exact
external library symbol. It needs to represent the repaired detector-family
pressure that prompted the structured-return lesson.

## Required Evidence Sources

In `ageo-atoms`:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- repaired detector families:
  - [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py)
  - [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py)

In `../ageo-matcher`:

- [return_shapes.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/return_shapes.py)
- [chunker.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/chunker.py)
- [emitter.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/emitter.py)
- [test_ingester_return_shapes.py](/Users/conrad/personal/ageo-matcher/tests/test_ingester_return_shapes.py)

## Implementation Strategy

### 1. Add one new explicit allowlist entry

Use the same mechanism already in place:

- exact `subject_name`
- exact `source_method`
- exact output-name set match

Do not add any looser matching or inference.

### 2. Reuse the existing `dict_field` emission path

The new case should require no new binding kinds. It should flow through the
already-validated `dict_field` path.

### 3. Extend tests in three layers

Add tests for:

- allowlist helper resolution for the new case
- chunker binding inference for the new case
- emitter rendering for the new case

Retain and reassert the existing conservative fallback behavior for
non-allowlisted subjects.

## Questions The Coding Worker Must Resolve

1. Does the new case fit best as `OnsetDetector.detect_events`, or is a nearby
   detector method name cleaner while staying distinct from `PeakDetector.detect`?
2. What output field names are the smallest stable pair that still reflects the
   repaired detector-family lesson?
3. Can the new tests share helper builders from the existing
   `test_ingester_return_shapes.py` suite without making it harder to read?

## Expected Write Scope

The worker should own only:

- `../ageo-matcher/sciona/ingester/return_shapes.py`
- `../ageo-matcher/tests/test_ingester_return_shapes.py`

The worker should not:

- edit smoke files
- edit regression harness files
- edit `ageo-atoms`
- touch unrelated local dirty files

If the worker discovers that a tiny edit to `chunker.py` or `emitter.py` is
strictly necessary, it should report that explicitly. But the intended path is
that no implementation change outside `return_shapes.py` is needed.

## Required Deliverables

1. one new explicit structured-return allowlist case
2. matching unit tests proving:
   - helper resolution
   - chunker inference
   - emitter extraction rendering
   - conservative fallback remains unchanged for non-allowlisted subjects
3. a short final worker report listing:
   - files changed
   - the new subject/method pair
   - the output fields

## Validation

Required matcher-side command:

- `pytest -q tests/test_ingester_return_shapes.py`

Optional if useful:

- run the broader related slice:
  - `pytest -q tests/test_ingester_return_shapes.py tests/test_ingester_chunker.py tests/test_ingester_emitter.py`

## Exit Criteria

This phase is complete when:

1. `return_shapes.py` contains one additional explicit allowlisted case
2. the new case is covered by focused matcher tests
3. non-allowlisted subjects still fall back conservatively
4. no broad heuristic extraction behavior was introduced

## Risks

- picking a case that is too close to the existing `PeakDetector.detect` path
  will add little value
- broadening the matching criteria would recreate the original failure mode
- introducing a new binding kind would make this phase too large

## Suggested Worker Slice

Implement one additional detector-like dict-return case only. Stop there even
if other candidates look easy.
