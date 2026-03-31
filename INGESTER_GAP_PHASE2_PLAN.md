# Ingester Gap Phase 2 Plan

This document is the detailed implementation plan for **Phase 2: Smoke
Coverage Expansion** from
[INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md).

It should be read together with:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Objective

Expand matcher-side ingest-time smoke validation to cover a few more
deterministic repaired families from `ageo-atoms`, without turning the matcher
into a second full audit system.

This phase should pay down the `partially_covered` rows in the crosswalk for:

- ingest-time smoke rejection
- grouped-family rollout breadth
- detector-family rollout breadth

## Why This Phase Comes Next

Phase 1 established the target map. The highest-leverage remaining matcher gap
is that smoke validation exists architecturally but still protects only a thin
canary surface.

The next implementation should therefore:

- extend smoke coverage only where `ageo-atoms` already proved deterministic
  value
- add focused matcher tests
- avoid broad family inference or stochastic probes

## Scope

In scope:

- `../ageo-matcher/sciona/ingester/smoke.py`
- `../ageo-matcher/tests/test_ingest_smoke.py`
- a small number of new allowlisted deterministic probe cases
- grouped-family and detector-family positive/negative cases
- pass/fail/not-applicable behavior where it materially changes

Out of scope:

- changes to `ageo-atoms` runtime probes
- broad parity replication in the matcher
- structured-return allowlist changes
- grouped-ingest workflow changes
- regression harness changes

## Candidate Families

Choose only families that meet all of these:

1. already repaired or validated in `ageo-atoms`
2. deterministic and cheap to run
3. safe to import and execute matcher-side
4. express a lesson the current smoke layer does not already cover

### Required first family slice

Start with the BioSPPy detector families already represented in
[runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py):

- ECG detector wrappers
- PPG detector wrappers
- EMG onset detector wrappers

These are the best next targets because they directly reflect the recent repair
history and are already backed by deterministic repo-local probes.

### Optional second slice if it fits naturally

One additional grouped-family or numerical helper slice is acceptable only if:

- it reuses existing probe helpers cleanly
- it does not bloat `smoke.py`

Do not add a second slice if it makes the phase materially larger.

## Required Evidence Sources

In `ageo-atoms`:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
- detector family modules:
  - [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py)
  - [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py)
  - [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py)

In `../ageo-matcher`:

- [smoke.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/smoke.py)
- [test_ingest_smoke.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_smoke.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py) only if status integration needs minor extension

## Implementation Strategy

### 1. Mirror only the smallest useful detector cases

Do not copy the `ageo-atoms` probe library wholesale.

Instead:

- reuse the same synthetic data shapes where helpful
- reduce each family to a minimal positive-path and negative-path matcher probe
- keep assertions structural, not numerically overfit

Examples:

- ECG detector:
  - positive: returns a monotonic non-empty peak index array
  - negative: rejects a missing signal or bad sampling rate
- PPG detector:
  - positive: returns monotonic event/onset indices
  - negative: rejects a missing signal or bad rate
- EMG detector:
  - positive: returns a valid monotonic onset index array or valid empty array
  - negative: rejects a missing signal

### 2. Keep matcher probe semantics intentionally narrower than audit probes

The matcher’s job is early rejection of obviously bad staged outputs, not full
parity.

So the smoke checks should validate only:

- imports work
- the function can run on a sane canonical input
- the output has the expected broad structural form
- a simple negative path still raises

### 3. Keep the allowlist explicit

Add explicit probe registrations for:

- detector symbol
- package/family basename where needed

Do not infer entire families from names.

## Questions The Coding Worker Must Resolve

1. Which existing helper functions in `smoke.py` can be reused for monotonic
   index-array validation?
2. Should detector probes dispatch by function symbol only, or by both symbol
   and package basename?
3. Is it cleaner to add a small helper family for detector signal synthesis
   inside `smoke.py`, or inline one minimal synthetic fixture per probe?
4. Does any new pass/fail detail shape need test updates beyond the new cases?

## Expected Write Scope

The worker should own only:

- `../ageo-matcher/sciona/ingester/smoke.py`
- `../ageo-matcher/tests/test_ingest_smoke.py`

The worker should not:

- edit `ageo-atoms`
- touch regression harness files
- touch return-shape files
- touch unrelated local dirty files in either repo

## Required Deliverables

1. smoke allowlist expansion for the selected BioSPPy detector slice
2. focused matcher tests for:
   - direct smoke-runner success on detector probes
   - direct smoke-runner failure where appropriate
   - ingest-time publication behavior for at least one allowlisted detector case
3. a short final report from the worker listing:
   - files changed
   - symbols added
   - any detector families intentionally deferred

## Validation

Required matcher-side commands:

- `pytest -q tests/test_ingest_smoke.py`

Optional if useful and cheap:

- one small import sanity check for `sciona.ingester.smoke`

No `ageo-atoms` tests are required in this phase.

## Exit Criteria

This phase is complete when:

1. matcher smoke coverage clearly extends beyond grouped sklearn images and the
   current numerical canaries
2. at least one repaired detector-style family is protected matcher-side
3. the new probes remain deterministic, cheap, and narrow
4. the patch stays within the expected write scope

## Risks

- making detector probes too numerically specific will create brittle tests
- trying to cover every detector family at once will bloat `smoke.py`
- accidentally mirroring full audit semantics inside the matcher would be a
  design mistake

## Suggested Worker Slice

Implement the smallest useful detector slice:

- one ECG detector pair
- one PPG detector pair
- one EMG detector pair

If that lands cleanly and still feels small, include one more detector symbol
from the same helper family. Otherwise stop there.
