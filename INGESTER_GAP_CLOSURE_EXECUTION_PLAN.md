# Ingester Gap-Closure Execution Plan

This document turns the remaining findings from
[INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)
into a bounded multi-phase execution plan.

It is written for planning agents that will expand and execute **one phase at a
time**. Each phase below is intentionally self-contained and includes enough
context to let the next planner succeed without reconstructing the recent
history from scratch.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Why This Plan Exists

The main hardening program is done. The audit result is now:

- core ingester lessons are absorbed
- major architecture gaps are closed
- remaining gaps are mostly about rollout depth, traceability, and a few narrow
  follow-on capabilities

Those remaining gaps are:

1. cross-repo traceability is weak
2. matcher-side smoke coverage is still much narrower than the repo-local audit
   oracle
3. curated matcher regression coverage is still small relative to the repair
   history in `ageo-atoms`
4. structured-return knowledge is intentionally narrow and may need a few more
   explicit allowlisted cases
5. grouped output publication exists, but grouped ingest ergonomics are still
   behind the desired workflow

This plan is for closing those gaps without reopening broad heuristic behavior.

## Working Rules

1. `ageo-atoms` remains the acceptance oracle.
2. Prefer narrow explicit coverage over broad inference.
3. Every phase must have a bounded write scope.
4. Do not silently broaden matcher behavior across all families just to reduce
   audit counts.
5. Every phase should end with:
   - matcher-side tests
   - at least one realistic validation slice
   - a clear statement of what gap is now closed versus still partial

## Recommended Execution Order

The recommended order is:

1. Phase 1: Cross-Repo Lesson Traceability
2. Phase 2: Smoke Coverage Expansion
3. Phase 3: Regression Corpus Expansion
4. Phase 4: Structured-Return Allowlist Expansion
5. Phase 5: Grouped Ingest Ergonomics
6. Phase 6: Final Cross-Repo Coverage Audit

This order matters.

- Phase 1 gives the durable map needed to choose the right follow-on families.
- Phases 2 and 3 deepen coverage without changing matcher architecture.
- Phase 4 adds only narrowly justified structured-return knowledge after the
  regression surface is stronger.
- Phase 5 is the only phase that changes workflow ergonomics.
- Phase 6 is the closeout audit that determines whether the remaining gaps are
  truly paid down.

## Current Baseline

Already implemented:

- signature fidelity hardening
- conservative return/output binding hardening
- witness/decorator hardening
- grouped publication metadata and monitor surface
- ingest-time deterministic smoke gate
- initial smoke allowlist for grouped sklearn images and narrow numerical
  helpers
- curated regression harness cases for grouped sklearn images and detector-like
  structured output
- narrow structured-return allowlist for `PeakDetector.detect`

Primary evidence in `../ageo-matcher`:

- `sciona/ingester/emitter.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/return_shapes.py`
- `sciona/ingester/smoke.py`
- `sciona/ingester/regression_harness.py`
- `sciona/commands/ingest_cmds.py`
- `sciona/ingester/monitor.py`

Primary evidence in `ageo-atoms`:

- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)
- [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
- [AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md)
- repaired family examples such as:
  - [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images)
  - [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py)
  - [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py)
  - [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py)
  - [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets)

## Phase 1: Cross-Repo Lesson Traceability

### Objective

Create a durable mapping from major repaired `ageo-atoms` families to matcher
coverage.

### Problem This Phase Solves

Right now we can describe the lessons verbally, but we do not have a stable
artifact answering:

- which repaired families drove which matcher changes
- which matcher tests protect those lessons
- which repaired families still have no corresponding matcher probe/regression

That makes later coverage work less targeted than it should be.

### Scope

In scope:

- a crosswalk artifact in `ageo-atoms`
- one row per major repaired family or repair theme
- links to:
  - `ageo-atoms` repair evidence
  - matcher implementation files
  - matcher tests and smoke cases
- classification such as:
  - `covered`
  - `partially_covered`
  - `uncovered`

Out of scope:

- adding new matcher behavior
- changing audit scoring
- broad repo-wide automated extraction of all repair history

### Likely Files To Inspect

In `ageo-atoms`:

- `INGESTER_RISK_LESSONS_AUDIT.md`
- `AUDIT_INGEST.md`
- `scripts/auditlib/runtime_probes.py`
- recent repaired families under `ageoa/`

In `../ageo-matcher`:

- `sciona/ingester/smoke.py`
- `sciona/ingester/regression_harness.py`
- `tests/test_ingest_smoke.py`
- `tests/test_ingest_regression_harness.py`
- `tests/test_ingester_return_shapes.py`
- `tests/test_ingester_emitter.py`

### Questions The Next Planner Must Answer

1. What is the right granularity for one “lesson row”: family, repair theme, or
   individual atom?
2. Should the crosswalk live as Markdown, JSON, or both?
3. Which families are the minimum high-value set to track first?
4. How should grouped families like `sklearn/images` and `tempo_jl/offsets` be
   represented?

### Required Deliverables

- a dedicated crosswalk document or structured artifact
- an explicit initial mapped set covering the highest-signal repaired families
- clear labels for uncovered and partially covered lessons

### Exit Criteria

- a future worker can point to one artifact to know which repair families still
  need matcher follow-up
- the next phases can select targets from this crosswalk instead of from memory

### Phase Risks

- trying to make the crosswalk exhaustive will stall the phase
- conflating audit-only concerns with matcher concerns will reduce signal

## Phase 2: Smoke Coverage Expansion

### Objective

Expand matcher-side smoke validation to cover a few more deterministic families
that are already proven safe in `ageo-atoms`.

### Problem This Phase Solves

The smoke gate works, but its allowlist is still tiny compared with the runtime
probe surface that actually reduced risk in `ageo-atoms`.

### Scope

In scope:

- a narrow set of additional deterministic families
- matcher-local positive/negative probes only
- grouped-family and top-level function coverage
- monitor/status behavior on allowlisted failures

Out of scope:

- copying the full `ageo-atoms` runtime probe library
- stochastic or expensive probe families
- parity-equivalent replication inside the matcher

### Candidate First Targets

Only use families already proven deterministic and useful in `ageo-atoms`.
Good candidates include:

- BioSPPy detector helpers with simple deterministic inputs
  - ECG detector wrappers
  - PPG detector wrappers
  - EMG onset detector wrappers
- stable numerical helpers that already passed the audit cleanup process

### Likely Files To Inspect

In `../ageo-matcher`:

- `sciona/ingester/smoke.py`
- `tests/test_ingest_smoke.py`

In `ageo-atoms`:

- `scripts/auditlib/runtime_probes.py`
- `tests/test_audit_runtime_probes.py`

### Questions The Next Planner Must Answer

1. Which `ageo-atoms` probe cases are safe to mirror in a reduced matcher form?
2. How many new families fit into one bounded phase without bloating
   `smoke.py`?
3. Should grouped-family probes stay symbol-centric or gain a family-level
   aggregate path?

### Required Deliverables

- smoke allowlist expansion
- focused matcher smoke tests
- at least one realistic grouped-family or detector-family smoke slice

### Exit Criteria

- matcher smoke coverage clearly extends beyond sklearn images and the current
  numerical canaries
- at least one repaired detector-style family is now protected matcher-side

### Phase Risks

- overly broad smoke expansion will turn the matcher into a second audit system
- fragile or noisy probes will reduce trust in the gate

## Phase 3: Regression Corpus Expansion

### Objective

Turn more of the actual repair history into curated matcher regression cases.

### Problem This Phase Solves

The current regression harness is valuable but small. It covers grouped images
and one detector-like structured-output case, but many learned families still
exist only as historical knowledge in `ageo-atoms`.

### Scope

In scope:

- new small curated fixtures under matcher tests
- corresponding golden artifacts
- explicit comments linking fixtures to real repaired families
- harness-level assertions around grouped output, witness shape, return shape,
  and publication summaries

Out of scope:

- live ingestion of external libraries during test
- giant fixture suites
- broad end-to-end replay of the whole atoms repo

### Candidate Families

Choose only a few compact representatives, for example:

- BioSPPy detector-like family with structured output pressure
- grouped helper family beyond sklearn images
- one FFI or stateful family only if it can be represented cheaply

### Likely Files To Inspect

In `../ageo-matcher`:

- `sciona/ingester/regression_harness.py`
- `tests/test_ingest_regression_harness.py`
- `tests/fixtures/ingest_regression/`
- `tests/golden/ingest_regression/`

In `ageo-atoms`:

- repaired family modules and sidecars that motivated the fixture shape

### Questions The Next Planner Must Answer

1. Which repaired families are both high-signal and cheap to mirror?
2. Which artifact assertions are stable enough for goldens?
3. Should some crosswalk rows from Phase 1 become mandatory regression targets?

### Required Deliverables

- one or more new curated fixtures and goldens
- harness test updates
- explicit mapping from each new fixture back to the originating repair lesson

### Exit Criteria

- the matcher regression suite covers more than the original canary families
- future regressions in those learned areas would trip matcher tests directly

### Phase Risks

- unstable goldens will create churn
- overly synthetic fixtures may stop representing the real repair pressure

## Phase 4: Structured-Return Allowlist Expansion

### Objective

Add a few more explicit structured-return cases without reintroducing broad
heuristics.

### Problem This Phase Solves

The conservative fallback is now correct by default, but some repaired families
still have known structured outputs that are not yet encoded matcher-side.

### Scope

In scope:

- a small expansion of the explicit allowlist in matcher return-shape handling
- exact-match activation only
- regression tests for allowlisted and non-allowlisted behavior

Out of scope:

- field-name inference
- tuple-slot guessing from names alone
- automatic discovery of structured returns

### Candidate First Cases

Only choose cases that already meet all of these:

- repaired successfully in `ageo-atoms`
- output fields are stable and unambiguous
- deterministic test fixture is cheap to maintain

### Likely Files To Inspect

In `../ageo-matcher`:

- `sciona/ingester/return_shapes.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `tests/test_ingester_return_shapes.py`

In `ageo-atoms`:

- repaired detector or helper families with explicit output fields

### Questions The Next Planner Must Answer

1. Which repaired structured-output cases are strong enough to encode?
2. Should allowlist keys remain subject/method based, or also use atom ids?
3. What is the smallest useful expansion that still proves the phase matters?

### Required Deliverables

- allowlist expansion
- explicit matcher tests
- one realistic validation slice tied back to a repaired family

### Exit Criteria

- at least one additional real structured-return case is covered matcher-side
- conservative fallback remains the default for everything else

### Phase Risks

- broadening the allowlist too quickly will recreate the original problem
- weakly justified cases will encode repo accidents instead of stable APIs

## Phase 5: Grouped Ingest Ergonomics

### Objective

Close the gap between “grouped publication exists” and “grouped ingest is a
smooth production workflow.”

### Problem This Phase Solves

Today the matcher can publish grouped outputs and track `family` scope, but the
operator experience for intentionally building a family package from several
targets is still limited.

### Scope

In scope:

- workflow/CLI ergonomics for grouped ingest
- directory-scope and monitor-summary behavior for grouped targets
- publication safety when a family package is built incrementally
- tests covering grouped workflow semantics

Out of scope:

- large scheduler/orchestration features
- redesigning the core ingestion planner
- batch ingest support for arbitrary repo-wide jobs

### Likely Files To Inspect

In `../ageo-matcher`:

- `sciona/cli.py`
- `sciona/commands/ingest_cmds.py`
- `sciona/ingester/monitor.py`
- `tests/test_ingest_output_scope.py`
- possibly regression harness coverage if a grouped workflow becomes stable

In `ageo-atoms`:

- grouped package examples:
  - [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images)
  - [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets)

### Questions The Next Planner Must Answer

1. What concrete grouped-ingest workflow is currently painful?
2. Is the right abstraction repeated symbol ingests into one family dir, or a
   true multi-target ingest command?
3. How should monitor artifacts represent partially built grouped families?

### Required Deliverables

- one bounded ergonomic improvement to grouped ingest flow
- tests for monitor/publication behavior under that flow
- clear documentation of the supported grouped-ingest path

### Exit Criteria

- grouped family ingest is easier and less ad hoc than it is today
- users no longer need to infer grouped publication behavior from scattered
  options and conventions

### Phase Risks

- trying to design full batch orchestration will explode scope
- changing publication semantics carelessly could endanger existing outputs

## Phase 6: Final Cross-Repo Coverage Audit

### Objective

Re-audit `ageo-atoms` versus `../ageo-matcher` after the follow-on phases land.

### Problem This Phase Solves

The earlier audit established that the architecture was mostly covered and the
remaining gaps were rollout-oriented. After these phases, we need a new audit
to verify whether those rollout gaps are truly reduced.

### Scope

In scope:

- refresh the cross-repo audit
- compare before/after status for each remaining gap
- classify remaining items as:
  - resolved
  - intentionally partial
  - deferred

Out of scope:

- new matcher feature work unless a blocker is discovered

### Likely Files To Inspect

In `ageo-atoms`:

- `INGESTER_RISK_LESSONS_AUDIT.md`
- the Phase 1 crosswalk artifact

In `../ageo-matcher`:

- smoke, return-shape, grouped-ingest, and regression harness files touched by
  earlier phases

### Questions The Next Planner Must Answer

1. Which metrics should be used to say rollout depth materially improved?
2. Which gaps are acceptable to leave intentionally narrow?
3. Should the audit output update the existing audit doc or produce a new one?

### Required Deliverables

- a refreshed cross-repo audit document
- explicit status for each gap from this plan
- recommendations for any future low-priority follow-up

### Exit Criteria

- there is a durable closeout artifact showing what was actually paid down
- future agents can distinguish completed refinement from intentionally deferred
  rollout work

### Phase Risks

- closing the plan without a real audit will recreate the current ambiguity

## Suggested First Planner Handoff

The next planning agent should start with **Phase 1: Cross-Repo Lesson
Traceability**.

That planner should receive:

- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)
- this execution plan
- access to both repos

Its job is to produce a **detailed Phase 1 implementation plan only**, with:

- the target artifact format
- the minimum family set to map first
- the exact files it will inspect
- the exit criteria for saying the crosswalk is good enough to drive later
  implementation phases
