# Ingester Phase 5 Plan

This document is the implementation plan for **Phase 5: Ingest-Time
Deterministic Smoke Validation** from
[INGESTER_HARDENING_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_HARDENING_EXECUTION_PLAN.md).

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Add a lightweight matcher-side smoke gate that rejects obviously bad emitted
wrappers before they land in `ageo-atoms`.

The target is not to reproduce the full audit pipeline. The target is to catch
the highest-signal failures early:

- generated module cannot import
- generated atom has an obviously bad public call surface
- allowlisted deterministic probes fail on a basic positive path
- allowlisted deterministic probes fail on a basic negative path

## Why This Phase Is Needed

The trust-debt remediation work in `ageo-atoms` repeatedly exposed wrappers
that were broken in ways the matcher could have detected immediately:

- wrappers imported but raised on the simplest call
- wrappers had wrong result-shape assumptions
- wrappers passed mypy/ghost but still failed trivial runtime behavior checks

The audit stack in `ageo-atoms` is now strong enough to define what “obviously
bad” means. Phase 5 moves a small, deterministic subset of that bar upstream
into the ingester.

## Current Baseline In `../ageo-matcher`

The matcher already has post-generation validation stages, but they are mostly
structural:

- mypy validation
- ghost simulation
- monitor/marker/status reporting

What is missing is a narrow runtime smoke stage between “generated” and
“publishable”.

Relevant current files:

- `sciona/commands/ingest_cmds.py`
- `sciona/ingester/monitor.py`
- current validation helpers already used during ingest
- test files around ingest command orchestration and monitor state

## Non-Goals

Do not expand this phase into:

- full parity coverage
- the complete `ageo-atoms` runtime probe library
- stochastic or stateful end-to-end validation
- human review workflow
- repo-specific policy about trusted vs acceptable atoms

## Design Constraints

1. The smoke stage must be deterministic.
2. The smoke stage must be safe.
3. The smoke stage must be narrow enough to run during normal ingest.
4. Unknown or unsupported atoms must fail open into “no smoke coverage”, not
   into fabricated probes.
5. Smoke results must be recorded in monitor/provenance output so bad ingests
   are diagnosable.

## Target State

For allowlisted, deterministic targets, `sciona ingest` should:

1. emit files
2. run existing structural checks
3. run a small smoke validator against the staged output
4. record smoke results in status/summary/markers
5. either:
   - publish normally when smoke passes
   - fail cleanly before publication when smoke fails
   - record “not_applicable” or “no_probe” when no safe probe exists

The minimal acceptance result should be visible both in:

- `.ingest_status.json`
- `COMPLETED.json` / `FAILED.json`

## Recommended Architecture

Prefer a small dedicated matcher-side module, for example:

- `sciona/ingester/smoke.py`

That module should own:

- probe result model(s)
- allowlist matching
- safe import/execution harness
- probe runner entrypoint used by `ingest_cmds.py`

Keep command orchestration in:

- `sciona/commands/ingest_cmds.py`

Keep status reporting in:

- `sciona/ingester/monitor.py`

## Probe Policy

Phase 5 should not try to invent probes from arbitrary signatures.

Instead, define a narrow allowlist shape such as:

- fully qualified target symbol
- optional grouped output symbol name
- callable probe function
- expected result mode:
  - `pass`
  - `fail`
  - `not_applicable`

The first implementation should focus on pure, deterministic functions that
already have good evidence in `ageo-atoms`.

Recommended first candidates:

- small NumPy-style numerical helpers
- scipy top-level helpers already proven safe in the audit repo
- grouped sklearn image helpers that are pure and fixture-light

Do not include:

- stochastic wrappers
- stateful class wrappers
- wrappers requiring network, GPU, filesystem mutation, or long runtimes

## Failure Policy

Phase 5 should define explicit matcher behavior for three outcomes:

### Outcome A: `pass`

- smoke validation succeeded
- publish continues normally

### Outcome B: `fail`

- import failed, positive probe failed, or negative probe failed
- ingest should fail before final publication
- marker/status should explain that smoke validation rejected the output

### Outcome C: `not_applicable`

- no safe probe exists for the target
- ingest proceeds without smoke enforcement
- status records the reason

The default policy should be:

- enforce failure only when a probe is explicitly allowlisted and run
- do not punish unknown targets with synthetic failures

## Implementation Workstreams

### Workstream 1: Define Smoke Result Schema

Objective:

- make smoke validation results durable and inspectable

Tasks:

- define a small result schema for:
  - `status`
  - `probe_id`
  - `target_symbol`
  - `message`
  - optional positive/negative case details
- decide where the summary lives inside monitor status/marker payloads
- keep the schema stable and JSON-serializable

Likely files:

- `../ageo-matcher/sciona/ingester/monitor.py`
- new `../ageo-matcher/sciona/ingester/smoke.py`

### Workstream 2: Build A Narrow Smoke Runner

Objective:

- run safe deterministic probes against generated output

Tasks:

- implement import-by-path or import-by-output-dir loading for staged output
- define probe function contract
- run positive and negative cases for allowlisted targets
- convert exceptions into structured smoke results

Likely files:

- new `../ageo-matcher/sciona/ingester/smoke.py`
- possibly small import helpers if existing ones are reusable

### Workstream 3: Integrate Smoke Validation Into Ingest Flow

Objective:

- place the smoke runner in the right point of the ingest pipeline

Preferred ordering:

1. emit/stage generated files
2. run mypy / ghost as today
3. if those pass far enough to justify runtime probing, run smoke validation
4. if smoke fails, mark ingest failed and do not publish final artifacts
5. if smoke passes or is not applicable, continue publication

Tasks:

- decide whether smoke runs on staged files or published files
- ensure failures do not leave misleading completed markers
- ensure failure reasons survive into marker/status output

Likely files:

- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- `../ageo-matcher/sciona/ingester/monitor.py`

### Workstream 4: Add Matcher Tests

Objective:

- pin the behavior so Phase 5 remains stable

Required coverage:

- allowlisted passing probe
- allowlisted failing probe
- unsupported target -> `not_applicable`
- ingest command behavior when smoke validation fails before publication
- monitor/marker payload includes smoke result summary

Likely files:

- new `../ageo-matcher/tests/test_ingest_smoke.py`
- targeted command/monitor tests if better placed in existing files

### Workstream 5: Validate Against `ageo-atoms`

Objective:

- prove the new smoke gate is aligned with the audit repo’s acceptance bar

Recommended validation slice:

- one or two grouped sklearn image helpers
- one stable top-level numerical helper already known to pass deterministic
  audit probes

Validation questions:

- does matcher smoke pass on known-good atoms?
- does an intentionally broken wrapper fail before publication?
- is the recorded smoke provenance clear enough to debug from status files?

## Concrete File Targets

Primary write scope:

- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- `../ageo-matcher/sciona/ingester/monitor.py`
- `../ageo-matcher/sciona/ingester/smoke.py`

Primary test scope:

- `../ageo-matcher/tests/test_ingest_smoke.py`
- existing ingest command / monitor tests if needed

Secondary validation scope:

- temp grouped outputs or controlled smoke fixtures

`ageo-atoms` should not need code changes for Phase 5 itself beyond this plan
document unless a matcher smoke fixture is borrowed or mirrored there.

## Key Decisions The Implementing Agent Must Make

1. Should smoke run before or after publication to the final output dir?
2. Should smoke import from staged files in `.partial/` or from the final
   output package?
3. What is the minimal allowlist format that stays maintainable?
4. How much of the `ageo-atoms` runtime-probe logic can be reused safely
   without coupling the repos too tightly?
5. Should smoke failure mark the ingest as fully failed, or as a completed run
   with a rejected-publication summary?

Recommended answers unless implementation evidence suggests otherwise:

- run smoke before final publication when practical
- import from staged output if possible; otherwise publish to temp and validate
- keep the allowlist matcher-local and small
- treat allowlisted smoke failures as true ingest failures

## Verification Commands

Minimum:

- targeted matcher tests for smoke validation
- command-level tests covering publication/failure behavior

Recommended:

- `pytest -q` on the Phase 5 matcher tests
- one temp-dir ingest smoke run against a known safe function/grouped family
- one intentionally broken fixture that proves failed smoke blocks publication

Record for the smoke run:

- invoked command
- target symbol
- output dir
- smoke result summary
- whether publication occurred

## Exit Criteria

Phase 5 is complete when:

- matcher has a deterministic smoke-validation stage
- allowlisted targets can pass or fail with structured results
- smoke failures stop publication and produce clear status/marker evidence
- unsupported targets record `not_applicable` without synthetic failure
- at least one real ingest slice validates against known-good output

## Risks

- too-broad probe ambition could recreate the audit system inside the matcher
- import-time side effects could make probes flaky if target selection is loose
- publishing semantics can become messy if validation runs after final write
- poorly chosen probes could reject acceptable wrappers for incidental reasons

## Recommended First Implementation Slice

Start with the smallest useful vertical slice:

1. add `smoke.py` with a tiny local allowlist
2. support one passing grouped-family case
3. support one failing fixture case
4. wire smoke summary into monitor/marker payloads
5. stop publication on allowlisted failures

That is enough to validate the architecture before broadening probe coverage.
