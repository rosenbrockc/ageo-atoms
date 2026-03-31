# Ingester Gap Phase 5 Plan

This document is the detailed implementation plan for **Phase 5: Grouped
Ingest Ergonomics** from
[INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_GAP_CLOSURE_EXECUTION_PLAN.md).

It should be read together with:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- [INGESTER_RISK_LESSONS_AUDIT.md](/Users/conrad/personal/ageo-atoms/INGESTER_RISK_LESSONS_AUDIT.md)

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Objective

Add one bounded ergonomic improvement for grouped ingest: make family-scope
replacement of an already-populated output directory explicit and visible,
instead of allowing silent overwrite behavior.

This phase should pay down the `partially_covered` row in the crosswalk for:

- grouped publication for grouped package families

It should also materially improve the `tempo_jl/offsets` grouped-family rollout
story by making family ingestion safer to operate.

## Why This Phase Is Bounded This Way

The real grouped-ingest gap is not lack of publication metadata. That part is
already implemented.

The gap is that the current workflow is still too easy to misuse:

- a `family`-scope ingest into an existing grouped output directory can replace
  prior family artifacts without an explicit operator acknowledgement
- monitor output does not clearly distinguish a fresh family publication from a
  replacement of an existing grouped package

Trying to solve “true multi-target grouped ingest” in this phase would be too
large. This phase should instead add a safety rail and a clearer workflow.

## Scope

In scope:

- one CLI/operator-facing guardrail for family-scope publication
- monitor/status visibility for whether an existing family output is being
  replaced
- tests covering:
  - safe failure when a grouped family dir already exists
  - explicit opt-in path for replacing an existing grouped family dir

Likely implementation files:

- `../ageo-matcher/sciona/cli.py`
- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- `../ageo-matcher/sciona/ingester/monitor.py`
- `../ageo-matcher/tests/test_ingest_output_scope.py`
- `../ageo-matcher/tests/test_cli_ingest_status.py` only if surface output
  changes materially

Out of scope:

- merging generated `atoms.py` / `witnesses.py` across multiple ingests
- batch orchestration
- regression harness expansion
- smoke validation changes
- return-shape changes

## Recommended Feature Shape

Add one explicit family replacement flag, for example:

- `--allow-family-replace`

Behavior:

1. If `output_scope == family` and the target directory already contains one or
   more previously published canonical artifacts, ingest should fail early by
   default with an actionable message.
2. If `--allow-family-replace` is present, ingest may proceed.
3. Monitor/status/marker output should record whether the run:
   - created a fresh family output
   - replaced an existing family output

This gives operators a safer grouped workflow without inventing merge logic.

## Why This Is The Right First Ergonomic Improvement

It is:

- narrowly scoped
- easy to explain
- directly relevant to grouped families such as:
  - [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images)
  - [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets)
- useful even before any future multi-target grouped-ingest workflow exists

## Required Evidence Sources

In `ageo-atoms`:

- [INGESTER_LESSON_CROSSWALK.md](/Users/conrad/personal/ageo-atoms/INGESTER_LESSON_CROSSWALK.md)
- grouped family examples:
  - [sklearn/images](/Users/conrad/personal/ageo-atoms/ageoa/sklearn/images)
  - [tempo_jl/offsets](/Users/conrad/personal/ageo-atoms/ageoa/tempo_jl/offsets)

In `../ageo-matcher`:

- [cli.py](/Users/conrad/personal/ageo-matcher/sciona/cli.py)
- [ingest_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/ingest_cmds.py)
- [monitor.py](/Users/conrad/personal/ageo-matcher/sciona/ingester/monitor.py)
- [test_ingest_output_scope.py](/Users/conrad/personal/ageo-matcher/tests/test_ingest_output_scope.py)
- [test_cli_ingest_status.py](/Users/conrad/personal/ageo-matcher/tests/test_cli_ingest_status.py)

## Implementation Strategy

### 1. Detect existing grouped-family publication surface

Use a narrow definition of “existing family output”:

- target dir exists
- `output_scope == family`
- one or more canonical published artifacts already exists in the target dir

Canonical artifacts should be derived from the existing standard surface, not
from arbitrary extra files.

### 2. Fail closed by default

If a grouped family dir already exists and the operator has not supplied the
explicit replacement flag, ingest should fail before any publication occurs.

The error message should be actionable and mention the explicit flag.

### 3. Record replacement intent and result in monitor state

The monitor/publication summary should expose enough information to tell:

- whether the family output previously existed
- whether replacement was explicitly allowed
- whether the run was a fresh family publish or a replacement publish

This should be visible in status and completion/failure markers.

## Questions The Coding Worker Must Resolve

1. Where should the preflight grouped-family surface check live:
   `_cmd_ingest`, `IngestMonitor.start`, or a small helper?
2. What is the narrowest reliable rule for “already-populated family dir”?
3. Which publication summary fields should be added so tests and operators can
   distinguish `fresh_family_publish` from `family_replace`?
4. Does the new flag belong only on `ingest`, or should `ingest-status` surface
   the replacement mode too?

## Expected Write Scope

The worker should own only:

- `../ageo-matcher/sciona/cli.py`
- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- `../ageo-matcher/sciona/ingester/monitor.py`
- `../ageo-matcher/tests/test_ingest_output_scope.py`
- optionally `../ageo-matcher/tests/test_cli_ingest_status.py` if the output
  surface changes materially

The worker should not:

- edit smoke files
- edit regression harness files
- edit `ageo-atoms`
- touch unrelated local dirty files

## Required Deliverables

1. one explicit grouped-family replacement flag
2. one preflight guard that blocks silent replacement of an existing grouped
   family dir
3. monitor/publication summary fields capturing fresh-vs-replacement family
   publication
4. focused matcher tests for:
   - parser support for the new flag
   - default failure on existing grouped family dir
   - explicit replacement success path

## Validation

Required matcher-side command:

- `pytest -q tests/test_ingest_output_scope.py`

Optional if the status output changes:

- `pytest -q tests/test_cli_ingest_status.py`

## Exit Criteria

This phase is complete when:

1. grouped-family replacement is no longer silent by default
2. operators have an explicit opt-in path for replacing an existing grouped
   family package
3. monitor/status output records the replacement context
4. the patch remains narrowly focused on grouped-ingest ergonomics rather than
   expanding into merge/orchestration features

## Risks

- trying to support true merge semantics will explode scope
- blocking too broadly could break normal symbol-scope ingests
- the replacement check must ignore monitor-side operational files and focus on
  canonical artifacts

## Suggested Worker Slice

Implement only the explicit replacement guard and monitor/status visibility.
Do not attempt grouped-file merging or batch-family ingest in this phase.
