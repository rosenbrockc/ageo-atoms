# Ingester Phase 4 Plan

This document is the implementation plan for **Phase 4: Package Scope And
Output Layout Hardening** from
[INGESTER_HARDENING_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_HARDENING_EXECUTION_PLAN.md).

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Make grouped/module-family output a first-class ingest path instead of relying
on one-symbol-per-output-directory as the de facto default.

The immediate target is not a full migration of old repositories. It is to
make the matcher capable of publishing coherent grouped artifact sets without
awkward manual conventions.

## Why This Phase Is Needed

The trust-debt cleanup in `ageo-atoms` showed that many ingest-derived atoms
were over-fragmented into single-atom subpackages. That created:

- excessive directory sprawl
- duplicated sidecars
- harder family-level maintenance
- more consolidation work after the fact

We already proved that grouped output works better in practice with the sklearn
`images/` family, but that grouping currently depends on how `--output` is
manually chosen rather than on first-class ingest behavior.

## Current Baseline In `../ageo-matcher`

Relevant files:

- `sciona/commands/ingest_cmds.py`
- `sciona/ingester/monitor.py`
- `sciona/ingester/graph.py`
- `tests/` around ingest command behavior, monitor/publish, and regression
  harnesses if needed

Current behavior:

- `sciona ingest` writes to one `output_dir`
- the command stages and publishes a fixed artifact set:
  - `atoms.py`
  - `state_models.py`
  - `witnesses.py`
  - `cdg.json`
  - `matches.json`
- monitor/publish logic assumes one output directory per ingest invocation
- grouping is possible only because the caller can choose a broader `--output`
  path manually

That means grouped output is *possible*, but it is not yet modeled as a
deliberate ingest concept.

## Non-Goals

Do not broaden this phase into:

- signature fidelity
- output-binding inference
- witness semantic quality
- runtime smoke-validation framework
- bulk migration of existing `ageo-atoms` directories

## Target State

The matcher should support a clear distinction between:

- **symbol-scoped output**
  one ingest target, one output package
- **group-scoped output**
  multiple related targets land in one coherent package/family directory

Phase 4 does not need to support multi-symbol ingest in one command if that is
too large. It does need to make grouped publication deliberate and coherent.

## Design Direction

Prefer minimal, explicit changes:

1. Keep `--output` as the concrete publication path.
2. Add explicit grouped-output metadata/policy instead of relying only on
   naming convention.
3. Ensure the staged artifact set remains coherent when the target directory is
   a family/module package rather than a one-symbol leaf.
4. Make grouped output discoverable in matcher-side provenance and/or publish
   summaries.

## Key Open Decision

There are two reasonable implementation levels for Phase 4:

### Option A: Lightweight grouping support

- keep one target per ingest command
- formalize that `--output` may refer to a family package
- add explicit scope metadata like `output_scope = symbol | family`
- ensure publish/provenance/reporting reflect that scope

### Option B: True multi-target grouped ingest

- one command ingests multiple related targets into one family package
- command needs to merge or append generated artifacts coherently

For this phase, **Option A is the right target** unless the implementation
turns out to be trivially close to Option B. The goal is to make grouping
first-class, not to redesign the entire ingest command.

## Implementation Workstreams

### Workstream 1: Define Grouped Output Semantics

Objective:

- make the matcher’s concept of grouped output explicit

Tasks:

- define output scope terms in code/docs:
  - `symbol`
  - `family` or `module_family`
- decide where that metadata lives:
  - ingest command config
  - monitor/provenance summary
  - publish summary payload
- ensure the command can report whether a run was symbol-scoped or grouped

Likely files:

- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- possibly `../ageo-matcher/sciona/ingester/monitor.py`

### Workstream 2: Harden Publication For Grouped Targets

Objective:

- ensure grouped output directories still get a coherent artifact set

Tasks:

- inspect staging/publication assumptions in:
  - `ingest_cmds.py`
  - `monitor.py`
  - `graph.py`
- verify that grouped outputs do not accidentally inherit leaf-only assumptions
- add any small adjustments needed so grouped packages publish the same stable
  artifact surface without hidden path assumptions

Potential issues to inspect:

- output-dir defaulting logic
- monitor marker/status messages that imply one symbol per dir
- helper code that derives names from output-dir basename

### Workstream 3: Add Explicit Group-Scope API Surface

Objective:

- expose grouping as an intentional matcher feature instead of a manual caller
  convention

Possible implementation paths:

- add a flag like `--output-scope {symbol,family}`
- or add a narrower flag like `--family-output`
- or record family grouping automatically when `--output` is not the default
  symbol path

Preferred path:

- a small explicit flag is better than magic inference

This phase should keep the CLI change small and backwards-compatible.

Likely file:

- `../ageo-matcher/sciona/commands/ingest_cmds.py`

### Workstream 4: Matcher Tests

Objective:

- pin grouped-output behavior in matcher tests

Required coverage:

- command/output planning test for grouped scope metadata
- monitor/publish test ensuring grouped target paths still publish the standard
  artifact set
- regression test for a grouped output directory path that is not equal to the
  target symbol name

If CLI flags change:

- add command parsing tests too

Likely files:

- existing ingest command tests if present
- otherwise a focused new test file for ingest command output behavior
- possibly monitor tests if they already exist

### Workstream 5: Practical Validation In `ageo-atoms`

Objective:

- prove grouped output remains the recommended operational path

Recommended smoke target:

- a grouped sklearn family directory such as `ageoa/sklearn/images/`

Validation should answer:

- does the matcher now report grouped/family scope explicitly?
- does the grouped output still publish the stable artifact set cleanly?
- does the resulting directory shape align with the repo’s grouped-family
  policy?

## Concrete File Targets

Primary write scope:

- `../ageo-matcher/sciona/commands/ingest_cmds.py`
- `../ageo-matcher/sciona/ingester/monitor.py`

Secondary write scope only if needed:

- `../ageo-matcher/sciona/ingester/graph.py`
- matcher-side tests covering ingest command/output behavior

`ageo-atoms` should not need code changes for this phase beyond this plan
document unless a smoke-validation helper is added locally.

## Verification Commands

Minimum:

- matcher tests covering the changed ingest/output behavior

Recommended:

- targeted `pytest -q ...` on the changed matcher test files
- one grouped-output ingest smoke run into a temp directory

If a grouped smoke run is used, record:

- invoked command
- chosen `--output`
- resulting published files
- whether grouped scope metadata was visible

## Exit Criteria

Phase 4 is complete when:

1. grouped output is represented intentionally in matcher behavior
2. publication/publish summaries remain coherent for grouped directories
3. matcher tests cover the grouped-path behavior
4. a grouped smoke ingest succeeds with the expected stable artifact set

## Risks

- adding too much CLI surface too early may overcomplicate usage
- grouped output metadata may be duplicated later by provenance work
- existing tooling may implicitly assume symbol-scoped default output names

## Recommended Coding Order

1. formalize grouped output semantics
2. patch ingest command and monitor/publish behavior
3. add matcher regression tests
4. run a grouped-output smoke ingest
5. only then widen the feature if necessary
