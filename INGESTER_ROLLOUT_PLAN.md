# Ingester Rollout Plan

This document covers the **next modifications after the five ingester
hardening phases**. The hardening work is complete; this plan focuses on
turning that new matcher behavior into durable regression coverage and broader
operational value.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Why This Plan Exists

The hardening phases fixed the main architecture problems exposed by trust-debt
remediation in `ageo-atoms`:

- signature drift
- bad return binding defaults
- bad witness/decorator emission
- fragmented grouped-output handling
- no early smoke gate

What remains is rollout depth, not architecture:

- the smoke allowlist is still tiny
- the matcher lacks stable end-to-end regression fixtures for the families that
  taught us the lessons
- some return-shape cases still need explicit, evidence-backed allowlisting

The goal here is to close that gap without rebuilding the full audit system
inside the matcher.

## Plan Structure

This rollout is split into three implementation tracks that can be executed in
order. They are intentionally bounded so a worker can take them over without
reconstructing the full repo history.

1. Expand deterministic smoke coverage for known-safe families
2. Add end-to-end ingest regression fixtures for the families that drove the
   recent repairs
3. Add a narrow return-shape knowledge layer for explicitly known structured
   outputs

## Working Rules

1. `ageo-atoms` remains the acceptance oracle.
2. Prefer narrow allowlists over broad inference.
3. Do not add matcher-side smoke probes for stochastic, stateful, or costly
   families unless they are clearly deterministic and safe.
4. Do not broaden return extraction heuristics globally; only add structured
   handling where evidence is explicit.
5. Each workstream must land matcher-side tests and at least one realistic
   validation slice.

## Workstream 1: Expand Phase 5 Smoke Coverage

### Objective

Broaden matcher-side smoke validation beyond the initial architectural canary.

### Immediate Targets

Start with families that already proved safe and useful in `ageo-atoms`:

- grouped sklearn image helpers
  - `extract_patches_2d`
  - `reconstruct_from_patches_2d`
  - `img_to_graph`
  - `grid_to_graph`
- stable top-level NumPy/SciPy helpers already paid down by deterministic audit
  work
  - FFT helpers
  - sorting/search helpers
  - selected `scipy.integrate`, `scipy.stats`, and `scipy.optimize` functions

### Scope

In scope:

- matcher-local smoke allowlist expansion
- reusable probe helpers for grouped pure functions
- pass/fail/not-applicable coverage in matcher tests
- one or two real ingest smoke runs into temp dirs

Out of scope:

- parity-equivalent coverage
- large probe libraries copied from `ageo-atoms`
- stateful/stochastic family smoke coverage

### Implementation Questions

1. Which existing `ageo-atoms` runtime probes can be mirrored safely with a
   smaller matcher-local form?
2. Should grouped-family probes dispatch by exported symbol name or by package
   family id?
3. How should failures be reported when a grouped package contains one passing
   and one failing exported symbol?

### Required Deliverables

- `../ageo-matcher/sciona/ingester/smoke.py` expansion
- focused matcher smoke tests
- one grouped sklearn smoke example and one top-level numerical example

### Exit Criteria

- smoke allowlist covers more than one artificial canary function
- grouped sklearn image helpers have matcher-side smoke coverage
- at least one real numerical helper family has matcher-side smoke coverage

## Workstream 2: Add E2E Regression Fixtures For Learned Families

### Objective

Make the families that taught us the lessons part of the matcher’s regression
surface.

### Priority Families

Start with:

- sklearn `images/`
- BioSPPy ECG/EMG/PPG detector patterns

The goal is not to ingest the whole external libraries in test. The goal is to
create stable local fixtures that preserve the same shape pressures:

- grouped pure functions
- structured returns
- witness/decorator wiring
- optional/defaulted parameters

### Scope

In scope:

- new or expanded test fixtures under `../ageo-matcher/tests/fixtures/`
- ingest regression tests that assert emitted outputs or monitor summaries
- regression cases for grouped family output
- regression cases for detector-like return shapes where passthrough must be
  preserved

Out of scope:

- live external library imports in test
- heavyweight integration against the entire `ageo-atoms` tree

### Implementation Questions

1. Which parts of the BioSPPy family should be mirrored with minimal fixtures?
2. Should grouped sklearn fixtures be function-only or include a shallow class
   wrapper too?
3. Which emitted artifacts are stable enough to assert directly versus via
   monitor summary?

### Required Deliverables

- new fixture files in matcher tests
- regression tests covering grouped output plus detector-style shaped outputs
- documentation/comments that explain which real-world repair each fixture
  represents

### Exit Criteria

- at least one grouped-family regression fixture exists
- at least one detector-style regression fixture exists
- future ingester changes that reintroduce the original learned failures would
  trip matcher tests

## Workstream 3: Add A Narrow Return-Shape Knowledge Layer

### Objective

Handle explicitly known structured-return APIs without reopening broad
heuristic output extraction.

### Problem

The current return policy is appropriately conservative, but some known-good
structured outputs still need explicit handling. Today that knowledge mostly
lives outside the matcher, in manual fixes and audit knowledge.

### Scope

In scope:

- a narrow matcher-side allowlist for known structured output patterns
- code paths that authorize extraction only when the API is explicitly listed
- regression tests for both:
  - allowlisted structured extraction
  - fallback passthrough for everything else

Out of scope:

- general inference of dict/list/tuple return fields
- learned extraction from names alone
- ambitious decomposition-time output invention

### Candidate First Cases

Start only if the first two workstreams are stable. Initial candidates should
be small and explicit, likely from detector-like or graph-helper wrappers where
the output structure is obvious and already proven by repo fixes.

### Implementation Questions

1. Where should the allowlist live: chunker, emitter, or a dedicated helper?
2. Should the allowlist key by fully qualified symbol, emitted atom id, or both?
3. How do we make sure unsupported structured outputs still default to
   passthrough?

### Required Deliverables

- a dedicated matcher helper or config structure for return-shape exceptions
- regression tests for allowlisted extraction and conservative fallback
- one realistic fixture showing the intended structured case

### Exit Criteria

- structured extraction only occurs when explicitly allowlisted
- known structured cases no longer require manual emitter cleanup
- non-allowlisted targets still stay conservative

## Recommended Execution Order

The order matters:

1. Workstream 1 first
2. Workstream 2 second
3. Workstream 3 third

Why:

- smoke coverage expansion gives immediate value with low design risk
- regression fixtures then lock those lessons into matcher tests
- return-shape knowledge should come last because it is the easiest place to
  accidentally reintroduce over-interpretation

## Suggested First Implementation Slice

The first worker should **not** try to implement the full plan at once.

It should implement:

- Workstream 1 for grouped sklearn image helpers and one stable numerical
  helper family
- the first Workstream 2 grouped sklearn fixture if it fits naturally

That slice is enough to prove:

- the smoke gate scales beyond the initial canary
- grouped-family ingest remains a first-class path
- matcher regression coverage now includes one real family that we actively use

## Concrete File Targets For The First Worker

Primary write scope in `../ageo-matcher`:

- `sciona/ingester/smoke.py`
- `sciona/commands/ingest_cmds.py` only if smoke summaries need small extension
- relevant matcher tests:
  - `tests/test_ingest_smoke.py`
  - new fixture/regression tests if needed

Secondary write scope only if a grouped fixture needs it:

- `tests/fixtures/ingest_regression/...`

`ageo-atoms` should not need code changes for the first worker beyond this plan
document.

## Verification Expectations

Minimum:

- targeted matcher `pytest -q` on the changed smoke/regression tests
- import sanity for changed matcher modules

Recommended:

- one temp-dir grouped ingest smoke run
- one temp-dir top-level helper ingest smoke run
- report the smoke result summaries from those runs

## Handoff Requirements For A Worker

When assigning this plan to a coding worker:

1. Tell the worker to start with **Workstream 1 only**, unless Workstream 2
   fixture support falls out naturally.
2. Keep the write scope narrow.
3. Tell the worker not to touch unrelated `.claude` files or any unrelated
   local dirt.
4. Require the worker to list:
   - changed files
   - tests run
   - any smoke-run commands used
   - what remains intentionally unimplemented

## Success Definition

This rollout is successful when:

- matcher-side smoke coverage is materially broader
- the key learned families are represented in matcher regression tests
- return-shape special handling, where needed, stays explicit and narrow
- new ingests increasingly fail early instead of creating medium-risk cleanup
  debt in `ageo-atoms`
