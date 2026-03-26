# Refine Ingest Brief

This document is the shared briefing for planner agents drafting follow-up work
for `sciona ingest`.

The original refine-ingest program is complete. Phases 1 through 12 landed, and
the ingest runtime is now canonical-first, regression-backed, and operationally
stabilized enough that future work should be framed as cleanup, hardening, or
tooling follow-up rather than continuation of the original semantic redesign.

This file is not itself an implementation plan. It is the current state brief
and recommendation set that should let a planner agent produce a concrete next
phase plan from a cold start.

## Status

Completed phases:

- phase 1: deterministic semantic fact extraction
- phase 2: canonical ingest IR redesign
- phase 3: deterministic decomposition and planning
- phase 4: canonical wrapper/state emission
- phase 5: deterministic verification classification and fail-fast repair
- phase 6: lightweight regression harness
- phase 7: non-Python / tree-sitter / FFI canonical parity
- phase 8: canonical contract artifacts
- phase 9: canonical-first runtime execution
- phase 10: compatibility-boundary narrowing and canonical-first cleanup
- phase 11: curated real-world regression corpus and golden snapshots
- phase 12: cache, monitor, and operational-surface stabilization

There are no unfinished phases remaining from the original ingest-refinement
effort.

## Goal

The ingest system should:

- recover as much semantic truth as possible deterministically from source
- use LLMs only where ambiguity remains around grouping, naming, or explanation
- keep canonical IR and planning as the runtime source of truth
- preserve the ingestion contract in:
  - [INGEST_PROMPT.md](/Users/conrad/personal/ageo-atoms/INGEST_PROMPT.md)
  - [INGESTION.md](/Users/conrad/personal/ageo-atoms/INGESTION.md)
- avoid regressions in the protected non-sklearn families already working in
  this repository

## Current Architecture

### Canonical Source Of Truth

Canonical truth now flows through:

- deterministic extraction facts in `RawDataFlowGraph` / `MethodFact`
- canonical ingest IR:
  - `IngestIRPlan`
  - `OperationSpec`
  - `StateSlotSpec`
  - `OutputBindingSpec`
  - `OperationEdge`
- canonical planning output:
  - `IngestPlanGraph`
  - `PlannedOperationGroup`
  - `DecompositionDecision`

### Canonical-First Runtime

The runtime now works canonically first:

- chunking and decomposition consume canonical IR/planning
- emission consumes canonical bindings and state semantics first
- witnesses, CDG metadata, match metadata, and state-model shaping prefer
  canonical context
- procedural ingest attaches canonical IR directly
- non-Python / FFI paths participate where evidence is sufficient

### Compatibility Layer

Legacy `MacroAtomSpec`, `StateModelSpec`, and related fields still exist, but
they are no longer the semantic source of truth.

They should now be treated as:

- compatibility exports
- adapter surfaces for remaining downstream expectations
- cleanup candidates when clearly unused

Future planners should not design new behavior that depends on those legacy
structures as mutable runtime state.

### Verification And Repair

Verification is deterministic-first:

- verification failures are classified deterministically
- only narrow mechanical failures are repaired
- semantic failures fail fast with published artifacts

### Regression And Operational Surfaces

The ingest runtime now has:

- a curated regression harness in `sciona/ingester/regression_harness.py`
- a checked-in real-world fixture corpus and golden snapshots
- versioned ingest-cache envelopes
- stabilized monitor status / marker / surface schemas
- lightweight runtime and cache summary metrics

## Non-Negotiable Constraints

Every future plan should preserve these constraints.

### Semantic Constraints

- canonical IR/planning remains the source of truth
- later layers must not invent outputs, state, attributes, or signatures not
  supported by canonical evidence
- repair must not be used to hide semantic planning mistakes

### Contract Constraints

Generated artifacts must continue to satisfy the ingest contract in:

- [INGEST_PROMPT.md](/Users/conrad/personal/ageo-atoms/INGEST_PROMPT.md)
- [INGESTION.md](/Users/conrad/personal/ageo-atoms/INGESTION.md)

That includes:

- valid typed Python outputs
- witness compatibility
- state-model correctness
- meaningful contracts/docstrings
- CDG / witness / test compatibility

### Protected Families

Regressions in these families should be treated as unacceptable unless
explicitly approved:

- flat NumPy / SciPy style atoms
- stateful rolling / windowed classes
- Bayesian / stochastic / message-passing atoms
- DSP and biosignal atoms
- Rust / Julia / C++ / FFI-backed atoms
- opaque DL boundary atoms
- procedural ingestion mode

### Operational Constraints

Future work should preserve:

- monitor and trace visibility
- staged artifact publication
- partial artifact publication on failure
- deterministic fallback behavior when LLM calls fail
- practical runtime for local and CI use

## What Landed

This summary is intentionally compact but concrete so a planner does not need to
re-read twelve old phase plans first.

### Phases 1-4

Semantic extraction, canonical IR, canonical planning, and canonical emission
are all real and active.

The runtime now understands:

- exact signatures
- call and return facts
- provenance / spans
- explicit unknowns
- config vs fitted vs derived inventories
- canonical operations, state slots, outputs, and edges
- deterministic keep / decompose / block planning decisions
- exact wrapper/state emission from canonical bindings

### Phases 5-8

Verification and artifact correctness were tightened:

- deterministic verification classification
- mechanical-only repair
- fail-fast semantic failures
- non-Python canonical parity for representative tree-sitter / FFI cases
- canonical witnesses, CDG metadata, and match metadata

### Phases 9-12

The runtime and operational surfaces were hardened:

- canonical-first runtime execution
- narrower compatibility-export boundary
- curated regression corpus with golden snapshots
- versioned cache envelopes
- stabilized monitor schemas
- lightweight cache/runtime metrics in the harness

## Remaining Recommendations

The ingestion phase program is complete. What remains now is follow-up work.

These are the recommended next items, in priority order.

## Recommendation 1: Remove More Dead Transitional Surface Area

### Mission

- reduce remaining dead or low-value transitional code now that the runtime is
  canonically driven end-to-end

### Why This Matters

Phases 9 through 12 made the runtime canonical-first, but some transitional
helpers and compatibility-export surfaces still exist in:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`

That is safer than the pre-phase-9 design, but it still creates:

- cognitive overhead
- duplicated maintenance burden
- ambiguity around which helper layer is truly authoritative

### Target Outcomes

- clearly dead transitional helpers are removed
- compatibility exports become even more obviously export-only
- tests stop asserting transitional internals unless those internals remain part
  of a deliberate compatibility contract
- runtime call paths become easier to reason about and maintain

### Scope Guidance

In scope:

- identifying and deleting dead transitional helper paths
- tightening canonical-first helper/API usage
- shrinking compatibility-export plumbing where no longer needed
- updating tests to assert the intended architecture directly

Out of scope:

- changing canonical semantics
- reworking verification policy again
- changing public artifact/file contracts without a specific reason

### Primary Touchpoints

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- representative chunker/emitter/stateful/procedural tests

### Planner Notes

A good plan for this item should:

- name the exact transitional helpers or compatibility access patterns to
  remove or narrow
- distinguish export-only compatibility surfaces from still-legitimate public
  compatibility surfaces
- preserve regression coverage over protected families
- avoid mixing semantic redesign into cleanup work

## Recommendation 2: Add One Real Cache-Enabled End-To-End Ingest Test

### Mission

- prove that the actual cached ingest path behaves correctly across miss then
  hit, not just that cache serialization helpers round-trip

### Why This Matters

Phase 12 stabilized:

- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`
- harness cache metrics

But the remaining confidence gap is the real `IngesterAgent` cached path in
`sciona/ingester/graph.py`.

### Target Outcomes

- one focused integration test exercises a cache miss followed by a cache hit
- the test proves semantic outputs stay identical across the two paths
- monitor/cache surfaces expose the expected operational state

### Scope Guidance

In scope:

- a narrow end-to-end cached ingest test
- explicit assertions about cache-hit/cache-miss behavior
- keeping the integration test practical and deterministic

Out of scope:

- broad cache benchmark work
- full CI matrix expansion in the same phase

### Primary Touchpoints

- `sciona/ingester/graph.py`
- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`
- targeted integration tests

### Planner Notes

A good plan here should keep the integration scope very small and should not
turn into another cache architecture phase.

## Recommendation 3: Wire The Regression Corpus Into CI Deliberately

### Mission

- treat the curated corpus and golden snapshots as a protected regression suite

### Why This Matters

The regression corpus is now valuable enough to serve as a real gate, but it
should be wired into CI deliberately rather than by expanding every job.

### Target Outcomes

- a fast always-on slice
- a slightly broader scheduled or less-frequent slice
- explicit ownership of golden updates

### Scope Guidance

In scope:

- CI job design for the corpus
- practical default suite sizing
- golden-update workflow clarity

Out of scope:

- large hosted dashboards
- a broad product/CLI redesign

### Primary Touchpoints

- CI configuration
- `sciona/ingester/regression_harness.py`
- `tests/test_ingest_regression_harness.py`
- golden corpus directories

### Planner Notes

A good plan should separate frequent and infrequent coverage and avoid making
golden maintenance noisy.

## Recommendation 4: Write A Maintainer Architecture Note

### Mission

- document the canonical ingest architecture for future maintainers

### Why This Matters

The architecture is much cleaner now, but its conceptual model is spread across:

- `sciona/ingester/extractor.py`
- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/regression_harness.py`

### Target Outcomes

- one maintainer-oriented doc that explains:
  - canonical data flow
  - compatibility boundaries
  - where semantics are allowed to change
  - how regression/harness/cache/monitor surfaces fit together

### Planner Notes

This is documentation and architecture communication work, not another runtime
refactor.

## Recommendation 5: Decide What Is Now A Stable Public Contract

### Mission

- be explicit about which schemas and artifacts are intended to stay stable

### Why This Matters

Phase 12 improved stability, but future teams still need a clear statement of
what is intentionally supported long-term:

- cache envelopes
- monitor markers and status surfaces
- regression golden formats
- canonical runtime exports that other tooling may rely on

### Planner Notes

A good plan here should separate:

- intentionally stable contracts
- best-effort implementation details
- fields that may evolve without semantic meaning

## Planner Instructions

When writing a concrete implementation plan for one of these recommendations,
include:

- phase or follow-up goal
- exact scope boundaries
- current code touchpoints
- why the work is needed now
- deterministic vs LLM responsibilities
- data model or schema changes
- rollout steps
- regression risks to protected families
- concrete test or verification plan
- acceptance criteria
- what should remain deferred

Future work should emphasize simplification, confidence, maintainability, and
operational clarity rather than new semantic machinery.
