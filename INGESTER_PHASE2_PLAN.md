# Ingester Phase 2 Plan

This document is the implementation plan for **Phase 2: Return-Shape And
Output Extraction Hardening** from
[INGESTER_HARDENING_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_HARDENING_EXECUTION_PLAN.md).

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Stop the ingester from emitting broken output extraction logic when the return
shape is underspecified.

For this phase, conservative behavior means:

- pass the upstream return value through directly when there is a single output
  and no reliable evidence for tuple/attribute extraction
- keep tuple and attribute extraction only when explicitly supported by return
  facts
- fail closed for multi-output or ambiguous cases instead of inventing
  extraction

## Why This Phase Is Needed

Recent validation showed a clean Phase 1 signature-preserving ingest could
still emit:

- `binding_kind="unknown"` in canonical IR
- a wrapper body that ends in `NotImplementedError("unsupported binding kind")`

That is better than fabricating extraction, but it still means the ingester is
publishing wrappers that are obviously unusable even when a conservative
passthrough would be safer and more faithful.

Concrete example:

- a simple single-output function ingest produced a correct signature but still
  failed because the output binding stayed `unknown` instead of degrading to
  `return_value`

## Non-Goals

Do not broaden this phase into:

- signature fidelity work already covered by Phase 1
- witness/decorator emission
- grouped package layout
- smoke-validation framework

## Current Baseline In `../ageo-matcher`

Relevant code:

- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `tests/test_ingester_chunker.py`
- `tests/test_ingester_emitter.py`

Relevant functions:

- `_binding_from_return_fact(...)`
- `_infer_output_bindings(...)`
- `_canonical_output_expression(...)`
- canonical wrapper emission paths that currently raise on unsupported binding
  kinds

Current failure pattern:

1. chunker emits `OutputBindingSpec(binding_kind="unknown")`
2. emitter reaches `_canonical_output_expression(...)`
3. wrapper generation degrades into a hard `NotImplementedError`

## Design Direction

The ingester should become **less eager to infer structure and more willing to
use direct passthrough**.

Policy changes for this phase:

1. If there is exactly one declared output and the return behavior is otherwise
   unstructured, prefer `return_value`.
2. Only use:
   - `attribute_read`
   - `tuple_element`
   - `constant`
   - `metadata_object`
   when return facts explicitly justify them.
3. Multi-output ambiguous cases must still fail closed.
4. The emitter should have a last-resort conservative fallback for legacy or
   partially underspecified plans that still carry `binding_kind="unknown"` in
   a single-output passthrough-safe case.

## Implementation Workstreams

### Workstream 1: Normalize Unknown Single-Output Returns In The Chunker

Objective:

- avoid producing `binding_kind="unknown"` for simple single-result return
  paths

Tasks:

- inspect `_infer_output_bindings(...)`
- add a conservative rule:
  - when `legacy_outputs` has exactly one entry
  - and there is a single non-`self` return path
  - and no explicit tuple/attribute extraction evidence
  - normalize to `return_value` (or `metadata_object` for explicit metadata
    methods)
- apply the same idea to the no-legacy-output path when output name defaults to
  `result`

Write scope:

- `../ageo-matcher/sciona/ingester/chunker.py`

### Workstream 2: Add An Emitter-Side Last-Resort Fallback

Objective:

- keep older or partially underspecified plans from emitting obviously broken
  wrappers in single-output passthrough-safe cases

Tasks:

- inspect `_canonical_output_expression(...)`
- add a narrow fallback for:
  - a single output binding
  - `binding_kind="unknown"`
  - no tuple index
  - no attribute read requirement
- map that case to direct return passthrough instead of `NotImplementedError`

Write scope:

- `../ageo-matcher/sciona/ingester/emitter.py`

### Workstream 3: Regression Tests

Objective:

- lock in conservative passthrough behavior and preserve fail-closed behavior
  for ambiguous cases

Required matcher tests:

- chunker test: unknown single-output return normalizes to `return_value`
- emitter test: single unknown output binding emits `return _ret_0`
- emitter test: multi-output unknown binding still fails closed
- optional metadata method test if metadata passthrough differs from ordinary
  return passthrough

Write scope:

- `../ageo-matcher/tests/test_ingester_chunker.py`
- `../ageo-matcher/tests/test_ingester_emitter.py`

## Validation

Minimum:

- `pytest -q tests/test_ingester_emitter.py tests/test_ingester_chunker.py`

Recommended:

- `pytest -q tests/test_ingester_emitter.py tests/test_ingester_chunker.py tests/test_chunker_depth.py`

Practical smoke check:

- run one constrained ingest on a simple single-output function target
- confirm the wrapper no longer ends in an unsupported-binding
  `NotImplementedError`

## Exit Criteria

Phase 2 is complete when:

1. single-output unknown returns are normalized to passthrough
2. tuple/attribute extraction still requires explicit evidence
3. multi-output ambiguous cases still fail closed
4. matcher regression tests pass
5. a practical single-output ingest no longer emits a broken unsupported-output
   fallback

## Risks

- over-broad fallback could hide real tuple/attribute ambiguity
- metadata methods may need slightly different handling than ordinary return
  passthrough
- legacy plans may still carry underspecified output names even if the emitter
  fallback is improved

## Recommended Coding Order

1. patch chunker normalization first
2. add emitter-side fallback second
3. add regression tests
4. run matcher tests
5. rerun a constrained single-output ingest smoke check
