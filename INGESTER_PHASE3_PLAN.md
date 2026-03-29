# Ingester Phase 3 Plan

This document is the implementation plan for **Phase 3: Witness And Decorator
Emission Hardening** from
[INGESTER_HARDENING_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_HARDENING_EXECUTION_PLAN.md).

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Harden generated witness/decorator emission so the wrapper surface is
structurally correct and does not add obviously wrong runtime contracts.

For the current slice, the highest-value target is:

- stop emitting naive scalar `icontract.require(...)` checks for `*args` and
  `**kwargs`

That issue is now visible in the constrained ingest smoke path even after
Phase 1 and Phase 2 are fixed.

## Why This Phase Is Needed

Recent validation exposed a residual emitter problem:

- the generated wrapper signature for `head, /, *args, scale=..., **kwargs`
  is now correct
- the output binding fallback is now correct
- but default contract generation still emits:
  - scalar numeric checks on `args`
  - scalar numeric checks on `kwargs`

That is semantically wrong and produces a wrapper that still looks
ingester-shaped rather than source-shaped.

This phase also covers the adjacent witness/decorator safety guarantee:

- `@register_atom(...)` should always reference a witness symbol directly, not
  a quoted string or other invalid form

## Non-Goals

Do not expand this phase into:

- package layout
- full witness semantic quality
- runtime smoke-validation framework
- return-shape inference beyond what Phase 2 already changed

## Current Baseline In `../ageo-matcher`

Relevant files:

- `sciona/ingester/emitter.py`
- `tests/test_ingester_emitter.py`

Relevant code paths:

- default `icontract.require(...)` generation in:
  - `generate_atom_wrappers(...)`
  - `generate_stateful_wrappers(...)`
- witness import and `@register_atom(...)` emission

## Design Direction

Decorator generation should be conservative:

1. Do not emit type contracts for `vararg` / `kwarg` params using scalar
   assumptions.
2. Prefer no contract over a wrong contract.
3. Keep witness registration symbol-based and import-backed.
4. Add regression coverage for both ordinary wrappers and the variadic case.

## Implementation Workstreams

### Workstream 1: Variadic Contract Suppression

Objective:

- stop generating incorrect scalar/non-null contracts for variadic params

Tasks:

- inspect default require generation in `generate_atom_wrappers(...)`
- inspect the analogous path in `generate_stateful_wrappers(...)`
- skip naive default requires for:
  - `ParameterFact.kind == "vararg"`
  - `ParameterFact.kind == "kwarg"`
- keep ordinary parameter contracts unchanged

### Workstream 2: Witness Registration Regression Coverage

Objective:

- lock in direct-symbol witness registration

Tasks:

- add or strengthen tests that confirm:
  - `@register_atom(witness_name)` is emitted
  - quoted witness names are not emitted

### Workstream 3: Variadic Wrapper Regression Coverage

Objective:

- prove the variadic wrapper surface is cleaner after the patch

Tasks:

- add a test using a canonical binding with:
  - one regular param
  - `*args`
  - one keyword-only default
  - `**kwargs`
- assert that the generated source:
  - preserves the public signature
  - does not emit scalar numeric contracts for `args` or `kwargs`
  - still emits ordinary contracts for the non-variadic input where justified

## Validation

Minimum:

- `pytest -q tests/test_ingester_emitter.py`

Recommended:

- `pytest -q tests/test_ingester_emitter.py tests/test_ingester_chunker.py tests/test_chunker_depth.py`

Practical smoke check:

- rerun the same constrained `aggregate(head, /, *args, scale=..., **kwargs)`
  ingest and verify the generated wrapper no longer contains the bogus
  variadic scalar `icontract.require(...)` lines

## Exit Criteria

Phase 3 is complete when:

1. variadic params no longer receive obviously wrong scalar default contracts
2. witness registration remains direct-symbol based
3. matcher tests pass
4. the constrained variadic ingest smoke output no longer contains the bad
   `args` / `kwargs` scalar contract lines
