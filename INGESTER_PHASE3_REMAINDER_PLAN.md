# Ingester Phase 3 Remainder Plan

This document is the detailed implementation plan for the **remaining work in
Phase 3: Witness And Decorator Emission Hardening**.

Use this plan after the already-landed Phase 3 subset:

- direct-symbol `@register_atom(...)` coverage
- suppression of naive scalar contracts for `*args` / `**kwargs`

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Finish Phase 3 by hardening the witness/ghost surface for nontrivial callable
signatures, especially canonical wrappers with:

- positional-only parameters
- `*args`
- `**kwargs`
- keyword-only defaults
- other signatures where the public wrapper is now correct but the generated
  witness surface is still simplistic or semantically weak

## Why This Remaining Work Exists

After Phase 1 and Phase 2, a constrained ingest such as:

- `aggregate(head, /, *args, scale=..., **kwargs)`

now produces:

- a better public wrapper signature
- correct passthrough return behavior
- cleaner default contracts

But the witness/ghost surface is still weak:

- witness parameters are derived from `wrapper_inputs` only
- variadic parameters are represented as if they were ordinary scalar inputs
- witness construction still assumes simple one-parameter-per-input modeling
- ghost fidelity is not aligned with the improved wrapper signature fidelity

This is no longer a wrapper-signature problem or an output-binding problem. It
is the remaining Phase 3 witness/decorator problem.

## Current Baseline

Relevant code in `../ageo-matcher`:

- `sciona/ingester/emitter.py`

Relevant functions:

- `_canonical_witness_param_specs(...)`
- `_ghost_abstract_type(...)`
- `_ghost_constructor_expr(...)`
- `generate_ghost_witnesses(...)`
- default witness generation branches for canonical wrappers
- any witness generation paths for stateful wrappers and concept-specific
  templates

Relevant tests:

- `tests/test_ingester_emitter.py`

Observed current limitations:

1. Witness parameters are shaped from `wrapper_inputs`, but not from
   `MethodBinding.signature` kind semantics.
2. `vararg` / `kwarg` inputs are not modeled specially in witness signatures.
3. Witness return generation is adequate for simple single-return cases, but
   the input abstraction is not trustworthy for variadic APIs.
4. The constrained smoke ingest still does not fully pass mypy/ghost, and the
   witness surface is one of the remaining likely causes.

## Scope

In scope:

- witness parameter modeling for canonical wrappers
- variadic witness parameter treatment
- positional-only / keyword-only witness parameter ordering where relevant
- witness return generation only insofar as it must stay consistent with the
  improved input modeling
- regression tests for witness surfaces of canonical wrappers

Out of scope:

- package grouping
- runtime smoke-validation framework
- broader ghost semantics for every concept family
- return-binding inference beyond already landed Phase 2 behavior

## Design Direction

Witness generation should be conservative and structurally aligned with the
wrapper surface.

Guiding rules:

1. Witness parameter shape should follow canonical binding semantics, not just
   flat wrapper input names.
2. `*args` should not be represented as an ordinary scalar argument.
3. `**kwargs` should not be represented as an ordinary scalar argument.
4. If the witness system cannot model a variadic parameter richly, it should
   use a broad abstract container-like surrogate rather than a false scalar.
5. The witness surface should remain syntactically simple enough for current
   ghost tooling.

## Key Planning Decision

The most practical target for this phase is not “perfect abstract modeling of
variadics.” It is:

- **stop lying**

That means:

- do not emit scalar witness params for variadics
- prefer broad/structural abstract types
- keep witness bodies shape-preserving and minimal

## Recommended Implementation Strategy

### Workstream 1: Add Binding-Aware Witness Param Metadata

Objective:

- carry enough parameter-kind information into witness generation to treat
  variadics and other special parameter kinds differently

Tasks:

- extend `_canonical_witness_param_specs(...)` so it returns more than
  `wrapper_inputs`
- include:
  - ordered witness inputs
  - a parameter-fact map or ordered parameter metadata
  - any state info already needed by witness generation
- ensure witness generation can see `ParameterFact.kind`

Likely file:

- `../ageo-matcher/sciona/ingester/emitter.py`

### Workstream 2: Introduce Conservative Witness Types For Variadics

Objective:

- prevent `*args` and `**kwargs` from becoming false scalars in witness
  signatures

Tasks:

- design a conservative witness-parameter mapping for:
  - `vararg`
  - `kwarg`
- likely options:
  - `*args` -> `AbstractArray` or `tuple[AbstractScalar, ...]`-like surrogate
  - `**kwargs` -> `dict[str, AbstractScalar]` or `dict[str, AbstractArray]`
    depending on what current ghost tooling can tolerate
- prefer whatever is easiest for current ghost parsing/type-fixing to accept

Important constraint:

- do not introduce exotic abstract types unless they already exist or are very
  cheap to support

Likely file:

- `../ageo-matcher/sciona/ingester/emitter.py`

### Workstream 3: Keep Witness Ordering Consistent With Canonical Signatures

Objective:

- make witness parameter order match the canonical wrapper surface closely

Tasks:

- preserve the same meaningful ordering for:
  - positional inputs
  - variadic input slot
  - keyword-like slot(s)
  - state parameter if present
- do not necessarily reproduce `/` and `*` syntax in witness functions if that
  makes ghost tooling harder; structural consistency matters more than exact
  Python signature markers for witnesses

Likely file:

- `../ageo-matcher/sciona/ingester/emitter.py`

### Workstream 4: Regression Tests For Canonical Witness Surfaces

Objective:

- pin the new witness behavior in matcher tests

Required tests:

- canonical witness for a simple non-variadic wrapper still uses exact inputs
- canonical witness for a variadic wrapper:
  - does not use scalar witness params for `args` / `kwargs`
  - remains valid Python
  - has a stable expected signature string
- if practical, a stateful canonical witness case still behaves correctly

Likely file:

- `../ageo-matcher/tests/test_ingester_emitter.py`

### Workstream 5: Practical Smoke Validation

Objective:

- prove the witness surface improved on the known demo case

Tasks:

- rerun the constrained `aggregate(head, /, *args, scale=..., **kwargs)` ingest
- inspect generated `witnesses.py`
- confirm:
  - no scalar witness params for `args` / `kwargs`
  - witness remains syntactically valid
  - wrapper/witness pair are more coherent than before

Nice-to-have:

- if this also improves mypy/ghost outcomes, record that
- if it does not, record the remaining blocker explicitly

## Files The Coding Agent Should Expect To Touch

Primary:

- `../ageo-matcher/sciona/ingester/emitter.py`
- `../ageo-matcher/tests/test_ingester_emitter.py`

Avoid touching other files unless strictly necessary.

## Verification Commands

Minimum:

- `pytest -q tests/test_ingester_emitter.py`

Recommended:

- `pytest -q tests/test_ingester_emitter.py tests/test_ingester_chunker.py tests/test_chunker_depth.py`

Practical smoke check:

- constrained `sciona ingest` of the temp/demo variadic function target
- inspect generated `witnesses.py`

## Exit Criteria

This remainder is complete when:

1. canonical variadic witness signatures no longer model `args` / `kwargs` as
   false scalar inputs
2. witness generation remains syntactically valid
3. non-variadic witness behavior does not regress
4. matcher tests pass
5. the constrained smoke ingest shows a materially cleaner `witnesses.py`

## Risks

- current ghost tooling may not accept richer abstract container annotations
- overcomplicating witness signatures could create more fragility than value
- witness fidelity and ghost pass rates may still be limited by issues outside
  witness parameter modeling

## Recommended Coding Order

1. inspect current canonical witness param derivation
2. implement binding-aware witness parameter metadata
3. add conservative variadic witness mapping
4. add regression tests
5. run matcher tests
6. rerun the constrained smoke ingest and inspect `witnesses.py`
