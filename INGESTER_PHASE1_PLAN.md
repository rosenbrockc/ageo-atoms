# Ingester Phase 1 Plan

This document is the implementation plan for **Phase 1: Signature Fidelity
Hardening** from
[INGESTER_HARDENING_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_HARDENING_EXECUTION_PLAN.md).

It is written so a coding agent can implement the phase directly, and so a
future planning agent can resume without reconstructing context.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Make emitted wrappers preserve upstream callable signatures as faithfully as
possible by default.

For this phase, “signature fidelity” means:

- preserve parameter names
- preserve parameter order
- preserve parameter kinds:
  - positional-only
  - positional-or-keyword
  - keyword-only
  - `*args`
  - `**kwargs`
- preserve defaulted versus required parameters
- preserve default expressions when safely representable
- avoid inventing stronger contracts or narrower calling conventions than the
  source callable actually has

This phase is about the emitted wrapper interface and call plumbing. It is not
about return-shape extraction or package layout.

## Why This Phase Is Needed

Recent `ageo-atoms` remediation repeatedly found wrappers that ran correctly
only after manual edits, while still retaining audit debt from signature drift.

Observed failure modes:

- defaulted parameters converted into “required unless `_SCIONA_UNSET`”
- upstream keyword behavior represented only approximately
- wrapper signatures shaped by IOSpec filtering instead of the original method
  binding
- invented or narrowed parameter types
- loss of exact function/method signature shape for function targets

These show up in the deterministic audit stack as `RISK_SIGNATURE_MISMATCH`,
even when the wrapper body has been repaired.

## Current Baseline In `../ageo-matcher`

Relevant code paths:

- [emitter.py](../ageo-matcher/sciona/ingester/emitter.py)
- [chunker.py](../ageo-matcher/sciona/ingester/chunker.py)
- [models.py](../ageo-matcher/sciona/ingester/models.py)
- [test_ingester_emitter.py](../ageo-matcher/tests/test_ingester_emitter.py)

Relevant structures:

- `ParameterFact`
- `MethodBinding.signature`
- `_canonical_wrapper_inputs(...)`
- `_binding_parameter_facts(...)`
- `_wrapper_param_declaration(...)`
- `_wrapper_param_list(...)`
- `_canonical_call_arguments(...)`
- canonical wrapper emission around `_call_kwargs_*`

Important current behavior:

- the emitter already tracks parameter `kind`, `has_default`, and
  `default_expression`
- the wrapper signature is still derived from `wrapper_inputs`, which come from
  filtered IOSpecs rather than a first-class signature model
- defaulted non-vararg params are generally emitted as
  `param = _SCIONA_UNSET`
- conditional keyword plumbing is used to avoid forwarding unset defaults
- tests already exist for:
  - function targets
  - keyword-only defaults
  - optional inputs not being forced non-null

That means this phase is an evolution of the current emitter, not a rewrite
from scratch.

## Non-Goals

Do not expand this phase into:

- return binding / extraction policy
- witness decorator generation
- grouped package output policy
- ingest-time smoke validation framework
- broad audit changes in `ageo-atoms`

## Design Direction

The ingester should become **more conservative and more literal**.

Policy changes to implement:

1. Wrapper parameters should be driven from `MethodBinding.signature`, not only
   from filtered IOSpecs.
2. Exact parameter kind/order/default information should survive to emission.
3. If the ingester cannot prove a narrow type, it should preserve a broad type
   rather than invent a strict one.
4. `*args` and `**kwargs` must remain structurally intact when upstream
   exposes them.
5. Keyword-only parameters should be emitted as keyword-only because that is
   part of the source API, not an optional detail.
6. The fallback sentinel path should only remain where necessary to preserve
   “do not forward default unless explicitly passed” semantics. It should not
   distort the public signature more than needed.

## Open Technical Constraint

There is a real tension between:

- mirroring the upstream Python signature exactly
- preserving the current “only forward a defaulted kwarg when explicitly
  provided” behavior using `_SCIONA_UNSET`

The implementation for this phase must make that tradeoff explicit.

The likely acceptable compromise is:

- preserve exact order and parameter kind
- preserve exact parameter names
- keep sentinel-backed defaults only where forwarding the actual default would
  alter runtime behavior versus the upstream API
- add tests that document this as intentional behavior rather than accidental
  drift

If exact default-expression mirroring is possible without changing wrapper
behavior, prefer it.

## Implementation Workstreams

### Workstream 1: Trace Signature Data Flow

Objective:

- confirm where parameter fidelity is currently lost between extraction,
  canonical IR, and final emission

Tasks:

- inspect how `MethodFact.signature` becomes `MethodBinding.signature`
- inspect how `operation.direct_inputs` / `atom.inputs` filter wrapper inputs
- identify whether `ParameterFact.annotation` is currently ignored at emission
- identify which tests already pin desired behavior and which only pin current
  compromises

Expected output:

- no repo artifact required
- implementation notes should be reflected in code comments and tests

### Workstream 2: Make Wrapper Signature Emission Binding-Driven

Objective:

- generate wrapper parameter lists from actual binding signatures instead of
  mostly from IOSpecs

Tasks:

- introduce a binding-aware wrapper signature builder
- keep compatibility with state/config inputs that are not ordinary call
  parameters
- preserve:
  - order
  - kind
  - required/default split
  - `*args`
  - `**kwargs`
- avoid pulling in extra IOSpec-only inputs that are not part of the callable
  signature

Likely files:

- `../ageo-matcher/sciona/ingester/emitter.py`

Expected outcome:

- canonical wrapper signatures are shaped by `MethodBinding.signature` first
- IOSpec remains a type/semantic source, not the primary parameter-order source

### Workstream 3: Improve Default And Keyword Semantics

Objective:

- reduce public signature drift caused by current default handling

Tasks:

- review `_wrapper_param_declaration(...)`
- review `_wrapper_param_list(...)`
- review `_canonical_call_arguments(...)`
- review dynamic kwargs emission around `_call_kwargs_*`
- decide and document when `_SCIONA_UNSET` is still required
- preserve keyword-only barriers where upstream requires them
- verify positional-only parameters stay positional-only if the current model
  can express them

Likely files:

- `../ageo-matcher/sciona/ingester/emitter.py`
- possibly `../ageo-matcher/sciona/ingester/models.py` if parameter metadata
  needs enrichment

Expected outcome:

- fewer wrapper signatures that look “ingester-shaped” rather than
  source-shaped

### Workstream 4: Use Parameter Annotation/Default Metadata More Carefully

Objective:

- prefer upstream-derived metadata when available and weaken only when needed

Tasks:

- determine whether `ParameterFact.annotation` should influence emitted
  annotations more directly
- ensure unsafe annotation text is still sanitized
- avoid inventing narrow scalar/array types when evidence is weak
- preserve broad types where exact source annotations cannot be trusted

Likely files:

- `../ageo-matcher/sciona/ingester/emitter.py`
- possibly `../ageo-matcher/sciona/ingester/chunker.py`

Expected outcome:

- better alignment between extracted signature metadata and emitted types
- fewer audit failures caused by overconfident parameter typing

### Workstream 5: Regression Tests

Objective:

- lock the new signature-fidelity guarantees into matcher tests

Required test additions:

- function target with:
  - keyword-only defaults
  - preserved symbol name
  - no invented positional forwarding
- wrapper with `*args` and `**kwargs`
- wrapper with positional-only parameters if the source model can express them
- wrapper that preserves required-vs-defaulted ordering
- wrapper that does not leak IOSpec-only extras into the signature
- at least one test that checks emitted public signature text directly

Recommended test file:

- `../ageo-matcher/tests/test_ingester_emitter.py`

Optional additional tests:

- if a helper layer is refactored substantially, add a smaller focused unit
  test file instead of only growing the large emitter suite

## Validation In `ageo-atoms`

At least one end-to-end validation slice must be run after matcher-side tests.

Preferred validation targets:

- BioSPPy EMG / PPG detector family
- sklearn grouped function helpers

The point is not a full ingest campaign. The point is to confirm that a known
signature-problem family emits cleaner wrappers with less or no manual editing.

Minimum validation steps:

1. run matcher tests
2. run one constrained ingest against a known-problem function/module target
3. inspect emitted wrapper signature text
4. if practical, run the relevant local runtime probe or smoke test in
   `ageo-atoms`

## Concrete File Targets

Primary write scope in `../ageo-matcher`:

- `sciona/ingester/emitter.py`
- `tests/test_ingester_emitter.py`

Secondary write scope only if needed:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`

`ageo-atoms` should not need code changes for this phase unless a temporary
validation harness is added locally. Avoid editing repo code here unless
strictly necessary.

## Verification Commands

Matcher-side:

- `pytest -q tests/test_ingester_emitter.py`
- if touched: `pytest -q tests/test_ingester_chunker.py`

Recommended broader regression after implementation:

- `pytest -q tests/test_ingester_emitter.py tests/test_ingester_chunker.py tests/test_chunker_depth.py`

Optional end-to-end check:

- a constrained `sciona ingest ...` against one known-problem target using the
  current preferred model/provider path

If a local ageo-atoms validation slice is run, record:

- the target ingested
- whether signature text improved
- whether the wrapper still needs manual edits

## Exit Criteria

Phase 1 is complete when:

1. emitted wrapper signatures are binding-driven rather than IOSpec-driven
2. exact parameter names/order/kinds are preserved for covered canonical cases
3. defaulted keyword behavior is documented and regression-tested
4. `*args` / `**kwargs` survive emission correctly
5. matcher-side regression tests pass
6. at least one known-problem ingest target emits a materially cleaner wrapper
   without manual signature repair

## Known Risks

- exact signature mirroring may conflict with current `_SCIONA_UNSET`
  forwarding behavior
- parameter annotations extracted upstream may be too noisy to trust directly
- state/config injection for class targets may still require some divergence
  from the raw method signature
- function-target and class-target codepaths may drift if only one is hardened

## Recommended Coding Order

1. inspect and document current signature data flow
2. refactor emitter helpers to be binding-driven
3. add focused regression tests for exact signature shape
4. run matcher tests
5. run one constrained end-to-end ingest validation
6. only then make any small follow-up adjustments

## Handoff Notes For The Coding Agent

Do not broaden the task into return-binding cleanup or witness-generation
cleanup.

The highest-value result for this phase is:

- fewer wrappers whose public signatures obviously diverge from the upstream
  callable before anyone even looks at runtime semantics

If the exact-default mirroring problem cannot be fully resolved in one pass,
prefer:

- explicitly documented conservative behavior
- stronger regression coverage
- smaller public-signature drift

over a large speculative refactor.
