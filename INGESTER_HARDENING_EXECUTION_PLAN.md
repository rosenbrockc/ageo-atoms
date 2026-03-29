# Ingester Hardening Execution Plan

This document captures the concrete implementation phases implied by the most
recent trust-debt remediation work in `ageo-atoms`.

It is written for planning agents that will expand one phase at a time into a
more detailed implementation plan. Each phase below is intentionally bounded
and includes enough local context to let the next planning agent succeed
without reconstructing the full recent history.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Why This Plan Exists

Recent atom repairs exposed a repeatable pattern:

- wrappers drift from upstream signatures
- wrappers guess the wrong return shape and unwrap results incorrectly
- generated witness/decorator code can still be syntactically valid but
  semantically wrong
- package scoping is too fragmented by default
- deterministic validation happens too late, after bad wrappers have already
  landed in `ageo-atoms`

Those problems are now visible because the deterministic audit stack in this
repo is good enough to catch them. The goal of this plan is to move the most
useful lessons back into the ingester so fewer broken or misleading atoms are
generated in the first place.

## Current Baseline

Observed repair themes from recent manual fixes:

- signature drift in BioSPPy EMG / PPG detector wrappers
- bad return extraction in ECG detector wrappers
- incorrect string-based witness registration in generated code
- excessive one-symbol-per-package output fragmentation
- wrappers that only failed once deterministic runtime probes were added

Evidence examples in `ageo-atoms`:

- [ecg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ecg_detectors.py)
- [ppg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/ppg_detectors.py)
- [emg_detectors.py](/Users/conrad/personal/ageo-atoms/ageoa/biosppy/emg_detectors.py)
- [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
- [test_audit_runtime_probes.py](/Users/conrad/personal/ageo-atoms/tests/test_audit_runtime_probes.py)

Relevant existing `../ageo-matcher` touchpoints:

- `sciona/ingester/emitter.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/prompts.py`
- `sciona/commands/ingest_cmds.py`
- current ingester tests under `../ageo-matcher/tests/`

## Working Rules For Future Planning Agents

1. Treat the audit stack in `ageo-atoms` as the acceptance oracle.
2. Prefer conservative generation over ambitious interpretation.
3. Preserve upstream structure exactly unless there is explicit evidence for a
   narrower wrapper contract.
4. Keep implementation phases independent enough that each can be planned and
   executed without waiting on all later phases.
5. Every phase must end with both matcher-side tests and at least one
   ageo-atoms end-to-end validation slice.

## Phase Order

The recommended order is:

1. Signature fidelity hardening
2. Return-shape and output extraction hardening
3. Witness and decorator emission hardening
4. Package-scope and output-layout planning
5. Ingest-time deterministic smoke validation

This order matters. Phases 1 through 3 reduce obviously wrong emitted code.
Phase 4 changes output shape policy. Phase 5 adds the final early rejection
layer once the emitter is already more trustworthy.

## Phase 1: Signature Fidelity Hardening

### Objective

Make emitted wrappers mirror upstream callable signatures as faithfully as
possible.

### Problem This Phase Solves

Current emitted wrappers still accumulate audit debt from:

- parameter renaming
- dropped defaults
- weakened or invented keyword behavior
- stronger-than-upstream contracts
- mismatches between wrapper signature and upstream importable callable

These show up in the audit system as `RISK_SIGNATURE_MISMATCH` and often remain
even after the wrapper’s runtime behavior is repaired manually.

### Scope

This phase should focus on function and method signature emission only.

In scope:

- exact parameter names
- positional-only / keyword-only preservation where possible
- default value preservation
- `*args` / `**kwargs` preservation when upstream exposes them
- conservative parameter typing when inference is weak

Out of scope:

- return-shape extraction
- witness emission
- package grouping
- runtime smoke validation

### Likely Files To Inspect In `../ageo-matcher`

- `sciona/ingester/emitter.py`
- `sciona/ingester/chunker.py`
- any helper that builds canonical function contracts before emission
- tests already covering optional/defaulted parameters

### Key Questions The Next Planning Agent Must Answer

1. Where is the final emitted Python signature assembled?
2. At what point are upstream parameter names/defaults lost?
3. Which current transformations are intentional policy versus accidental drift?
4. Can canonical IR carry richer signature metadata without breaking existing
   emitters?
5. Should unknown types default to `object` / `Any` instead of invented narrow
   scalar/array types?

### Required Deliverables

- phase-specific implementation plan
- matcher-side regression tests covering:
  - optional/defaulted parameters
  - keyword-only parameters
  - `*args` / `**kwargs`
  - function targets and class-method-derived targets
- at least one end-to-end ingest check against a known-problem family, such as
  BioSPPy EMG/PPG or sklearn grouped functions

### Exit Criteria

- newly emitted wrappers preserve upstream callable shape by default
- signature-drift repairs stop being a common manual cleanup step
- at least one previously mismatched family ingests cleanly without manual
  signature editing

### Phase Risks

- preserving exact signatures may expose more unknown types than current code
- canonical IR may currently assume simplified parameter forms
- exact preservation may require relaxing some existing contract templates

## Phase 2: Return-Shape And Output Extraction Hardening

### Objective

Stop the ingester from guessing incorrect output extraction logic.

### Problem This Phase Solves

Some wrappers are currently broken because the emitted code assumes the
upstream callable returns:

- a dict with a specific key
- a tuple with a specific slot
- a scalar when the upstream actually returns an array
- an array when the upstream actually returns structured output

This caused recent manual repairs in ECG detector wrappers and is a major
source of semantically misleading atoms.

### Scope

This phase should harden how the ingester decides whether to:

- return the upstream result directly
- unwrap a field from a dict-like structure
- select from a tuple/list structure
- transform outputs before registration

In scope:

- canonical IR result-shape metadata
- emitter policy for result passthrough versus extraction
- prompt constraints for decomposition plans that mention outputs

Out of scope:

- package layout policy
- witness decorator syntax
- ingest-time smoke validation framework itself

### Likely Files To Inspect In `../ageo-matcher`

- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/prompts.py`
- any validator for canonical IR output nodes

### Key Questions The Next Planning Agent Must Answer

1. Where is the decision made to emit `result["field"]` or `result[0]`?
2. What evidence currently authorizes output extraction?
3. Can the ingester default to passthrough unless extraction is explicitly
   justified?
4. Are there safe metadata sources for known dict-returning APIs?
5. Can decomposition validation reject child plans that invent unsupported
   output fields?

### Required Deliverables

- phase-specific implementation plan
- matcher-side regression tests for:
  - passthrough ndarray return
  - dict-return extraction only when explicitly supported
  - tuple-return extraction only when explicitly supported
  - a prior known-bad wrapper pattern such as Hamilton segmentation
- one end-to-end ingest rerun on a known previously broken return-shape case

### Exit Criteria

- wrappers no longer invent extraction logic by default
- known-problem detector wrappers ingest with correct return handling
- canonical IR validation rejects unsupported output-field claims

### Phase Risks

- some older decomposition prompts may rely on optimistic output extraction
- stricter rules may reduce ambitious decomposition coverage in the short term

## Phase 3: Witness And Decorator Emission Hardening

### Objective

Ensure generated witness wiring and decorator emission are always structurally
and semantically correct.

### Problem This Phase Solves

Recent repairs found wrappers whose registration code was wrong even when the
wrapper body was mostly correct. The concrete example was string-based witness
registration instead of symbol-based registration.

This phase exists because syntactically minor emission defects create noisy
manual cleanup and can undermine provenance in otherwise acceptable wrappers.

### Scope

In scope:

- `@register_atom(...)` emission
- witness symbol imports
- ordering of decorators/imports
- generated witness module references
- matcher-side validation that emitted wrapper and witness symbols line up

Out of scope:

- witness semantic quality
- package grouping
- runtime smoke validation

### Likely Files To Inspect In `../ageo-matcher`

- `sciona/ingester/emitter.py`
- any witness codegen helper modules
- witness-related tests under `../ageo-matcher/tests/`

### Key Questions The Next Planning Agent Must Answer

1. Where is witness decorator syntax assembled?
2. How can symbol identity be validated before writing the file?
3. Are there other decorator emission paths besides `register_atom` that need
   the same hardening?
4. Should the ingester add a post-emit AST validation step for generated
   modules?

### Required Deliverables

- phase-specific implementation plan
- matcher-side regression tests for witness import and decorator syntax
- at least one negative test that would have produced the old string-based
  registration bug

### Exit Criteria

- emitted wrappers consistently reference witness symbols directly
- broken decorator forms are caught in matcher tests before publication

### Phase Risks

- there may be multiple emitter codepaths for function/class/refinement modes
- witness generation may be partially duplicated across codepaths

## Phase 4: Package Scope And Output Layout Hardening

### Objective

Move the ingester toward coherent family/module-level output grouping instead
of defaulting to one package per symbol.

### Problem This Phase Solves

The repository had to be manually consolidated because many ingested atoms were
over-fragmented into single-atom subpackages. That increases path noise,
creates duplicated sidecars, and makes family-level maintenance harder.

Recent grouped sklearn image atoms showed that a denser scope is workable and
cleaner.

### Scope

In scope:

- output path planning for grouped ingests
- module-family grouping policy
- compatibility with existing single-package and grouped-package layouts
- guidance for when one symbol should still have its own package

Out of scope:

- signature fidelity
- return-shape extraction
- smoke validation mechanics

### Likely Files To Inspect In `../ageo-matcher`

- `sciona/commands/ingest_cmds.py`
- any output directory planner or publish helper
- decomposition/publish helpers that assume one-symbol-per-dir

### Key Questions The Next Planning Agent Must Answer

1. What currently determines whether output goes to one symbol directory or a
   grouped module directory?
2. Can the CLI express grouped scope directly without awkward overloading?
3. Which sidecars need to become family-aware instead of symbol-aware?
4. What is the migration policy for existing output paths already in repos?

### Required Deliverables

- phase-specific implementation plan
- proposed grouping rules with concrete examples:
  - grouped image helpers
  - grouped detector families
  - cases that should remain separate because of state/provenance boundaries
- at least one end-to-end grouped ingest example in a sandbox path

### Exit Criteria

- grouped output becomes a first-class supported path
- new ingests do not default to needless single-atom fragmentation
- sidecars such as `cdg.json` and `matches.json` still remain coherent under
  grouped outputs

### Phase Risks

- existing tooling in `ageo-atoms` may assume certain directory shapes
- grouped outputs can complicate provenance if symbol ownership is unclear

## Phase 5: Ingest-Time Deterministic Smoke Validation

### Objective

Add a lightweight validation gate inside the ingester so obviously bad outputs
are rejected before they reach `ageo-atoms`.

### Problem This Phase Solves

The audit stack now catches many broken wrappers, but only after they have been
generated and reviewed. Recent remediation showed that several wrappers would
have been rejected immediately if the ingester had run even a small positive
and negative smoke check.

### Scope

In scope:

- safe post-emit validation hooks
- allowlisted pure-function smoke probes
- basic import checks
- positive and negative probe outcomes
- failure policy: reject, quarantine, or mark as partial

Out of scope:

- full audit replication inside the matcher
- human review workflow
- broad stochastic/stateful validation

### Likely Files To Inspect In `../ageo-matcher`

- `sciona/commands/ingest_cmds.py`
- monitor/publish helpers
- any existing mypy / ghost simulation validation hooks

### Key Questions The Next Planning Agent Must Answer

1. Where should smoke validation run relative to mypy and ghost simulation?
2. Should smoke validation be opt-in, opt-out, or automatic for an allowlisted
   pure subset?
3. How should failures be surfaced in provenance and publish status?
4. Can the matcher reuse probe definitions from `ageo-atoms`, or should it
   generate a smaller local subset?
5. What is the minimal validation contract that improves quality without
   recreating the whole audit system?

### Required Deliverables

- phase-specific implementation plan
- design for a small allowlisted smoke-validation framework
- matcher-side tests for:
  - successful smoke validation
  - expected negative-case rejection
  - skip behavior for unsupported/stateful targets
- one end-to-end example where a previously bad wrapper is rejected before
  publication

### Exit Criteria

- clearly broken pure wrappers fail earlier in the ingest pipeline
- smoke validation outcomes are recorded in matcher provenance / publish state
- `ageo-atoms` receives fewer trivially bad generated wrappers

### Phase Risks

- too-broad validation will slow ingest or create flaky failures
- duplicating audit logic inside the matcher can cause divergence if not scoped
  carefully

## How To Hand Off To The Next Planning Agent

When asking a planning agent to expand one phase, provide:

1. This document.
2. The exact phase name and number.
3. The current relevant files from `../ageo-matcher`.
4. One or two concrete bad examples from `ageo-atoms`.

The next planning agent should produce:

- a phase-specific plan
- specific file targets
- test strategy
- end-to-end validation strategy
- explicit exit criteria
- any open questions or blocking assumptions

## Recommended First Follow-Up

Start with **Phase 1: Signature Fidelity Hardening**.

It is the broadest remaining source of repetitive audit debt, and it should
reduce manual cleanup across multiple families before the later validation and
layout phases are added.
