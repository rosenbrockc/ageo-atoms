# Ingester Rollout Workstream 3 Plan

This document is the implementation plan for **Workstream 3** from
[INGESTER_ROLLOUT_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_ROLLOUT_PLAN.md):
a narrow return-shape knowledge layer for explicitly known structured outputs.

Primary implementation repo:

- `../ageo-matcher`

Primary validation repo:

- `ageo-atoms`

## Goal

Add a small explicit matcher-side allowlist for structured return handling so
known-good APIs can emit stable extraction logic without reopening broad
heuristic output interpretation.

This is the final rollout slice from the learned ingester work. The intent is
not to become ambitious again. The intent is to make a few explicitly known
structured-return cases first-class and deterministic.

## Why This Slice Exists

The hardening phases deliberately moved the matcher toward conservative
passthrough. That was the right default, but it leaves a gap:

- some APIs really do have stable, structured outputs
- those outputs are now only handled well when the matcher already knows the
  exact shape or when a human repairs the emitted wrapper later

Recent trust-debt cleanup in `ageo-atoms` showed that detector-like wrappers
are the clearest example. The matcher should support those cases explicitly,
but only when the API is named and the return shape is backed by evidence.

## Current Baseline

Relevant matcher behavior today:

- `chunker.py` builds `OutputBindingSpec` entries from return facts
- `emitter.py` turns those bindings into concrete Python extraction code
- unknown bindings are now handled conservatively
- no matcher-side structured-return knowledge layer exists yet

Relevant matcher files:

- `../ageo-matcher/sciona/ingester/chunker.py`
- `../ageo-matcher/sciona/ingester/emitter.py`
- optionally `../ageo-matcher/sciona/ingester/prompts.py` only if a tiny prompt
  clarification is necessary

Relevant learned family shapes:

- detector-style outputs like `{"rpeaks": ..., "quality": ...}`
- tuple-return helpers only where slot meaning is explicit and stable

## Non-Goals

Do not broaden this slice into:

- general dict field inference
- general tuple-slot inference
- learned extraction from variable names alone
- new ambitious decomposition logic
- broad smoke coverage work
- additional grouped-output changes

## Design Rules

1. Unsupported cases must still default to passthrough.
2. Structured extraction must only occur when explicitly allowlisted.
3. The allowlist should be keyed narrowly and deterministically.
4. The implementation should be small enough to audit by reading one helper.
5. Tests must cover both the allowlisted path and the fallback path.

## Recommended Architecture

Prefer a small dedicated matcher helper, for example:

- `../ageo-matcher/sciona/ingester/return_shapes.py`

That helper should own:

- the allowlist entries
- the lookup function
- the small data model describing known structured-return behavior

The helper should then be consumed by:

- `chunker.py` when deciding output bindings
- or `emitter.py` if that is the cleaner insertion point

Preferred insertion point:

- keep the knowledge layer close to binding construction in `chunker.py`
- let `emitter.py` remain a renderer of already-validated binding choices

## Candidate First Cases

Start with one or two highly explicit detector-style cases only.

Recommended first case:

- the synthetic detector-style regression fixture added in Workstream 2
  (`detector_structured_output`)

Optional second case if implementation is naturally small:

- one tuple-return helper with clearly documented slot semantics

Do not start with broad library coverage.

## Allowlist Design

The allowlist should probably map:

- fully qualified or fixture-local subject name
- optionally emitted target symbol name
- to a structured description of outputs

Suggested shape:

- `subject_name`
- `source_method` or exported symbol
- output bindings such as:
  - `output_name="rpeaks", binding_kind="dict_field", field_key="rpeaks"`
  - `output_name="quality", binding_kind="dict_field", field_key="quality"`

If tuple support is included, the same model can support:

- `binding_kind="tuple_element", tuple_index=0`

## Key Implementation Questions

1. Should the allowlist key by canonical IR `subject_name`, exported symbol
   name, source method name, or a composite of these?
2. Where is the least fragile place to intercept current default bindings?
3. Do we need a new binding kind such as `dict_field`, or can we reuse an
   existing shape cleanly?
4. How should fallback work when an allowlisted case is partially matched but
   missing outputs?

## Recommended Answers

Unless the code proves otherwise:

- key the allowlist by `subject_name` plus source method or exported symbol
- add a small explicit binding kind such as `dict_field`
- resolve the allowlisted structured bindings in `chunker.py`
- if an allowlist entry is incomplete or mismatched, fail closed to passthrough
  rather than partial extraction

## Workstreams

### Workstream A: Add A Small Return-Shape Knowledge Helper

Objective:

- define the allowlist in one auditable matcher module

Tasks:

- create the helper module
- define the structured-return entry model
- define the lookup function
- keep the initial entry set intentionally tiny

Exit criteria:

- the matcher has one clear home for explicit structured-return knowledge

### Workstream B: Integrate The Knowledge Layer Into Binding Construction

Objective:

- let known structured cases produce explicit bindings while everything else
  stays conservative

Tasks:

- inspect current output-binding construction in `chunker.py`
- add a narrow branch that checks the allowlist first
- emit structured bindings only for allowlisted cases
- preserve current fallback behavior for all non-allowlisted cases

Exit criteria:

- allowlisted case yields explicit structured bindings
- non-allowlisted cases retain passthrough or current conservative behavior

### Workstream C: Add Regression Tests

Objective:

- prove both the allowlisted path and fallback path

Required coverage:

- unit coverage for the new allowlist helper
- binding-construction coverage for the allowlisted structured case
- fallback coverage for a non-allowlisted structured return
- if practical, one end-to-end emitter-level assertion on emitted extraction

Preferred files:

- `../ageo-matcher/tests/test_ingester_chunker.py`
- `../ageo-matcher/tests/test_ingester_emitter.py`
- optionally a dedicated small test file if that is cleaner

Exit criteria:

- matcher tests fail if the allowlisted structured case regresses
- matcher tests fail if non-allowlisted cases accidentally become extractive

### Workstream D: Validate Against The Workstream 2 Fixture

Objective:

- ensure the new return-shape layer is grounded in the curated regression corpus

Tasks:

- use the detector-style fixture/golden family from Workstream 2
- confirm the generated bindings/emission align with the intended structured
  output semantics
- keep the validation deterministic and local

Exit criteria:

- one realistic structured-return case is explicitly covered by the new layer

## Concrete File Targets

Primary write scope in `../ageo-matcher`:

- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py` only if needed
- `sciona/ingester/return_shapes.py` (new helper, preferred)
- targeted matcher tests around chunker/emitter

Secondary write scope only if necessary:

- Workstream 2 detector fixture/golden files, but only if a tiny correction is
  required to align the new return-shape assertions

`ageo-atoms` should not need code changes for implementation of this slice
beyond this plan document.

## Verification Commands

Minimum:

- targeted matcher tests for the touched chunker/emitter/helper files

Recommended:

- `pytest -q tests/test_ingester_chunker.py tests/test_ingester_emitter.py`
- a focused selector if new tests live in a dedicated file

If the detector-style harness case is used directly, record:

- whether the allowlisted extraction path was exercised
- whether fallback still held for non-allowlisted cases

## Success Criteria

This slice is complete when:

- the matcher has a narrow explicit return-shape allowlist
- at least one detector-style structured case uses it successfully
- unsupported cases still stay conservative
- no broad heuristic output extraction is reintroduced

## Suggested Worker Scope

Assign a coding worker ownership of:

- `../ageo-matcher/sciona/ingester/chunker.py`
- `../ageo-matcher/sciona/ingester/emitter.py` if needed
- `../ageo-matcher/sciona/ingester/return_shapes.py` if created
- focused matcher tests around chunker/emitter/structured returns

Tell the worker:

- do not touch unrelated `.claude` files
- do not broaden beyond one or two explicit structured-return cases
- preserve the current conservative fallback path
- report exactly which allowlisted case(s) were implemented

## Follow-On Audit

After this slice lands, the next non-implementation task should be a dedicated
audit of `ageo-atoms` vs `../ageo-matcher` to verify that the risk-audit
lessons are fully reflected in the ingester. That audit should compare:

- the fixed high/medium-risk families in `ageo-atoms`
- the five hardening phases
- the three rollout workstreams
- any remaining manual repairs that the matcher still does not explain

That audit should be treated as a separate work item after Workstream 3, not
folded into this implementation slice.
