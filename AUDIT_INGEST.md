# Audit Ingest Plan

This document defines how to audit every atom in this repository for semantic
fidelity, not just syntactic validity.

The motivating failure mode is now clear:

- an ingest can complete
- generated code can pass mypy
- ghost simulation can pass
- and the resulting atom can still be misleading or wrong from the perspective
  of a developer trying to use the upstream library

That is unacceptable. Every public atom in `ageoa/` should make sense to a
developer using the original library or algorithm.

This plan is for auditing the current repository and building the audit harness
that will keep future ingests honest.

## Goal

For every atom in this repo, answer:

- Does it represent what the upstream code actually does?
- Does it expose inputs, outputs, and state in a way that a developer would
  recognize as correct?
- Does the generated decomposition help a developer understand and use the
  underlying algorithm, rather than inventing a fake API?

The audit should classify each atom as one of:

- `trusted`: semantically faithful and usable
- `acceptable_with_limits`: useful, but intentionally approximate or skeletonized
- `misleading`: compiles and registers but misrepresents behavior
- `broken`: fails contract, import, witness, runtime, parity, or semantic checks

`unknown` is allowed only as a temporary workflow state in the manifest before
review. It is not a final audit verdict.

Overall verdicts must be derived from component checks, not assigned ad hoc:

- `broken` if any required structural or runtime check fails, or if the atom
  cannot meet its claimed contract
- `misleading` if the atom executes but materially misstates upstream API
  behavior, state, outputs, or usage semantics
- `acceptable_with_limits` if the atom is useful but intentionally approximate,
  incompletely validated, or awaiting evidence required for trust, with limits
  explicitly documented
- `trusted` only if structural and runtime checks pass, semantic and developer
  semantics review pass, provenance is recorded, and parity or
  usage-equivalence evidence exists

## Scope

Audit all public atoms under `ageoa/`.

Current rough scope:

- `163` `atoms.py` files are present in this repo
- each may contain one or more public atoms
- the audit must cover:
  - atom wrappers
  - witnesses
  - state models
  - CDGs
  - tests
  - scholarly references where applicable

Do not limit the audit to recent sklearn outputs. The sklearn failure is the
signal that the whole repository needs a semantic review pass.

## Audit Principle

Passing the current ingest pipeline is necessary but not sufficient.

We need to audit across four levels:

1. Structural validity
2. Executable validity
3. Upstream behavioral fidelity
4. Developer-meaningful semantics

The current repo already has some tooling for level 1. This plan extends the
repo to levels 2 through 4.

## What “Makes Sense To A Developer” Means

An atom makes sense to a developer when all of these are true:

- The function name and docstring describe a real upstream operation.
- The parameters correspond to actual upstream inputs.
- The return value corresponds to an actual upstream return value or a clearly
  documented derived artifact.
- For stateful APIs, state is separated into meaningful categories:
  - configuration
  - learned/fitted state
  - transient outputs
- Wrapper calls match upstream signatures.
- The atom does not invent attributes, pseudo-objects, or control flow that a
  developer would not find in the upstream code.
- A user could read the atom and understand how to use the upstream API more
  clearly, not less.

## Audit Dimensions

Every audit pass should evaluate each atom across these dimensions.

### 1. Structural

Questions:

- Does `atoms.py` parse?
- Are decorators valid and ordered correctly?
- Are witnesses present and typed?
- Are CDGs well-formed and acyclic?
- Are exports and registrations correct?

Existing tools:

- `python scripts/audit.py --verbose`
- `python ../ageo-matcher/scripts/verify_atoms_repo.py . --package ageoa`
- `python scripts/type_and_isomorphism_audit.py --verbose`

### 2. Runtime / Executable

Questions:

- Does the atom import cleanly?
- Does it raise the right precondition errors on bad inputs?
- Does it produce a concrete output on sane inputs?
- If it is a wrapper, does it call a real upstream implementation?

Existing tools:

- targeted pytest
- registry tests
- ghost simulator tests

Gap:

- many atoms still have no runtime execution audit beyond importability

### 3. Upstream Fidelity

Questions:

- Do wrapper signatures match upstream callable signatures?
- Are outputs faithful to upstream returns or real state transitions?
- Are state models sourced from real persistent fields?
- Are helper/query methods distinguished from mutators?
- Are docstrings aligned with the authoritative docs/source?

This is the core missing audit dimension.

### 4. Developer Semantics

Questions:

- Would a library user recognize this as a sensible decomposition?
- Does the atom reveal the actual usage pattern?
- Is the chosen abstraction useful, or is it a hallucinated decomposition?
- Does the atom improve understanding, or would it mislead a developer?

This requires human or high-quality model review over a structured rubric.

## Audit Strategy

The audit should proceed in phases.

## Phase 1: Inventory And Manifest

Build a repository-wide manifest of all public atoms.

Required output:

- one row per public atom
- owning module path
- domain/family
- whether it is hand-written, refined, or fully ingested
- whether it is stateful, stochastic, FFI-backed, procedural, or skeletonized
- whether parity tests exist
- whether scholarly references exist
- current status: trusted / acceptable / misleading / broken / unknown

Suggested implementation:

- add a script such as `scripts/build_audit_manifest.py`
- walk all `atoms.py` files
- collect public defs, decorator metadata, witness bindings, sibling files, and
  available tests
- write `data/audit_manifest.json` or `.csv`

Why:

- we need a durable index before auditing atom-by-atom
- the repo is too large to track manually

## Phase 2: Structural Bulk Audit

Run and extend the existing static audits across the full manifest.

Minimum checks:

- current `scripts/audit.py`
- current `verify_atoms_repo.py`
- CDG type/isomorphism audit
- missing witness/import/export issues
- heavy import leakage
- empty or placeholder docstrings
- `NotImplementedError` stubs

Deliverable:

- a normalized repository-wide structural audit report

This phase is necessary, but it does not establish semantic trust.

## Phase 3: Semantic Risk Triage

Classify atoms by likelihood of semantic failure.

High-risk categories should be audited first:

- recent LLM ingests
- object-oriented library wrappers
- stateful APIs
- atoms with generated state models
- atoms with invented nouns in outputs or state
- atoms with weak types like `object`, `Any`, or heavily normalized types
- atoms lacking parity tests
- FFI atoms that still contain skeletal bridging

Lower-risk categories:

- thin wrappers around a single stable upstream function
- mature hand-refined DSP atoms with existing parity tests

Suggested result:

- add a `risk_tier` per atom in the manifest:
  - `high`
  - `medium`
  - `low`

## Phase 4: Semantic Review Rubric

For every atom, perform a semantic audit using a standard rubric.

Each audit record should answer:

1. What upstream symbol(s) does this atom claim to represent?
2. What source/doc pages are authoritative?
3. Does the wrapper signature match the upstream interface?
4. Does the wrapper body call real upstream methods/functions correctly?
5. Are the outputs faithful?
6. For stateful APIs, is state separation faithful?
7. Does the decomposition into multiple atoms help or distort?
8. Does the docstring teach correct usage?
9. Are contracts meaningful for the real API?
10. Overall verdict: trusted / acceptable / misleading / broken

The audit should require line references to:

- local atom files
- local upstream vendored source when present under `third_party/`
- upstream docs/source links when the source is not vendored

Every review must also record the upstream basis for the judgment:

- upstream package version when reviewing against an installed dependency
- vendored commit, tag, or snapshot identity when reviewing against
  `third_party/`
- review timestamp for the authoritative docs/source consulted

Without pinned provenance, a semantic review is incomplete and not reproducible.

## Phase 5: Add Deterministic Semantic Checks

We should not rely only on manual review. Add automated semantic checks where
possible.

Examples of checks worth implementing:

### Signature Fidelity Checks

- compare wrapper parameter count/names to upstream callable signatures
- flag extra invented parameters
- flag obviously missing required parameters

### Return-Fidelity Checks

- flag wrappers that read nonexistent attributes after upstream calls
- flag wrappers that ignore real return values and substitute fabricated ones
- flag wrappers that call methods with known return values but then harvest
  unrelated object attributes

### Stateful API Checks

- detect fitted-state attributes from upstream source
- verify inference wrappers rehydrate fitted state before predict/transform
- verify query methods do not masquerade as mutating training steps

### Generated-Noun Checks

- flag outputs/state names that do not occur in upstream source or docs and are
  not documented as derived abstractions

### Wrapper Runtime Probes

- for importable Python upstream symbols, instantiate or call wrappers on tiny
  safe fixtures and check that the wrapper reaches a real upstream path

These checks should be conservative. They are intended to surface suspicious
atoms for review, not to prove full semantic equivalence.

## Phase 6: Parity And Usage Tests

Every trusted atom should have at least a minimal behavioral audit.

Required test categories:

- positive-path execution
- precondition violation tests
- upstream parity test or usage-equivalence test

For simple functional atoms:

- compare atom output to upstream output on representative inputs

For stateful atoms:

- compare wrapper behavior or derived outputs to upstream object behavior over a
  short usage sequence

For approximate or intentionally abstracted atoms:

- require an explicit limitations note in the audit record

The repo already has parity-test patterns. Expand them into a more systematic
coverage map rather than only ad hoc cases.

## Phase 7: Human / Model Review Pass

Some semantic failures are easy to miss with static tooling.

Use a reviewer agent or human reviewer to inspect each atom audit record,
especially for high-risk atoms. High-risk atoms should be reviewed first, but no
atom may be promoted to `trusted` without a structured review. Lower-risk atoms
may remain `acceptable_with_limits` until that review is complete. The reviewer
should:

- read the local atom
- read the authoritative upstream source/docs
- score the atom against the rubric
- propose either:
  - accept
  - refine
  - rewrite
  - downgrade to skeleton/approximate atom

This review should be structured, not free-form.

Suggested artifact:

- one markdown or JSON review record per atom under something like
  `data/audits/<atom_id>.json`

## Phase 8: Repository-Wide Remediation

Once the manifest and audit results exist, remediate in priority order.

Priority order:

1. broken atoms
2. misleading atoms
3. high-risk unknown atoms
4. acceptable_with_limits atoms lacking documented limitations
5. low-risk trusted atoms missing parity coverage

Important:

- do not rewrite everything at once
- audit should drive remediation
- every fix should update the atom’s audit record and regression coverage

## Phase 9: Ongoing Gating

After the initial repository audit, make semantic quality part of the ingest
workflow.

Suggested gates:

- structural audit must pass on every change
- new ingests must generate an audit record
- no ingest may be marked `trusted` without semantic review and pinned
  provenance
- high-risk ingests require semantic review before merge or release, not just
  before promotion
- parity coverage must exist before an atom can be promoted from
  `acceptable_with_limits` to `trusted`

## Repository Artifacts To Add

The audit effort should produce durable artifacts, not just terminal output.

Recommended additions:

- `data/audit_manifest.json`
- `data/audit_reviews/`
- `data/audit_scores.csv`
- `scripts/build_audit_manifest.py`
- `scripts/audit_semantics.py`
- `scripts/audit_signature_fidelity.py`
- `scripts/audit_runtime_probes.py`
- `scripts/report_audit_progress.py`

## Suggested Audit Record Schema

Each atom should eventually have a record with fields like:

- `atom_id`
- `module_path`
- `domain_family`
- `source_kind`
- `risk_tier`
- `upstream_symbols`
- `authoritative_sources`
- `upstream_version`
- `source_revision`
- `review_basis_at`
- `stateful`
- `ffi`
- `skeleton`
- `structural_status`
- `runtime_status`
- `semantic_status`
- `developer_semantics_status`
- `parity_test_status`
- `references_status`
- `overall_verdict`
- `verdict_reason`
- `review_notes`
- `required_actions`
- `last_reviewed_at`

## Verdict Derivation And Promotion Rules

Component statuses should be explicit and normalized, for example:

- `pass`
- `partial`
- `fail`
- `unknown`

The overall verdict should then be derived consistently:

- mark `broken` if `structural_status` or `runtime_status` is `fail`
- mark `misleading` if semantic or developer-semantics review finds the atom
  materially unfaithful, even when it executes
- mark `acceptable_with_limits` if the atom is non-broken and potentially
  useful but any required trust evidence is still `partial` or `unknown`
- mark `trusted` only if structural, runtime, semantic, and
  developer-semantics statuses are `pass`, provenance fields are populated, and
  parity evidence is present

This rollup should be implemented in tooling so that two reviewers evaluating
the same evidence cannot silently assign different final verdicts.

## Questions Every Audit Must Answer

No atom should be marked trusted until these questions have a clear answer.

### Wrapper Truth

- Does this wrapper call the real upstream function/method correctly?
- If not, is the abstraction intentional and documented?

### API Truth

- Would a developer using the upstream library understand how to use it from
  this atom?

### State Truth

- If the API is stateful, does the atom reflect real persistent state rather
  than fabricated pseudo-state?

### Output Truth

- Are outputs real returns or real post-call state?

### Decomposition Truth

- If a class was decomposed into multiple atoms, is that decomposition useful
  and faithful?

## Special Attention Areas

Based on current repo structure, these families should get early scrutiny:

- newly ingested sklearn outputs under `ageoa/sklearn/`
- generated `_d12` variants
- stateful robotics / quant engine atoms
- MCMC / Bayesian multi-step ingests
- FFI-backed atoms where wrappers may still be shallow or skeletal

Also review any atom families that were previously bulk-fixed in
[ISSUES.md](/Users/conrad/personal/ageo-atoms/ISSUES.md), because passing a
static audit after bulk edits does not guarantee good developer semantics.

## Minimum Definition Of Done

The repo-wide audit is not done when all files parse.

It is done when:

- every atom appears in the audit manifest
- every atom has at least a preliminary semantic classification
- every high-risk atom has a review record
- every broken or misleading atom is either fixed or clearly marked
- every trusted atom has a clear basis for trust
- the repository can answer, for any public atom, whether it is semantically
  faithful and developer-meaningful

## Immediate Next Steps

Recommended execution order:

1. Build the audit manifest
2. Run structural audits across all atoms and record results
3. Add risk-tier heuristics and rank the portfolio
4. Define the semantic review rubric and record schema
5. Audit high-risk atoms first
6. Add deterministic semantic checks for the common failure modes
7. Expand parity and usage tests
8. Iterate until every public atom has a semantic status

## Relationship To Ingest Refinement

This audit plan is complementary to
[REFINE_INGEST.md](/Users/conrad/personal/ageo-atoms/REFINE_INGEST.md).

- `REFINE_INGEST.md` is about improving the ingest system
- this file is about auditing the atoms already in the repo, plus defining the
  quality bar future ingests must meet

The audit should inform ingest refinement by showing which failure modes are
most common and most costly to developer trust.
