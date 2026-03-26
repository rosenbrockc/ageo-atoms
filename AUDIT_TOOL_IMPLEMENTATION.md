# Deterministic Audit Tool Implementation Plan

This document turns the deterministic audit ideas from
[AUDIT_INGEST.md](/Users/conrad/personal/ageo-atoms/AUDIT_INGEST.md) into an
implementation plan for this repository.

The design goal is conservative automation:

- deterministic tools should reliably identify `broken` atoms
- deterministic tools should surface `misleading` atoms and rank them
- deterministic tools should quantify how acceptable an atom looks under the
  audit definition
- deterministic tools should never grant `trusted` on their own

`trusted` remains a promoted state that requires structured review plus pinned
provenance.

## Scope

Implement these deterministic tools:

- `scripts/build_audit_manifest.py`
- `scripts/audit_signature_fidelity.py`
- `scripts/audit_runtime_probes.py`
- `scripts/audit_acceptability.py`
- `scripts/audit_semantics.py`
- `scripts/report_audit_progress.py`

These tools should reuse and wrap existing signals where possible:

- `scripts/audit.py`
- `scripts/type_and_isomorphism_audit.py`
- `scripts/audit_references.py`
- `scripts/generate_parity_tests.py`
- `scripts/atom_manifest.yml`

## Architecture

Use thin CLI scripts backed by a shared library package:

- `ageoa/audit/__init__.py`
- `ageoa/audit/models.py`
- `ageoa/audit/inventory.py`
- `ageoa/audit/upstream.py`
- `ageoa/audit/structural.py`
- `ageoa/audit/runtime.py`
- `ageoa/audit/fidelity.py`
- `ageoa/audit/acceptability.py`
- `ageoa/audit/reporting.py`

Rationale:

- the repo already has multiple one-off scripts
- deterministic semantic audit will need shared schemas and evidence loaders
- keeping logic in importable modules makes unit testing practical

## Data Artifacts

Keep the top-level artifact names from `AUDIT_INGEST.md`, and add deterministic
intermediates under `data/audit/`.

Primary artifacts:

- `data/audit_manifest.json`
- `data/audit_scores.csv`
- `data/audit_reviews/`

Deterministic intermediates:

- `data/audit/evidence/<atom_id>.json`
- `data/audit/results/<atom_id>.json`
- `data/audit/probes/<atom_id>.json`
- `data/audit/summary.json`

`data/audit_manifest.json` should be the inventory. Per-atom evidence and
results should be generated, replaceable artifacts.

## Canonical Schema

Each manifest row should contain at least:

- `atom_id`
- `atom_name`
- `module_path`
- `wrapper_symbol`
- `wrapper_line`
- `domain_family`
- `module_family`
- `source_kind`
- `risk_tier`
- `upstream_symbols`
- `upstream_version`
- `source_revision`
- `review_basis_at`
- `stateful`
- `ffi`
- `skeleton`
- `has_state_models`
- `has_witnesses`
- `has_cdg`
- `has_references`
- `has_parity_tests`
- `structural_status`
- `runtime_status`
- `semantic_status`
- `developer_semantics_status`
- `parity_test_status`
- `references_status`
- `overall_verdict`
- `acceptability_score`
- `acceptability_band`
- `max_reviewable_verdict`
- `blocking_findings`
- `required_actions`

Use normalized component statuses:

- `pass`
- `partial`
- `fail`
- `unknown`

## Tool 1: Inventory And Manifest

Script: `scripts/build_audit_manifest.py`

Responsibilities:

- walk every public atom under `ageoa/`
- identify atom functions and stable `atom_id` values
- collect sibling artifacts:
  - `witnesses.py`
  - `state_models.py`
  - `cdg.json`
  - `references.json`
  - `COMPLETED.json`
  - `FAILED.json`
  - `matches.json`
  - `trace.jsonl`
- map each atom to `scripts/atom_manifest.yml` when available
- detect source kind:
  - hand-written
  - refined ingest
  - generated ingest
  - skeleton
- infer initial risk tier from deterministic heuristics

Implementation notes:

- reuse AST walking logic from `scripts/audit.py`
- resolve `atom_id` as
  `fully.qualified.name@relative/path.py:<line>`
- do not depend on imports succeeding; inventory must work on a broken repo

Outputs:

- `data/audit_manifest.json`
- optional `data/audit_manifest.csv`

## Tool 2: Signature Fidelity Audit

Script: `scripts/audit_signature_fidelity.py`

Responsibilities:

- compare wrapper signatures to mapped upstream symbols
- flag extra invented parameters
- flag missing required upstream parameters
- compare defaults where introspection is possible
- compare return arity shape when mechanically knowable

Evidence sources in priority order:

1. installed Python symbol via `inspect.signature`
2. vendored Python source under `third_party/`
3. deterministic mapping metadata from `scripts/atom_manifest.yml`
4. static parse of local wrapper only, if nothing else exists

Checks:

- parameter count mismatch
- required parameter name mismatch
- suspicious renaming without alias note
- wildcard public signatures
- wrappers exposing only normalized `object` or `Any`
- missing return annotations

Outputs:

- `data/audit/evidence/<atom_id>.json`
  with a `signature_fidelity` section
- optional summary table of worst mismatches

## Tool 3: Runtime Probe Audit

Script: `scripts/audit_runtime_probes.py`

Responsibilities:

- import wrapper modules safely
- run tiny positive-path probes where fixtures are available
- run negative-path contract probes when input generators exist
- detect whether wrapper reaches a real upstream call path when probeable

Probe strategy:

- use explicit allowlists for safe probeable families first
- reuse existing parity fixtures in `tests/fixtures/`
- reuse or extend `scripts/generate_base_inputs.py`
- never synthesize expensive or side-effectful probes for unknown families
- store probe skip reasons deterministically

Checks:

- import success
- callable success on tiny fixtures
- expected contract failure on invalid fixtures
- parity result available
- `NotImplementedError` or skeleton path reached

Outputs:

- `data/audit/probes/<atom_id>.json`
- manifest updates for `runtime_status` and `parity_test_status`

## Tool 4: Deterministic Acceptability Scorer

Script: `scripts/audit_acceptability.py`

This is the key tool for quantifying whether an atom is acceptable under the
audit definition.

Important constraint:

- this tool may quantify acceptability
- this tool may cap the highest plausible verdict
- this tool may not assign `trusted`

### What The Score Means

The acceptability score answers:

"How much deterministic evidence do we have that this atom is non-broken,
non-misleading, and useful enough to remain in the repo while awaiting or
undergoing semantic review?"

It is not a semantic proof. It is a conservative portfolio-management score.

### Output Contract

For each atom, emit:

- `acceptability_score`: integer `0..100`
- `acceptability_band`
- `max_reviewable_verdict`
- `hard_blockers`
- `major_penalties`
- `dimension_scores`
- `dimension_evidence`

Bands:

- `0-19`: `broken_candidate`
- `20-49`: `misleading_candidate`
- `50-69`: `limited_acceptability`
- `70-84`: `acceptable_with_limits_candidate`
- `85-100`: `review_ready`

`review_ready` means only that the atom has enough deterministic evidence to be
worth a structured semantic review. It does not mean `trusted`.

### Dimension Model

Use a weighted score with hard caps.

Dimensions:

- structural evidence: 20 points
- runtime evidence: 20 points
- upstream fidelity evidence: 35 points
- developer-semantics evidence: 15 points
- trust-supporting evidence: 10 points

Base formula:

`acceptability_score = structural + runtime + fidelity + semantics + trust_support`

Each dimension is itself deterministic and decomposed into explicit checks.

### Structural Evidence: 20 points

Inputs:

- `scripts/audit.py`
- `scripts/type_and_isomorphism_audit.py`
- manifest artifact presence

Suggested sub-checks:

- parseable wrapper file
- valid registration and contract decorators
- witness presence and typing
- CDG presence and parseability
- no `NotImplementedError` placeholder in public atom body
- no obviously placeholder docstring

Scoring:

- full points only if all required structural checks pass
- subtract fixed penalties per finding code

Hard cap:

- if parse, registration, or witness integrity fails, cap score at `19`

### Runtime Evidence: 20 points

Inputs:

- runtime probe results
- parity test results when present

Suggested sub-checks:

- wrapper imports
- positive probe succeeds
- negative probe fails for the right reason
- parity or usage-equivalence evidence exists

Scoring:

- import-only is not enough for full credit
- no positive-path execution should cap runtime evidence at half

Hard cap:

- if import or positive-path execution fails, cap score at `19`

### Upstream Fidelity Evidence: 35 points

Inputs:

- upstream mapping from `scripts/atom_manifest.yml`
- signature audit
- return/state checks
- generated-noun checks

Suggested sub-checks:

- upstream symbol mapped
- upstream provenance pinned
- signature name and arity match
- wrapper returns upstream value or documented derived artifact
- wrapper does not read nonexistent attributes
- stateful wrapper rehydrates real fitted state before query methods
- generated nouns are present in source/docs or explicitly allowlisted as
  derived abstractions

Scoring:

- missing upstream mapping should lose most fidelity credit
- severe signature mismatch should lose substantial credit
- fabricated outputs or pseudo-state should trigger a major penalty

Hard caps:

- no upstream mapping: cap total score at `59`
- unpinned provenance: cap total score at `69`
- fabricated attribute access or pseudo-state: cap total score at `49`

### Developer-Semantics Evidence: 15 points

Inputs:

- wrapper name and docstring
- decomposition structure in `cdg.json`
- token overlap between wrapper API and upstream API/docs

Suggested sub-checks:

- wrapper name resembles upstream symbol or a documented abstraction
- docstring names a real operation
- decomposition does not split query accessors into fake training steps
- output names are explainable to a library user

Deterministic heuristics:

- tokenize snake/camel case into normalized terms
- compare tokens to upstream symbol names, docs, and vendored source terms
- apply penalties for invented nouns not seen upstream and not in an allowlist
- flag decompositions where multiple atoms claim the same mutating step without
  distinct upstream anchors

Hard cap:

- strong invented-noun or decomposition-distortion signal: cap total score at
  `49`

### Trust-Supporting Evidence: 10 points

Inputs:

- references status
- parity evidence
- review provenance fields

Suggested sub-checks:

- `references.json` exists or global references map to the atom
- parity coverage exists when applicable
- upstream version or source revision recorded

Hard caps:

- no parity or usage-equivalence evidence: cap total score at `84`
- missing provenance fields: cap total score at `69`

### Deterministic Ceilings

The scorer should emit both a numeric score and a verdict ceiling.

Rules:

- `broken` ceiling if structural or runtime hard failures exist
- `misleading` ceiling if fidelity or developer-semantics hard failures exist
- `acceptable_with_limits` ceiling otherwise
- `trusted` is never emitted by this tool

This makes the tool safe to use in CI and portfolio triage.

### Recommended Finding Codes

Examples:

- `STRUCT_PARSE_FAIL`
- `STRUCT_REGISTER_MISSING`
- `STRUCT_WITNESS_MISSING`
- `STRUCT_STUB_PUBLIC_API`
- `RUNTIME_IMPORT_FAIL`
- `RUNTIME_POSITIVE_PROBE_FAIL`
- `RUNTIME_NO_PROBE_EVIDENCE`
- `FIDELITY_UPSTREAM_UNMAPPED`
- `FIDELITY_SIGNATURE_MISMATCH`
- `FIDELITY_RETURN_FABRICATED`
- `FIDELITY_STATE_FABRICATED`
- `SEMANTICS_INVENTED_NOUN`
- `SEMANTICS_DECOMPOSITION_DISTORTION`
- `TRUST_PROVENANCE_MISSING`
- `TRUST_PARITY_MISSING`

## Tool 5: Semantic Deterministic Aggregator

Script: `scripts/audit_semantics.py`

Responsibilities:

- load manifest rows
- join structural, runtime, fidelity, and acceptability evidence
- derive component statuses
- derive deterministic ceiling verdict
- emit per-atom result files and a sortable portfolio report

This tool should be the single place that applies rollup logic from
`AUDIT_INGEST.md`.

Outputs:

- `data/audit/results/<atom_id>.json`
- `data/audit_scores.csv`
- `data/audit/summary.json`

## Tool 6: Progress Reporter

Script: `scripts/report_audit_progress.py`

Responsibilities:

- summarize counts by verdict ceiling, risk tier, family, and evidence gap
- list highest-priority atoms for human review
- list highest-frequency blocker codes
- show trend deltas if prior summary exists

Primary views:

- family coverage
- high-risk unresolved atoms
- review-ready atoms
- misleading candidates by severity
- broken candidates by repair cost

## Deterministic Heuristics For "Acceptability"

The most important design choice is to score acceptability as evidence-backed
usability, not as latent semantic correctness.

### Acceptability Definition

An atom is deterministically acceptable only if all of the following are true:

- it is not structurally broken
- it is not runtime broken on safe probes
- it has a plausible upstream anchor
- it does not show deterministic signs of fabricated API semantics
- its limitations are either absent because evidence is strong, or explicitly
  documented because evidence is partial

This maps directly to `acceptable_with_limits`, not `trusted`.

### Why A Numeric Score Helps

The repo is too large for binary review queues only. A numeric score allows:

- ranking which atoms need urgent human review
- finding atoms that are almost good enough but missing one deterministic input
- separating obvious garbage from plausible-but-unreviewed wrappers

### Why Hard Caps Matter

Pure weighted sums are unsafe here. A wrapper that parses, imports, and has
good references should still score poorly if it invents state or outputs.

That is why the scorer must apply hard caps before publishing a final score.

## Family-Specific Allowlists

Some derived outputs are legitimate even if token overlap is imperfect.

Add allowlists for:

- common signal-processing derived terms
- standard ML fitted-state names
- established probabilistic-inference summary terms
- known FFI bridge state handles

The allowlists should be explicit JSON files under:

- `data/audit/allowlists/generated_nouns.json`
- `data/audit/allowlists/state_fields.json`
- `data/audit/allowlists/decomposition_aliases.json`

These files should be reviewed, not generated ad hoc during scoring.

## Suggested Implementation Order

Phase 1:

- create `ageoa/audit/` package
- implement inventory models and manifest builder
- add per-atom stable ID generation

Phase 2:

- wrap current `scripts/audit.py` output into structured JSON
- wrap current CDG/type audit into structured JSON
- write deterministic manifest enrichment

Phase 3:

- implement upstream resolver backed by `scripts/atom_manifest.yml`
- implement Python signature comparison for installed and vendored sources

Phase 4:

- implement runtime probe harness for safe Python families
- integrate existing fixtures and parity tests

Phase 5:

- implement generated-noun, return-fidelity, and state-fidelity checks
- implement acceptability scorer with hard caps

Phase 6:

- implement semantic aggregator and progress reporter
- generate first full-repo portfolio report

Phase 7:

- add golden tests for representative atoms
- add CI gate for deterministic regressions

## Golden Test Corpus

Build a small hand-labeled corpus before trusting the scorer.

Minimum corpus:

- 10 atoms expected to be `broken`
- 10 atoms expected to be `misleading`
- 10 atoms expected to be `acceptable_with_limits`
- 10 atoms expected to be strong `review_ready` candidates

Use atoms from:

- `ageoa/sklearn/`
- `ageoa/biosppy/`
- `ageoa/pronto/`
- `ageoa/scipy/`
- `ageoa/tempo_jl/`

Golden tests should assert:

- finding codes
- verdict ceiling
- score band
- stable score deltas within a small tolerance only when heuristics are updated

## CI And Gating

Recommended CI policy after rollout:

- manifest build must succeed
- deterministic structural audit must not regress
- no new atom may merge without an audit manifest row
- no new high-risk atom may merge with a `broken` or `misleading` ceiling
- score drops greater than a threshold should require review

Do not use CI to auto-promote `trusted`.

## Immediate Deliverables

The first implementation batch should produce:

1. `scripts/build_audit_manifest.py`
2. `scripts/audit_signature_fidelity.py`
3. `scripts/audit_acceptability.py`
4. `data/audit_manifest.json`
5. `data/audit_scores.csv`

That gives the repo a deterministic inventory, a fidelity baseline, and the
acceptability quantification you asked for. Runtime probes can then land on top
of that foundation without forcing schema churn.
