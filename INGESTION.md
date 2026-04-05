# Atom Ingestion Guide

This document is the full ingestion contract for `ageoa`.

It covers two things:

1. how `../ageo-matcher` should emit atoms into this repository
2. what this repository requires before an ingest is considered durable,
   reviewable, and acceptable

The short operational version lives in
[INGEST_PROMPT.md](/Users/conrad/personal/ageo-atoms/INGEST_PROMPT.md). This
document is the longer policy and workflow reference.

## Core Principle

The repo does not accept the old definition of ingestion:

- run the matcher
- get wrapper files
- stop

Current rule:

- ingestion is only complete when the stable artifacts, package shape,
  provenance, documentation, runtime evidence, and deterministic audit state
  are all coherent

## Scope Boundary

Primary producer:

- `sciona ingest` in `../ageo-matcher`

Primary consumer and acceptance layer:

- this repository
- the deterministic audit tooling under [scripts](/Users/conrad/personal/ageo-atoms/scripts)

The matcher is responsible for code generation and ingestion-time guardrails.
This repo is responsible for durable metadata, audit evidence, and final
portfolio truthfulness.

## Repository Shapes

### Preferred New Layout

New atoms should normally use a per-atom directory:

```text
ageoa/
  <domain>/
    __init__.py
    <atom_name>/
      __init__.py
      atoms.py
      witnesses.py
      state_models.py
      cdg.json
      matches.json
      references.json
      uncertainty.json
```

Notes:

- `state_models.py` exists only when the atom is genuinely stateful
- `matches.json`, `references.json`, and `uncertainty.json` are optional in the
  filesystem, but they are first-class durable artifacts when present

### Legacy Layouts

Legacy atoms still exist and remain valid:

- `ageoa/<domain>/<module>.py`
- `ageoa/<domain>/<stem>_witnesses.py`
- `ageoa/<domain>/<stem>_cdg.json`

Do not churn a legacy atom into the new layout just for aesthetics. Change
layout only when the remediation or ingest work genuinely benefits from it.

## Stable Artifacts vs Ephemeral Byproducts

### Durable Files

These are part of the committed atom contract:

- `atoms.py`
- `witnesses.py`
- `state_models.py` when needed
- `cdg.json`
- legacy `*_cdg.json` for grandfathered atoms
- `matches.json`
- `references.json`
- `uncertainty.json`
- relevant `__init__.py`

### Operational Byproducts

These are not canonical repo state and should not be committed:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`
- local `logs/`
- local `.playwright-mcp/`

## Interface Contract

The public wrapper surface is the first trust boundary. The atom interface must
be honest before any audit logic can help.

### `atoms.py`

Every public atom must satisfy all of the following:

- fully type annotated
- no public `Any`
- no public `*args` or `**kwargs`
- decorated with `@register_atom(...)`
- meaningful `@icontract.require` and `@icontract.ensure`
- name matches the actual operation being exported
- defaults and optionality match the real wrapper behavior
- callable/oracle hooks are typed and contracted as callables, not numerics
- array-normalizing wrappers must not emit scalar-only contracts
- stateful wrappers must use explicit state models or explicit state payloads
- no hidden mutable ambient state

Allowed but constrained:

- thin adapters over upstream functions
- refined-ingest helpers
- conservative passthrough wrappers when structure is uncertain

Not allowed:

- fake overload collapse under one public symbol
- misleading public names like anonymous `f`, `process`, or `vol` when a more
  specific name is available locally
- placeholder-style wrappers whose docstrings or contracts imply semantics that
  are not actually implemented

### Decorator Ordering

`icontract` checks run bottom-up. Keep the safest checks nearest `def`:

1. `isinstance`
2. simple shape/domain checks
3. expensive finiteness/value checks
4. `@register_atom(...)` outermost

### `witnesses.py`

Witness rules:

- abstract ghost inputs and outputs only
- pure, side-effect free
- semantically conservative
- shape/dtype propagation should match the public wrapper
- stateful estimators and filters should use state-shaped abstractions rather
  than unrelated distribution-shaped abstractions
- do not import ghost aliases that do not actually exist

### `state_models.py`

State model rules:

- include only durable cross-call state
- do not store configuration secretly in undocumented fields
- name fields to match the real wrapper state semantics
- prefer explicit dataclasses/structures over ad hoc dict fragments when the
  family already models persistent state

### `cdg.json`

CDG rules:

- structurally valid JSON schema shape
- reflects the actual decomposition
- no duplicate anonymous child nodes for distinct exported atoms
- refined-ingest helpers should use compact honest nodes, not fake orchestration
- children should be stable and reviewable, not transient matcher noise

## Documentation Contract

The ingestion contract is not only about code. It is also about whether the
atom is intelligible to later users and reviewers.

### In-Wrapper Documentation

Every public wrapper should have a docstring that explains:

- what the atom does
- what each input means
- what each output means
- whether the atom is stateful, stochastic, or merely a pure transform
- what callable/oracle hooks do
- any important limits, approximations, or adapter behavior

### De-jargonization

This repo does not yet maintain a single canonical de-jargonized sidecar, but
the ingest bar should still anticipate it.

Rules:

- avoid placeholder text and implementation-noise prose
- use plain technical English where possible
- make the purpose understandable outside the immediate source discipline
- do not simplify so aggressively that the semantics become false

Practical implication:

- wrappers and metadata should be written so a later plain-language or
  de-jargonized layer can be derived cleanly

### Interface Documentation Expectations

The atom surface must be documentation-ready for:

1. inputs and outputs
2. parameters and defaults
3. state shape when state exists
4. uncertainties or limitations
5. provenance and scholarly references

## Provenance And Scholarly References

This repo distinguishes between two related but different artifacts:

1. upstream/provenance mapping
2. scholarly references

### Upstream Mapping

Use [scripts/atom_manifest.yml](/Users/conrad/personal/ageo-atoms/scripts/atom_manifest.yml)
to record:

- exact upstream repo/module/function anchors when available
- curated closest anchors for refined-ingest adapters
- notes when the wrapper is a decomposition over a larger upstream algorithm

Rules:

- prefer exact anchors
- when exact anchors do not exist, document the closest honest anchor
- do not use a single ambiguous manifest entry for multiple exported helpers
- keep overloaded or decomposed helpers disambiguated at the atom level

### Scholarly References

Use per-atom `references.json` and the global reference registry for papers,
books, standards, and other scholarly material.

Primary scripts:

- [scripts/add_reference.py](/Users/conrad/personal/ageo-atoms/scripts/add_reference.py)
- [scripts/build_references.py](/Users/conrad/personal/ageo-atoms/scripts/build_references.py)
- [scripts/audit_references.py](/Users/conrad/personal/ageo-atoms/scripts/audit_references.py)

Rules:

- prefer DOI-backed entries when possible
- keep `references.json` stable once curated
- preserve manual match metadata and notes
- references should strengthen reviewability, not just satisfy a checkbox

## Uncertainty

`uncertainty.json` is a durable post-ingest artifact when empirical
perturbation analysis has been run.

Primary script:

- [scripts/measure_uncertainty.py](/Users/conrad/personal/ageo-atoms/scripts/measure_uncertainty.py)

Rules:

- do not fabricate uncertainty files manually
- preserve existing `uncertainty.json`
- if an atom is a strong uncertainty candidate and the work already touches the
  family deeply, consider measuring uncertainty in the same tranche

## Matcher Grounding

`matches.json` is a durable matcher handoff artifact when present.

Why it matters:

- it records grounding output not represented elsewhere in exactly the same way
- it supports later synthesis and audit work
- it is materially different from transient monitor byproducts

Rule:

- keep it when matcher grounding emitted it

## Review Basis And Audit Reviews

Low risk does not automatically mean trusted. Review basis remains a separate
layer of trust.

Relevant surfaces:

- [data/audit_reviews](/Users/conrad/personal/ageo-atoms/data/audit_reviews)
- [scripts/init_audit_reviews.py](/Users/conrad/personal/ageo-atoms/scripts/init_audit_reviews.py)
- [scripts/validate_audit_reviews.py](/Users/conrad/personal/ageo-atoms/scripts/validate_audit_reviews.py)
- [scripts/apply_audit_reviews.py](/Users/conrad/personal/ageo-atoms/scripts/apply_audit_reviews.py)
- [scripts/report_audit_progress.py](/Users/conrad/personal/ageo-atoms/scripts/report_audit_progress.py)

Rules:

- deterministic remediation can lower risk but does not substitute for review
- preserve truthful review queues and draft review metadata
- do not call an atom trusted unless review basis exists

## Runtime Probes And Audit Evidence

Runtime probes are repository-side evidence, not matcher monitor output.

Current registry layout:

- [scripts/auditlib/runtime_probe_plans](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probe_plans)

Rules:

- keep probe definitions family-scoped where possible
- if you touch a family anyway, prefer bundled remediation:
  - wrapper fixes
  - witness fixes
  - CDG fixes
  - manifest mapping
  - references
  - runtime probes
  - focused tests
- probe coverage should be deterministic and bounded
- parity probes are repo-local validation, not a substitute for matcher smoke

## `sciona ingest` Usage

Run ingestion from `../ageo-matcher`:

```bash
cd ../ageo-matcher
sciona ingest <source_file> --class <ClassName> --output ../ageo-atoms/ageoa/<domain>/<atom_name>
```

Common options:

- `--procedural`
- `--llm-provider <provider>`
- `--llm-model <model>`
- `--trace`
- grouped-output options when the family is intentionally published together

Rules:

- ingest directly to the final repo location when feasible
- do not treat a scratch tree as the committed artifact
- if grouped output is intended, make that explicit rather than accepting
  accidental one-symbol sprawl or accidental grouped overwrite

## Deterministic Audit Workflow

The audit stack must be rerun after meaningful ingest or remediation work.

### Ordered Baseline

Run in this order:

```bash
../ageo-matcher/.venv/bin/python scripts/build_audit_manifest.py
../ageo-matcher/.venv/bin/python scripts/validate_audit_manifest.py
../ageo-matcher/.venv/bin/python scripts/audit_structural.py
../ageo-matcher/.venv/bin/python scripts/audit_signature_fidelity.py
env PYTHON_JULIAPKG_PROJECT=/tmp/ageoa_juliapkg_project \
  JULIA_DEPOT_PATH=/tmp/ageoa_julia_depot \
  MPLCONFIGDIR=/tmp/mpl \
  ../ageo-matcher/.venv/bin/python scripts/audit_runtime_probes.py
../ageo-matcher/.venv/bin/python scripts/report_parity_coverage.py
../ageo-matcher/.venv/bin/python scripts/audit_return_fidelity.py
../ageo-matcher/.venv/bin/python scripts/audit_state_fidelity.py
../ageo-matcher/.venv/bin/python scripts/audit_generated_nouns.py
../ageo-matcher/.venv/bin/python scripts/audit_semantics.py
../ageo-matcher/.venv/bin/python scripts/audit_acceptability.py
../ageo-matcher/.venv/bin/python scripts/audit_risk.py
```

### Sequential Tail Constraint

These stages still rewrite shared manifest state and must remain sequential:

- `report_parity_coverage.py`
- `audit_semantics.py`
- `audit_acceptability.py`
- `audit_risk.py`

Do not parallelize them.

### Focused Family Work

For family-local work:

- run focused `pytest` slices first
- run the full ordered audit stack before declaring the repo truthful again

## Acceptance Rules

An ingest or remediation pass is acceptable only when all of these are true:

- stable artifacts are correct and committed
- ephemeral byproducts are excluded
- package exports are coherent
- public interfaces are honest
- witnesses and state models are semantically faithful
- CDG shape is valid and truthful
- manifest/provenance mappings are coherent
- references and uncertainty artifacts are preserved or updated when required
- runtime probes and deterministic audit evidence are refreshed
- the audit manifest reflects the new truth
- the repo is not left in a broken or misleading state

## Practical Working Rules

### Bundle Local Fixes

When the remaining risks are local and understandable, fix them together:

- signature/default drift
- weak type surface
- local CDG issues
- witness drift
- naming/alignment drift
- manifest mapping
- runtime probes

Do not split such work into unnecessary multiple passes.

### Keep Sequential Work Small

Reserve sequential-only mode for:

- matcher/ingester refinements
- shared auditlib changes
- audit-tail reruns
- cases where multiple tasks would fight over the same shared artifact

### Prefer Honest Narrowness Over False Confidence

If the structure is uncertain:

- emit conservative wrappers
- use explicit notes in manifest mappings
- avoid invented structured-return or witness semantics
- leave review basis to the proper review workflow

## Remaining Matcher Gaps

The repo-side remediation work identified a small set of ingester gaps worth
finishing matcher-side. See
[INGESTER_REMAINING_GAPS_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-atoms/INGESTER_REMAINING_GAPS_IMPLEMENTATION_PLAN.md)
for the concrete implementation plan.

Current notable matcher-side themes:

- conservative contract generation for array-normalizing wrappers
- proper callable/oracle contract generation
- conservative witness generation for stateful estimator families
- validation of ghost abstract imports
- intentionally narrow but still expandable smoke/regression/structured-return
  coverage
- grouped ingest ergonomics beyond replacement guards

## Final Rule

Do not stop at “the codegen ran.”

Stop only when:

- the emitted atom is honest
- the metadata is coherent
- the deterministic evidence is refreshed
- the repository truthfully reflects the new state of the atom
