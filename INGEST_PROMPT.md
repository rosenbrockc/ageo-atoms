# Ingestion Agent Prompt

Primary task: use `sciona ingest` from `../ageo-matcher` to create or refresh
atoms in `ageoa/`.

Secondary task: leave this repository in a truthful, reviewable state. An
ingest is not complete because the matcher CLI succeeded. It is complete only
when the emitted atom artifacts, metadata, and audit state are all consistent.

This prompt is the shortest operational version of the ingestion contract.
See [INGESTION.md](/Users/conrad/personal/ageo-atoms/INGESTION.md) for the full policy.

## Success Condition

Treat an ingest as successful only when all of the following are true:

- the stable atom artifacts are written to the correct `ageoa/` location
- no transient ingest monitor files are being committed
- the atom interface is honest, typed, documented, and auditable
- the package exports and metadata sidecars are updated coherently
- the deterministic audit pipeline has been rerun in the required order
- the resulting risk/review state is acceptable for the intended change

## Environment

- run Python and tests with
  [../ageo-matcher/.venv/bin/python](/Users/conrad/personal/ageo-matcher/.venv/bin/python)
- run `sciona ingest` from `../ageo-matcher`
- treat this repo as the durable consumer of matcher output

## Default Output Layout

Prefer the per-atom directory layout for new work:

```text
ageoa/
  <domain>/
    __init__.py
    <atom_name>/
      __init__.py
      atoms.py
      witnesses.py
      state_models.py      # only when state is real and durable
      cdg.json
      matches.json         # when matcher grounding exists
      references.json      # when scholarly/provenance metadata exists
      uncertainty.json     # when perturbation analysis has been run
```

Legacy layouts remain valid for existing atoms. Do not churn legacy atoms into
new layout shapes unless the work genuinely requires it.

## Stable vs Ephemeral Files

Commit stable atom artifacts:

- `atoms.py`
- `witnesses.py`
- `state_models.py` when required
- `cdg.json`
- legacy `*_cdg.json` when the atom already uses that layout
- `matches.json` when matcher grounding produced it
- `references.json` when attribution/provenance exists
- `uncertainty.json` when uncertainty was measured
- relevant `__init__.py` updates

Do not commit ingest monitor sidecars:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `shared_context_metrics.json`
- local `logs/`
- local `.playwright-mcp/`

## Interface Rules

### `atoms.py`

Every public atom must satisfy these rules:

- fully type annotated
- no public `Any`, `*args`, or `**kwargs`
- registered with `@register_atom(...)`
- meaningful `@icontract.require` and `@icontract.ensure`
- honest public names that reflect the operation being exposed
- defaults, optionality, and callable/state parameters must match the wrapper
  behavior and the intended upstream semantics
- no fake overload collapse; semantically distinct helpers need distinct names
- no hidden mutable ambient state

Decorator ordering:

1. inexpensive `isinstance` checks nearest `def`
2. shape/domain checks next
3. expensive finiteness/value checks next
4. `@register_atom(...)` outermost

### `witnesses.py`

- use only abstract ghost types
- stay pure and side-effect free
- mirror the public atom interface honestly
- propagate shape/dtype/state semantics conservatively
- prefer state-shaped abstractions for estimator/filter wrappers
- do not emit conceptually unrelated ghost abstractions just because the
  emitter can

### `state_models.py`

- include only durable cross-call state
- do not hide configuration in undocumented fields
- name fields to match the public wrapper semantics

### `cdg.json`

- must be structurally valid
- must describe the actual decomposition
- must not contain duplicate anonymous child nodes when the public API has
  distinct helpers
- use compact, honest atomic nodes for refined-ingest adapters rather than
  placeholder orchestration

## Documentation Rules

Every public atom should end up with reviewable documentation, even if some of
that metadata is added post-ingest.

Minimum bar for the wrapper itself:

- docstring explains what the atom does
- inputs and outputs are documented in plain technical English
- stateful transitions are described honestly
- callable/oracle hooks are named and explained explicitly

Repository-side sidecars and metadata to preserve or add:

- `references.json` for scholarly attribution and provenance
- `uncertainty.json` for perturbation-based uncertainty evidence
- `matches.json` for matcher grounding output
- [scripts/atom_manifest.yml](/Users/conrad/personal/ageo-atoms/scripts/atom_manifest.yml)
  entry when upstream mapping is needed

Dejargonization requirement:

- public-facing descriptions should be understandable outside the immediate
  source discipline
- avoid placeholder text and implementation-noise prose
- if a lay or de-jargonized summary is added later, it must stay faithful to
  the technical wrapper contract rather than inventing simpler semantics

## Scholarly References And Provenance

When authoritative sources exist, ingestion work is not done until provenance
is reviewable.

Use:

- [scripts/add_reference.py](/Users/conrad/personal/ageo-atoms/scripts/add_reference.py)
- [scripts/build_references.py](/Users/conrad/personal/ageo-atoms/scripts/build_references.py)
- [scripts/atom_manifest.yml](/Users/conrad/personal/ageo-atoms/scripts/atom_manifest.yml)

Rules:

- prefer exact or curated upstream anchors over vague module-level notes
- map refined-ingest adapters to the closest honest upstream anchor and say so
- do not leave overloaded helpers under one ambiguous manifest entry
- preserve per-atom `references.json` when it already exists

## Uncertainty

When an atom has empirical perturbation coverage or is a good uncertainty
candidate, preserve or generate `uncertainty.json`.

Use:

- [scripts/measure_uncertainty.py](/Users/conrad/personal/ageo-atoms/scripts/measure_uncertainty.py)

Do not invent uncertainty sidecars by hand.

## Runtime Probes And Audit Evidence

Matcher ingest success is not acceptance. Repo-local deterministic evidence is
the acceptance layer.

For focused parity/runtime work, use the repo-local probe registry in:

- [scripts/auditlib/runtime_probe_plans](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probe_plans)

If you touch a family anyway, prefer bundled local cleanup:

- wrapper surface
- witness surface
- CDG shape
- manifest mapping
- runtime probes
- focused tests

## Required Audit Order

Run the deterministic audit stack in this order:

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

Do not run the tail stages in parallel. `report_parity_coverage.py`,
`audit_semantics.py`, `audit_acceptability.py`, and `audit_risk.py` still
rewrite shared manifest state and must remain sequential.

## Review And Trust Rules

Low risk is not the same thing as trusted.

- deterministic evidence can reduce risk and improve acceptability
- review basis is still a separate trust layer
- do not claim an atom is trusted just because it is low risk
- preserve and respect review workflow artifacts under
  [data/audit_reviews](/Users/conrad/personal/ageo-atoms/data/audit_reviews)

## Agent Procedure

1. Choose the final output path under `ageoa/`.
2. Run `sciona ingest` from `../ageo-matcher`.
3. Keep durable artifacts and discard monitor byproducts.
4. Fix local wrapper/witness/CDG issues immediately when they are file-scoped.
5. Update exports, manifest mappings, references, and uncertainty artifacts as
   needed.
6. Add or update runtime probes and focused tests when the family needs parity
   evidence.
7. Rerun the deterministic audit stack in order.
8. Inspect the resulting manifest, semantic status, acceptability, and risk.
9. Do not stop until the repo is left in a truthful, reviewable state.

## Acceptance Checklist

```text
Artifacts:
  [ ] Stable files are in the correct ageoa/ location
  [ ] No ingest monitor sidecars are committed
  [ ] Package exports are updated coherently

Interface:
  [ ] Public atoms are typed, registered, and honestly contracted
  [ ] Witnesses use valid abstract types only
  [ ] Stateful wrappers use explicit documented state
  [ ] CDG matches the actual wrapper decomposition

Metadata:
  [ ] matches.json kept when matcher grounding exists
  [ ] references.json preserved or added when authoritative sources exist
  [ ] atom_manifest.yml mapping updated when provenance matters
  [ ] uncertainty.json preserved or generated when appropriate

Audit:
  [ ] audit manifest rebuilt and validated
  [ ] structural/fidelity/probe audits rerun
  [ ] semantic/acceptability/risk tail rerun sequentially
  [ ] atom is not left broken, misleading, or falsely confident

Trust:
  [ ] risk state is acceptable for the intended change
  [ ] review-basis needs are not hand-waved away
```
