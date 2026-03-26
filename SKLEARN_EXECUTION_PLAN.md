# sklearn Ingestion Execution Plan

This document turns [SKLEARN.md](/Users/conrad/personal/ageo-atoms/SKLEARN.md)
from a raw target inventory into a restart-safe ingestion program for the
current `sciona ingest` flow and deterministic audit stack.

It is written for agents that may stop and restart. Treat it as the
operational source of truth for sklearn sequencing.

## Current Baseline

Inventory status:

- target inventory exists in [SKLEARN.md](/Users/conrad/personal/ageo-atoms/SKLEARN.md)
- current inventory size: `272` targets across `32` sklearn modules

Repo status:

- there is no live tracked `ageoa/sklearn/` source tree in the current repo
- the committed audit manifest still contains `6` stale sklearn entries from
  older `calibration_iter7` / `calibration_iter8` exploratory ingests
- those stale entries are all currently `misleading`, `high` risk, and
  `review_now`
- those manifest entries point at missing source paths and should not be
  treated as active atoms

Implication:

- `SKLEARN.md` is still useful as the source backlog
- the old sklearn ingest outputs should not be resumed as-is
- the next sklearn pass should start from fresh ingests under the current
  matcher/audit system

## Global Rules

1. Keep `SKLEARN.md` as the upstream inventory document.
2. New sklearn ingests must use the current per-atom directory layout under
   `ageoa/sklearn/<atom_name>/` unless there is a strong compatibility reason
   not to.
3. Every batch must rerun the deterministic audit workflow before it is
   considered complete.
4. Do not treat matcher CLI success as acceptance; the local audit state is the
   authoritative judgment.
5. Favor narrowly-scoped, semantically honest wrappers over ambitious
   estimator families that collapse into pseudo-state or fabricated semantics.
6. Do not begin with meta-estimators, composition wrappers, or classes whose
   semantics are mostly learned-state orchestration.

## Difficulty And Risk Model

Each target should be placed into one of four execution tiers before ingest.

### Tier A: Low difficulty, lower audit risk

Traits:

- pure helper functions or simple transforms
- limited hidden state
- clear input/output contracts
- deterministic runtime probe is easy to write

Examples:

- `extract_patches_2d`
- `reconstruct_from_patches_2d`
- `img_to_graph`
- `grid_to_graph`
- `random_projection`
- `feature_selection` score functions such as `chi2`, `f_classif`,
  `f_regression`, `r_regression`

### Tier B: Moderate difficulty, moderate audit risk

Traits:

- stateful estimators or transformers with a tractable learned state
- standard `fit` / `transform` / `predict` split
- runtime probes feasible, but state semantics must be explicit

Examples:

- `SimpleImputer`
- `MissingIndicator`
- `VarianceThreshold`
- `CountVectorizer`
- `TfidfTransformer`
- `PCA`
- `TruncatedSVD`
- `KMeans`

### Tier C: High difficulty or high audit risk

Traits:

- estimator semantics depend on rich fitted state, implicit defaults, or
  multiple public modes
- wrappers are prone to generated-noun drift or pseudo-state
- runtime probes are possible but nontrivial

Examples:

- `CalibratedClassifierCV`
- `GaussianProcessRegressor`
- `KernelPCA`
- `FastICA`
- `IterativeImputer`
- `NearestNeighbors`-style classes

### Tier D: Very high difficulty, defer until the system is proven

Traits:

- meta-estimators or orchestration wrappers
- ensemble/composition semantics dominate over a simple atomic contract
- likely to produce misleading wrapper names or under-specified state models

Examples:

- `StackingClassifier`
- `StackingRegressor`
- `VotingClassifier`
- `VotingRegressor`
- `BaggingClassifier`
- `BaggingRegressor`
- `MultiOutput*`
- `OneVsRestClassifier`-style wrappers

## Module Grouping

Use the following grouping when selecting work.

### Group 1: Start here

- `sklearn.feature_extraction.image`
- `sklearn.random_projection`
- `sklearn.feature_selection` function targets
- `sklearn.covariance` function targets
- `sklearn.inspection` function targets only if probe design is straightforward

Why:

- mostly function-shaped APIs
- easier deterministic parity and runtime probes
- limited learned-state modeling burden

### Group 2: Good second wave

- `sklearn.impute`
- `sklearn.feature_extraction.text`
- `sklearn.preprocessing`
- `sklearn.decomposition` selected estimators
- `sklearn.kernel_approximation`

Why:

- manageable stateful wrappers
- good coverage of common sklearn semantics
- useful for validating state models and acceptability scoring on fitted state

### Group 3: Only after Group 1 and 2 are stable

- `sklearn.cluster`
- `sklearn.neighbors`
- `sklearn.tree`
- `sklearn.naive_bayes`
- `sklearn.kernel_ridge`
- `sklearn.discriminant_analysis`

Why:

- learned state is still tractable, but semantics and parity probes become
  more domain-specific

### Group 4: Defer

- `sklearn.calibration`
- `sklearn.ensemble`
- `sklearn.multiclass`
- `sklearn.multioutput`
- `sklearn.gaussian_process`
- `sklearn.gaussian_process.kernels`
- `sklearn.semi_supervised`
- `sklearn.neural_network`
- `sklearn.mixture`
- `sklearn.manifold`

Why:

- highest risk of wrappers that look plausible but are semantically thin
- likely to need human review and stronger provenance before they can clear the
  audit bar

## First-Wave Targets

These are the recommended first targets under the current system.

### Batch 1: Pure functions and graph/image helpers

- `sklearn.feature_extraction.image.extract_patches_2d`
- `sklearn.feature_extraction.image.reconstruct_from_patches_2d`
- `sklearn.feature_extraction.image.img_to_graph`
- `sklearn.feature_extraction.image.grid_to_graph`
- `sklearn.feature_selection.chi2`
- `sklearn.feature_selection.f_classif`
- `sklearn.feature_selection.f_regression`
- `sklearn.feature_selection.r_regression`
- `sklearn.covariance.empirical_covariance`
- `sklearn.covariance.shrunk_covariance`

Why this batch first:

- functions are easier to keep semantically faithful
- signatures are clearer
- state modeling is minimal or absent
- runtime probes are cheap to define

### Batch 2: Simple fitted transformers

- `sklearn.impute.SimpleImputer`
- `sklearn.impute.MissingIndicator`
- `sklearn.feature_selection.VarianceThreshold`
- `sklearn.feature_extraction.text.CountVectorizer`
- `sklearn.feature_extraction.text.TfidfTransformer`
- `sklearn.random_projection.GaussianRandomProjection`
- `sklearn.random_projection.SparseRandomProjection`

Why second:

- introduces fitted state without jumping straight to complex estimator
  orchestration
- good testbed for `state_models.py`, runtime probes, and parity classification

### Batch 3: Controlled dimensionality reduction / clustering

- `sklearn.decomposition.PCA`
- `sklearn.decomposition.TruncatedSVD`
- `sklearn.cluster.KMeans`
- `sklearn.cluster.kmeans_plusplus`
- `sklearn.cluster.k_means`

Why third:

- still broadly useful
- semantically richer than pure transforms
- good signal on whether the refined ingester handles learned model objects
  honestly

## Explicit Deferrals

Do not prioritize these early even though they are high-value APIs:

- `CalibratedClassifierCV`
- `GaussianProcessClassifier`
- `GaussianProcessRegressor`
- `StackingClassifier`
- `StackingRegressor`
- `VotingClassifier`
- `VotingRegressor`
- `BaggingClassifier`
- `BaggingRegressor`
- `HistGradientBoostingClassifier`
- `HistGradientBoostingRegressor`

Reason:

- they are exactly the classes most likely to generate misleading high-risk
  wrappers under the current audit definition

## Pre-Batch Cleanup

Before any new sklearn ingest work:

1. rebuild the audit manifest
2. identify the `6` stale sklearn manifest entries
3. remove or regenerate those entries so the sklearn portfolio reflects the
   live tree rather than deleted exploratory outputs
4. validate that `data/audit_reviews/` does not retain review drafts for
   missing sklearn atoms, or explicitly archive them as superseded

Exit criteria:

- no sklearn manifest row points at a missing `module_path`
- no seeded review draft exists only for a missing old sklearn artifact

## Standard Batch Procedure

Run this procedure for every sklearn batch.

### Inputs

- source target list from [SKLEARN.md](/Users/conrad/personal/ageo-atoms/SKLEARN.md)
- current matcher repo in `../ageo-matcher`
- current audit tooling in `scripts/`

### Steps

1. choose `3` to `10` targets from a single execution group
2. ingest each target into `ageoa/sklearn/<atom_name>/`
3. retain only durable artifacts:
   - `atoms.py`
   - `witnesses.py`
   - `state_models.py` when needed
   - `cdg.json`
   - `matches.json` when present
4. update `ageoa/sklearn/__init__.py` only after wrappers are importable
5. add authoritative source metadata or references where available
6. rerun deterministic audit tools
7. inspect semantic status, acceptability, and risk
8. if results are misleading or high-risk for semantic reasons, either:
   - narrow the wrapper scope
   - improve state/runtime evidence
   - defer the target instead of forcing it through

### Idempotent commands

```bash
cd ../ageo-matcher
sciona ingest <source> --class <ClassName> --output ../ageo-atoms/ageoa/sklearn/<atom_name>

cd ../ageo-atoms
python scripts/build_audit_manifest.py
python scripts/validate_audit_manifest.py
python scripts/audit_structural.py
python scripts/audit_signature_fidelity.py
python scripts/audit_runtime_probes.py
python scripts/audit_return_fidelity.py
python scripts/audit_state_fidelity.py
python scripts/audit_generated_nouns.py
python scripts/audit_semantics.py
python scripts/audit_acceptability.py
python scripts/audit_risk.py
python scripts/report_parity_coverage.py
```

## Per-Batch Exit Criteria

A batch is complete only when:

- all new sklearn atoms exist in the live tree
- the manifest contains the new atoms and no stale deleted atoms from the same
  batch
- structural status is not `fail`
- no target remains `misleading` without an explicit defer/repair decision
- review queue placement is understood and recorded
- durable provenance or reference context exists for anything likely to need
  Phase 4/7 review

## Restart Checklist

After any interruption, the next agent should do this first:

1. read [SKLEARN.md](/Users/conrad/personal/ageo-atoms/SKLEARN.md)
2. read [SKLEARN_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/SKLEARN_EXECUTION_PLAN.md)
3. rebuild and inspect `data/audit_manifest.json`
4. verify whether `ageoa/sklearn/` exists in the live tree
5. compare live sklearn atoms against manifest sklearn rows
6. continue from the earliest incomplete batch

## Immediate Next Step

The next concrete implementation slice should be:

1. clean the stale sklearn manifest and review state
2. ingest Batch 1 only
3. add focused runtime probes for Batch 1 targets
4. evaluate whether the new ingester produces semantically faithful sklearn
   wrappers before expanding to fitted estimators

## Handoff Notes

- if Batch 1 still produces mostly `misleading` outcomes, stop and refine the
  ingest recipe before scaling to more sklearn modules
- do not use the old `calibration_iter7` / `calibration_iter8` naming pattern
  as the new baseline
- if an estimator wrapper needs extensive narrative explanation to justify its
  semantics, it is probably not a good early target
