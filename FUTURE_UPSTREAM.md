# Future Upstream Work

This file tracks audit and provenance gaps that are not really "atom wrapper bugs"
but still keep medium-risk atoms from falling to low because our upstream mapping or
signature extraction is weaker than the repo-side evidence.

## Current Themes

### 1. Weak upstream anchors for Rust and C++ methods

In several families, `scripts/atom_manifest.yml` points to the correct upstream type
or method, but the fidelity pipeline still only records a weak anchor because the
upstream extractor cannot derive a durable callable signature from the source.

Examples:
- `ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics`
  - mapping now points to `models.ground_vehicles.bicycle_kinematic :: Model.get_derivatives`
  - risk still includes `RISK_WEAK_UPSTREAM_ANCHOR`
- `ageoa.pronto.state_estimator.update_state_estimate`
  - mapping points to `pronto_core.state_est :: StateEstimator::addUpdate`
  - still limited to a weak anchor because the wrapper abstracts a narrower pure state-update step
- `ageoa.pronto.backlash_filter.*`
  - exact C++ class-method mappings exist, but fidelity still treats them as weak anchors

Implication:
- the manifest is informative enough for provenance
- but not strong enough for the fidelity stage to count as a high-confidence upstream signature

Future work:
- extend `scripts/auditlib/upstream.py` so Rust/C++ mappings can produce a stronger
  normalized callable anchor even when they are not Python-importable
- distinguish "exact source-method anchor" from "importable Python signature" so the
  former is not collapsed into the latter

### 2. Script-level upstream algorithms need better structured provenance

Some wrappers correspond to logic that is inline in an upstream script rather than a
named callable. We can map them, but they still score as weak or unmapped because the
current pipeline expects a function-like anchor.

Examples:
- `ageoa.institutional_quant_engine.kalman_filter.kalmanfilterinit`
- `ageoa.institutional_quant_engine.kalman_filter.kalmanmeasurementupdate`
- several `institutional_quant_engine` script-backed atoms already mapped with `function: ~`

Implication:
- repo-side provenance is mostly correct
- fidelity still penalizes them because "script-level algorithm slice" is not a first-class anchor type

Future work:
- add a structured anchor kind for "script-level algorithm region" or "inline algorithm block"
- allow line-range or class-plus-constructor/update anchors to count as stronger provenance

### 3. Exact mapping does not mean low risk when name alignment remains poor

Several atoms now have valid upstream mappings and runtime parity, but still remain
medium because the generated public symbol names are too far from the upstream symbol
or conceptual unit.

Examples:
- `ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics`
- `ageoa.pronto.backlash_filter.initializebacklashfilterstate`
- `ageoa.pronto.backlash_filter.updatealphaparameter`
- `ageoa.pronto.backlash_filter.updatecrossingtimemaximum`

Implication:
- this is not just an upstream-extractor issue
- some of it is a naming/abstraction issue in older ingest output

Future work:
- keep tracking these as wrapper debt first
- only treat it as upstream-tooling work when the mapping is clearly right but the
  extractor cannot express the relationship in a way the audit accepts

## Near-Term Candidates

If we decide to improve upstream tooling next, these are the best concrete examples to test against:
- `ageoa.rust_robotics.bicycle_kinematic.evaluateandinvertdynamics`
- `ageoa.pronto.state_estimator.update_state_estimate`
- `ageoa.pronto.backlash_filter.initializebacklashfilterstate`
- `ageoa.pronto.backlash_filter.updatealphaparameter`
- `ageoa.pronto.backlash_filter.updatecrossingtimemaximum`
- `ageoa.institutional_quant_engine.kalman_filter.kalmanfilterinit`
- `ageoa.institutional_quant_engine.kalman_filter.kalmanmeasurementupdate`

## Boundary

This tracker is for:
- provenance extraction
- upstream anchor normalization
- fidelity scoring boundary issues

It is not for:
- generic review-basis debt
- parity fixture creation
- wrapper code rewrites unless they directly expose a missing upstream/provenance need
