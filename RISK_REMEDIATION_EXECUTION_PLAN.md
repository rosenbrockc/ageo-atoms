# Risk Remediation Execution Plan

## Current Audit Snapshot

As of the latest ordered audit rerun:

- atoms: `505`
- `high`: `0`
- `medium`: `153`
- `low`: `352`

The medium queue is now dominated by a small number of repeated deficits rather than catastrophic breakage:

- `RISK_MISSING_REVIEW_BASIS`: `153`
- `RISK_LOW_NAME_ALIGNMENT`: `110`
- `RISK_MISSING_PARITY`: `97`
- `RISK_GENERATED_INGEST`: `78`
- `RISK_STRUCTURAL_PARTIAL`: `76`
- `RISK_FFI_BACKED`: `76`
- `RISK_WEAK_UPSTREAM_ANCHOR`: `65`
- `RISK_REFINED_INGEST`: `62`
- `RISK_STATEFUL_API`: `61`
- `RISK_SIGNATURE_MISMATCH`: `52`

This means the repo is no longer bottlenecked by grossly broken atoms. The remaining work is mostly one of:

1. removing the last parity debt from clustered families
2. fixing local wrapper/name/signature/type issues in families we already understand
3. addressing shared provenance/review-basis deficits


## Family Audit

### Tier A: Highest-leverage clustered families

These families are large enough that a coordinated pass can materially change the medium count:

- `ageoa.quantfin.monte_carlo_anti_d12`
  - `15` medium atoms
  - dominant deficits: `MISSING_PARITY`, `MISSING_REVIEW_BASIS`, `FFI_BACKED`, `REFINED_INGEST`, `WEAK_UPSTREAM_ANCHOR`
  - note: includes overloaded `process` symbols, so naming/identity work should be careful
- `ageoa.quantfin.rng_skip_d12`
  - `6` medium atoms
  - dominant deficits: `MISSING_PARITY`, `LOW_NAME_ALIGNMENT`, `UPSTREAM_UNMAPPED`
- `ageoa.pronto.yaw_lock`
  - `5` medium atoms
  - dominant deficits: `MISSING_PARITY`, `STATEFUL_API`, `GENERATED_INGEST`, `WEAK_UPSTREAM_ANCHOR`
- `ageoa.molecular_docking.greedy_mapping`
  - `5` medium atoms
  - dominant deficits: `MISSING_REVIEW_BASIS`, `SIGNATURE_MISMATCH`, `LOW_NAME_ALIGNMENT`, `STATEFUL_API`
- `ageoa.mcmc_foundational.mini_mcmc`
  - `5` medium atoms
  - dominant deficits: `MISSING_REVIEW_BASIS`, `WEAK_UPSTREAM_ANCHOR`, `MISSING_PARITY`, `STOCHASTIC`

### Tier B: Medium-size families with clean crossover potential

- `ageoa.biosppy.ecg_zz2018_d12`
- `ageoa.numpy.random_v2`
- `ageoa.scipy.sparse_graph_v2`
- `ageoa.quantfin.char_func_option_d12`
- `ageoa.pronto.dynamic_stance_estimator`
- `ageoa.pronto.backlash_filter`
- `ageoa.molecular_docking.quantum_solver_d12`
- `ageoa.institutional_quant_engine.almgren_chriss_v2`
- `ageoa.institutional_quant_engine.avellaneda_stoikov_d12`
- `ageoa.scipy.spatial_v2`

These are good candidates for small bundled remediation passes because they are already partly understood and several have already had parity debt reduced.

### Tier C: Cross-cutting generated-wrapper families

Single-atom or two-atom modules that still look like clean generated-wrapper cleanup rather than deep domain redesign:

- `ageoa.institutional_quant_engine.almgren_chriss`
- `ageoa.institutional_quant_engine.order_flow_imbalance`
- `ageoa.institutional_quant_engine.pin_model`
- `ageoa.molecular_docking.build_interaction_graph`
- `ageoa.particle_filters.basic`

These are good filler tasks for parallel lanes because they usually have:

- deterministic behavior
- small write sets
- no large family coordination burden

### Tier D: Sequential / special-case families

These should not be treated as normal parallel atom cleanup until a prerequisite is resolved:

- `ageoa.e2e_ppg.kazemi_wrapper_d12`
  - blocked by vendored `kazemi_peak_detection` visibility in the shared environment
- `ageoa.mcmc_foundational.advancedhmc.*`
  - blocked by `juliacall` lockfile / environment behavior
- overloaded-symbol `quantfin` families
  - especially `monte_carlo_anti_d12.process`
  - identity/naming issues are likely broader than a simple local parity pass
- families that still imply matcher-side changes
  - any wrapper where the remaining deficit clearly reflects a generator bug rather than local code debt


## Strategy Shift

Recent passes have often reduced score within `medium` without reducing the count of medium atoms. That is still useful, but it is no longer the most efficient default.

Going forward, remediation should be atom-centric and family-centric rather than parity-centric:

- when touching a family, fix all local, file-scoped deficits that are safe to fix together
- avoid revisiting the same family later for an obvious signature/type/name cleanup
- reserve parity-only passes for families where wrapper edits are risky or blocked

The target is a robust library, not just a smaller medium number. The plan below therefore prioritizes bundled local fixes, but still avoids unsafe multi-issue edits on ambiguous wrappers.


## Execution Model

### Parallel by default

The following work can run in parallel across multiple sub-agents if the write sets are disjoint:

- runtime/parity probe additions for different families
- local wrapper cleanup in unrelated families
- local upstream/provenance anchor cleanup in unrelated families
- review-basis draft generation for disjoint family sets
- isolated structural/CDG normalization for disjoint family directories

### Sequential only when unavoidable

Keep work sequential only for:

- full ordered audit reruns and final manifest/risk aggregation
- changes that require matcher / ingester refinement
- families that share the same files or same generated state model
- environment-fix work that affects all later probes
- overloaded-symbol resolution where naming policy must be chosen globally


## Phased Remediation Plan

### Phase 1: Parallel Crossover Pass

Objective:
- maximize `medium -> low` crossings using only local repo edits

Selection rule:
- choose families where removing one or two remaining deficits is likely to cross the threshold
- prefer deterministic wrappers and families already partly remediated

Parallel lanes:

- Lane A: refined-ingest deterministic families
  - `quantum_solver_d12`
  - `almgren_chriss_v2`
  - `avellaneda_stoikov_d12`
  - `scipy.spatial_v2`
- Lane B: generated deterministic wrappers
  - `almgren_chriss`
  - `order_flow_imbalance`
  - `pin_model`
  - `build_interaction_graph`
- Lane C: parity-heavy FFI families with low conceptual complexity
  - `rng_skip_d12`
  - clean `local_vol_d12` helpers
  - non-overloaded `monte_carlo_anti_d12` helpers

Required behavior:
- do not touch overloaded-symbol atoms in Lane C unless the worker owns the whole family pass
- when touching an atom, fix parity plus any obvious local signature/type/name defect in the same file

Exit criteria:
- all selected lane families rerun cleanly in focused tests
- one ordered audit rerun at the end
- family notes recorded for any atoms still blocked by generator issues

### Phase 2: Parallel Bundled Wrapper Cleanup

Objective:
- target medium families where parity is no longer dominant and local wrapper defects are now the main blocker

Candidate families:
- `biosppy.online_filter*`
- `queue_estimator`
- `fasta_dataset`
- `greedy_mapping`
- `minimize_bandwidth`

Work types:
- signature/default alignment
- weak public type cleanup
- obvious low-name-alignment fixes
- local witness/import/decorator cleanup
- small CDG normalization where clearly local

Parallel lanes:

- Lane A: BioSPPy wrapper families
- Lane B: quant / docking state-helper families
- Lane C: mint / dataset families

Sequential note:
- only combine structural changes when they are local to the family and do not require matcher changes

Exit criteria:
- targeted families each receive a single bundled pass instead of repeated piecemeal revisits

### Phase 3: Provenance and Review-Basis Coverage

Objective:
- address the deficits that now appear on nearly every remaining medium atom

Dominant repo-wide blocker:
- `RISK_MISSING_REVIEW_BASIS` on all `153` medium atoms

Parallel lanes:

- Lane A: provenance anchor improvements
  - Rust / C++ / script-backed mappings
  - weak upstream anchors
  - unmapped upstream rows
- Lane B: structured review-basis generation
  - generate or curate review records for disjoint family groups
- Lane C: final parity cleanup for any leftover easy deterministic atoms

Why this phase matters:
- parity-only work is no longer enough to reduce medium count broadly
- review-basis and provenance are now the main shared blockers

Sequential note:
- final review merge / validation stays sequential

### Phase 4: Stateful / FFI Family Remediation

Objective:
- tackle the remaining medium clusters that are conceptually sound but still carry stateful API or FFI debt

Candidate families:
- `pronto.yaw_lock`
- `pronto.dynamic_stance_estimator`
- `pronto.backlash_filter`
- `mini_mcmc`
- `advancedhmc` only if environment blockers are resolved

Parallel lanes:

- Lane A: `pronto` stateful families
- Lane B: `mcmc_foundational` families
- Lane C: `quantfin` FFI families without overloaded-symbol ambiguity

Sequential note:
- if a family’s state model or API mismatch clearly reflects matcher output policy, stop local surgery and route that issue into `FUTURE_INGESTER.md`

### Phase 5: Hard Families and Policy Decisions

Objective:
- finish the remaining mediums that cannot be reduced by local deterministic work alone

Families likely to land here:
- overloaded-symbol `quantfin` families
- vendored-environment families like `kazemi_wrapper_d12`
- stochastic/refined-ingest families where local edits no longer look trustworthy
- families needing global naming or matcher identity decisions

This phase is intentionally smaller and more sequential.

Allowed sequential work:
- matcher / ingester refinement
- environment modeling fixes
- family-wide naming policy choices
- overloaded-symbol identity decisions


## Suggested Sub-Agent Topology

Use multiple workers with disjoint family ownership.

Recommended pattern:

- Worker 1: `quantfin` deterministic helper families
- Worker 2: `institutional_quant_engine` deterministic/generated wrappers
- Worker 3: `molecular_docking` refined-ingest families
- Worker 4: `pronto` stateful families
- Worker 5: provenance / upstream anchor cleanup
- Main agent: integration, ordered audit reruns, commit boundaries, and escalation for matcher-side issues

Rules:

- each worker owns a disjoint directory or family set
- no worker should edit shared audit scripts unless that worker’s task is explicitly the shared probe lane
- only the main agent runs the final ordered audit stack and prepares the commit sequence


## Immediate Next Batch Recommendation

Best next batch, using this plan:

1. Parallel:
   - Worker A: `molecular_docking.build_interaction_graph`
   - Worker B: `rng_skip_d12` / non-overloaded `monte_carlo_anti_d12` helper cleanup
   - Worker C: `pronto.backlash_filter`
2. Main agent:
   - integrate
   - run one ordered audit rerun
   - choose the next crossover candidates based on the updated queue

Why this batch:
- it avoids environment-blocked families
- it avoids overloaded-symbol policy work
- it keeps write sets disjoint
- it has a good chance of reducing `medium` count instead of only reducing scores within `medium`


## Stop Conditions

Stop parallel local remediation and switch to sequential / matcher work only when one of these is true:

- a family’s remaining deficits are mostly `REVIEW_BASIS + PROVENANCE`, not local wrapper debt
- a family needs a global naming policy
- the wrapper is blocked by shared environment visibility
- the remaining issue clearly reflects a generator bug instead of a repo-local defect

At that point, record the blocker explicitly and avoid repeated local edits that only churn audit scores.
