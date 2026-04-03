**Purpose**
This plan covers two linked goals:

1. an immediate local refactor of [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py) so risk-remediation work can parallelize cleanly across family owners
2. a long-term migration path toward a sibling probe namespace such as `sciona.probes.physics`, rather than embedding probes under `sciona.atoms.*`

The target audience is planner/implementation agents that will execute the work in bounded phases.

**Why Now**
The current probe registry has become a coordination bottleneck:
- too many unrelated family edits land in one file
- sub-agents cannot safely take disjoint ownership without merge conflicts
- probe additions require broad context loading
- the current layout pushes work back toward sequential execution even when the underlying atoms are independent

The immediate refactor should preserve behavior while shrinking write scope. The longer-term namespace plan should keep probe machinery out of the public atom surface while staying compatible with future PEP 420 namespace repos.

**Design Position**
Keep the probe execution model centralized, but split family-specific probe plans out of the monolith.

Near-term:
- shared probe core remains centralized
- family/domain plan definitions move into separate modules
- a registry assembler merges all probe plans into one allowlist

Long-term:
- public atom surface lives under `sciona.atoms.<domain>`
- audit/runtime probe surface lives under `sciona.probes.<domain>`
- audit orchestration may later live under `sciona.audit.*` if it is split from repo-local scripts

Do not move probes under `sciona.atoms.<domain>.probes`. That would mix user-facing algorithm surface with internal validation machinery and make packaging/dependency boundaries worse.

**Phase A**
Immediate Local Refactor

Objective:
Split [runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py) into a stable core plus family-scoped plan modules without changing current audit behavior.

Likely files:
- [scripts/auditlib/runtime_probes.py](/Users/conrad/personal/ageo-atoms/scripts/auditlib/runtime_probes.py)
- new `scripts/auditlib/runtime_probes/` package
- [tests/test_audit_runtime_probes.py](/Users/conrad/personal/ageo-atoms/tests/test_audit_runtime_probes.py)

Target structure:
- `scripts/auditlib/runtime_probes/__init__.py`
- `scripts/auditlib/runtime_probes/core.py`
- `scripts/auditlib/runtime_probes/registry.py`
- `scripts/auditlib/runtime_probes/plans/`

Recommended first plan modules:
- `plans/numpy.py`
- `plans/scipy.py`
- `plans/biosppy.py`
- `plans/pronto.py`
- `plans/mcmc.py`
- `plans/molecular_docking.py`
- `plans/quantfin.py`
- `plans/institutional_quant_engine.py`
- `plans/mint.py`
- `plans/misc.py`

Ownership rule:
- `core.py` owns `ProbeCase`, `ProbePlan`, import helpers, probe execution, result shaping
- `registry.py` owns assembly of the plan dict
- each `plans/*.py` owns only family-specific helpers and plan definitions

Out of scope:
- no scoring redesign
- no change to evidence schema
- no change to the ordered audit pipeline
- no move to a public installable package yet

Deliverables:
- one behavior-preserving module split
- stable import path for current callers
- identical or near-identical audit outputs for unchanged atoms
- tests updated to target the new package layout

Exit criteria:
- `build_runtime_probe(...)` public behavior unchanged
- all current runtime probe tests pass
- at least 80% of family-specific plan definitions live outside `core.py`
- two sub-agents can safely edit different `plans/*.py` files in parallel without touching shared code

Implementation notes:
- keep `runtime_probes.py` as a thin compatibility wrapper initially if needed
- move helpers with narrow family relevance into the same family module
- keep generic validators in `core.py` only if they are truly reused across families
- prefer a small number of domain files over atom-per-file fragmentation

Risks:
- over-splitting into too many tiny files
- accidental circular imports between `core.py` and plan modules
- moving family-local validators into the wrong shared layer

**Phase B**
Parallel Remediation Enablement

Objective:
Use the new module layout to make family work parallel by default.

Planner guidance:
- assign one worker per family module where possible
- avoid cross-family workers that all touch `registry.py` at once
- batch registry changes in one integration pass if many workers add new plans simultaneously

Recommended coordination model:
- workers edit only `plans/<family>.py` and family wrapper/test files
- main agent integrates registry imports and reruns sequential audit stages

Deliverables:
- contributor guidance in the probe package header or README comments
- clear rule for where a new probe belongs
- minimal merge conflicts in multi-worker remediation rounds

Exit criteria:
- at least three family probe additions can be landed from disjoint workers with no overlapping write set outside final integration

**Phase C**
Future Namespace Packaging Model

Objective:
Define the long-term place of probes in a multi-repo PEP 420 namespace world.

Recommended namespace:
- atoms: `sciona.atoms.<domain>`
- probes: `sciona.probes.<domain>`

Examples:
- `sciona-atoms-physics` provides `sciona.atoms.physics`
- the same repo may also provide `sciona.probes.physics`
- `sciona-atoms-bio` provides `sciona.atoms.bio` and optionally `sciona.probes.bio`

Why this is preferable to `sciona.atoms.<domain>.probes`:
- keeps public algorithm surface separate from audit machinery
- avoids shipping audit dependencies as part of the main atom API by default
- makes agent discovery cleaner: discovered atoms are not mixed with probe modules
- keeps CI/test code in a sibling namespace that can evolve independently

Packaging guidance:
- use PEP 420 namespace packages for both `sciona.atoms` and `sciona.probes`
- domain repos may expose both namespaces from one checkout
- production users can install atom-only extras
- CI/audit jobs install the repo with an audit/dev extra that includes probe dependencies

CI guidance:
- repo CI should import `sciona.probes.<domain>` directly from the repo checkout
- central audit CI can aggregate domain probe registries across installed repos
- keep probe discovery explicit, not filesystem-magical

Search/discovery guidance:
- atom catalog search should expose only `sciona.atoms.*`
- agents may request probe surfaces only in audit or maintenance contexts
- probe namespaces should not appear as user-facing algorithm inventory

Out of scope:
- no immediate repackaging of this repo into `sciona.*`
- no current migration of audit scripts to import from `sciona.probes.*`

Deliverables:
- a namespace decision record
- future repo template guidance for domain repos
- a migration-compatible registry interface

Exit criteria:
- a future repo can add `sciona.probes.<domain>` without changing the central probe-core contract

**Phase D**
Migration Path

Objective:
Make the local refactor compatible with the future namespace packaging model.

Recommended migration sequence:
1. split current repo-local probe definitions into `scripts/auditlib/runtime_probes/plans/*.py`
2. stabilize a registry interface:
   - each plan module exports `get_probe_plans() -> dict[str, ProbePlan]`
3. later mirror that interface in packaged domain modules:
   - `sciona.probes.physics.get_probe_plans()`
4. teach central audit tooling to merge plan registries from one or more installed probe providers

Bridge rule:
- the local scripts-based registry remains the current source of truth until the packaged `sciona.probes.*` interface exists
- do not introduce dual truth sources during the first refactor

**Implementation Questions**
The next planner/implementer should answer:
- Which family split gives the biggest reduction in merge conflict first?
- Should `runtime_probes.py` remain as a wrapper module for one phase or be replaced immediately?
- Which validators should remain generic versus move into family plan modules?
- Is `misc.py` a temporary overflow bucket or should every current probe family get a dedicated file immediately?
- At what point should CI begin validating the future `sciona.probes.<domain>` interface, even if it is still repo-local only?

**Recommendation**
Do the work in this order:
1. Phase A local split, behavior-preserving
2. Phase B parallel remediation rounds using the new family files
3. write a short namespace decision doc for `sciona.probes.<domain>`
4. only then start any packaging-facing migration

That order gives immediate workflow benefit without forcing a packaging transition before the probe system is structurally stable.
