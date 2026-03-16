# Atom Verification Playbook

This document contains a self-contained agent prompt for auditing and verifying
atom wiring in the `ageo-atoms` repository. Copy the prompt below into a new
conversation (or use it as a task description for an autonomous agent) whenever
atoms have been added or modified.

---

## When to run this

- After wiring new stub atoms to upstream implementations
- After bulk ingestion of new atom directories
- Before running `../ageo-matcher/sync_catalog.sh`
- As a pre-push gate on the `main` branch

---

## Agent prompt

```
You are auditing the ageo-atoms repository at /Users/conrad/personal/ageo-atoms.
This repo contains "atoms" — thin Python wrappers around library functions,
each decorated with icontract contracts and registered with a ghost witness
via @register_atom. The downstream consumer is ageo-matcher at
/Users/conrad/personal/ageo-matcher, which indexes these atoms into a
searchable catalog and runs ghost simulation to validate graph wiring.

Your task is to verify that all atoms are correctly wired and pass the
ingestion contract. Follow each phase below in order. Do NOT skip phases.

═══════════════════════════════════════════════════════════════════════════
PHASE 1: Run the automated verification scripts
═══════════════════════════════════════════════════════════════════════════

Run these two commands and report their output:

    python scripts/audit.py --verbose
    python ../ageo-matcher/scripts/verify_atoms_repo.py . --package ageoa

Expected result: 0 violations, all directories pass.

If either script reports failures, fix them before proceeding.
Reference: INGESTION.md section 14.1 for what each script checks.

═══════════════════════════════════════════════════════════════════════════
PHASE 2: Count remaining NotImplementedError stubs
═══════════════════════════════════════════════════════════════════════════

Run:

    grep -r "raise NotImplementedError" ageoa/ --include="atoms.py" \
        | grep -v "_ffi" | grep -v "FFI" | wc -l

Expected result: 0

Any non-FFI NotImplementedError in an atoms.py file means an atom is still
a stub and needs to be wired to a real implementation. FFI stubs (functions
named _*_ffi) are intentional placeholders for C++/Rust bridge code and
should be ignored.

Also check for singledispatch base cases (these are valid):

    grep -r "raise NotImplementedError" ageoa/ --include="*.py" \
        | grep -v "_ffi" | grep -v "FFI" | grep -v "atoms.py"

These may appear in tempo.py or models.py as @singledispatch base functions
or abstract base class methods — both are correct patterns.

═══════════════════════════════════════════════════════════════════════════
PHASE 3: Detect shadow witness stubs
═══════════════════════════════════════════════════════════════════════════

This is the MOST CRITICAL check. A shadow stub is a local function
definition that overrides a real witness import:

    # BAD — import is shadowed by the stub below it
    from .witnesses import witness_my_atom
    def witness_my_atom(*args, **kwargs): pass   # <-- SHADOW STUB

    @register_atom(witness_my_atom)  # binds to the no-op stub, not the real witness

Search for this pattern:

    grep -rn "def witness_.*\*args.*\*\*kwargs.*pass" ageoa/ --include="atoms.py"

Expected result: 0 matches.

If any are found, check whether the file also imports the same witness name
from .witnesses. If both exist, DELETE the local stub — the import is the
real witness and the stub shadows it, breaking the ghost witness system.

Also check for stubs without *args:

    grep -rn "def witness_.*pass$" ageoa/ --include="atoms.py"

═══════════════════════════════════════════════════════════════════════════
PHASE 4: Detect broken imports
═══════════════════════════════════════════════════════════════════════════

Check for imports from ageoa.ghost.abstract that reference types not defined
there (common: Graph, Node, Edge):

    python -c "
    import ast, sys
    from pathlib import Path
    abstract = Path('ageoa/ghost/abstract.py').read_text()
    tree = ast.parse(abstract)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    print('Defined in ghost.abstract:', sorted(defined))
    "

Then search for any imports from ghost.abstract that reference undefined names:

    grep -rn "from ageoa.ghost.abstract import" ageoa/ --include="atoms.py"

Cross-reference each imported name against the defined set. Flag any that
don't exist.

Also check for duplicate registry imports:

    grep -rn "from ageoa.ghost.registry import register_atom" ageoa/ --include="atoms.py" \
        | awk -F: '{print $1}' | sort | uniq -c | sort -rn | awk '$1 > 1'

Expected result: no files with count > 1.

═══════════════════════════════════════════════════════════════════════════
PHASE 5: Verify decorator ordering
═══════════════════════════════════════════════════════════════════════════

Per INGESTION.md section 4 and 7.1:
- @register_atom MUST be the outermost decorator (first decorator above def)
- @icontract.require ordering: isinstance/is not None innermost (closest to
  def), isfinite/shape checks outermost
- @icontract.ensure must exist (at least one per atom)

The audit.py script checks this, but verify manually for any edge cases:

    grep -B1 "^def [a-z]" ageoa/*/atoms.py ageoa/*/*/atoms.py \
        | grep -v "^--$" | grep -v "def _" | grep -v "def witness_" \
        | head -40

Spot-check that the line above each public def is @icontract (not
@register_atom appearing below an @icontract).

═══════════════════════════════════════════════════════════════════════════
PHASE 6: Verify test coverage for wired atoms
═══════════════════════════════════════════════════════════════════════════

Check for stale test_raises_not_implemented tests that should have been
converted when atoms were wired:

    grep -rn "test_raises_not_implemented" tests/

Expected result: 0 matches (unless the atom is genuinely still a stub).

For each match, check whether the corresponding atom still raises
NotImplementedError. If the atom is now wired, convert the test to
test_returns_result:

    def test_returns_result(self):
        result = my_atom(valid_input)
        assert isinstance(result, np.ndarray)

Per INGESTION.md section 12, every atom should have 5 test categories:
1. Positive path (correct inputs produce correct output)
2. Precondition violations (one per @require)
3. Postcondition verification (assert @ensure properties)
4. Edge cases
5. Upstream parity

Minimum acceptable: categories 1 and 2.

═══════════════════════════════════════════════════════════════════════════
PHASE 7: Run the test suite
═══════════════════════════════════════════════════════════════════════════

    python -m pytest tests/ -q \
        --ignore=tests/test_biosppy_advanced.py \
        --ignore=tests/test_biosppy_ecg.py \
        --ignore=tests/test_parity_biosppy.py \
        --ignore=tests/test_parity_e2e_ppg.py

The biosppy/e2e_ppg tests are excluded due to environment-level numpy/
matplotlib incompatibilities — they are not atom wiring issues.

Known pre-existing failures to ignore:
- test_alphafold.py::test_structural_pipeline (JAX shape broadcasting bug)
- test_parity.py::biosppy/ppg_kavsaoglu/detectonsetevents (env dependency)

Any NEW failures must be investigated and fixed.

═══════════════════════════════════════════════════════════════════════════
PHASE 8: Summary report
═══════════════════════════════════════════════════════════════════════════

Produce a summary with these fields:

    audit.py:               X/Y pass, Z violations
    verify_atoms_repo.py:   PASS/FAIL
    NotImplementedError:    N remaining stubs (excluding FFI)
    Shadow witness stubs:   N found
    Broken imports:         N found
    Duplicate registries:   N found
    Stale NIE tests:        N found
    Test suite:             X passed, Y failed, Z skipped

If all phases pass, the repo is ready for:
    ../ageo-matcher/sync_catalog.sh
```

---

## Reference files

| File | Purpose |
|---|---|
| `INGESTION.md` | Full ingestion contract: signature rules, contract patterns, witness system, CDG schema, test requirements, checklist |
| `scripts/audit.py` | AST-based static audit: type annotations, decorator ordering, docstrings, contract presence, CDG structure |
| `../ageo-matcher/scripts/verify_atoms_repo.py` | Completeness verifier: undefined symbols, missing imports, syntax errors |
| `ageoa/ghost/registry.py` | `@register_atom` decorator — binds impl to witness in global `REGISTRY` dict |
| `ageoa/ghost/abstract.py` | Abstract metadata types (`AbstractSignal`, `AbstractArray`, etc.) used by witnesses |
| `../ageo-matcher/ageom/architect/source_catalog.py` | How matcher discovers and indexes atoms from this repo |
| `../ageo-matcher/ageom/graph_store.py` | Witness and contract metadata extraction for Memgraph upsert |
| `../ageo-matcher/sources.yml` | Declares `ageo-atoms` as a source (`path: ../ageo-atoms`, `package: ageoa`) |

## Common failure patterns

### Shadow witness stubs (critical)
```python
# File imports real witness...
from .witnesses import witness_my_atom
# ...then shadows it with a no-op
def witness_my_atom(*args, **kwargs): pass
```
**Fix:** Delete the local stub. The import is the real witness.

### Broken abstract imports
```python
from ageoa.ghost.abstract import Graph, Node  # Graph and Node don't exist here
```
**Fix:** Use `Graph = Any` as a type alias instead, or remove the import.

### Stale test expecting NotImplementedError
```python
def test_raises_not_implemented(self):
    with pytest.raises(NotImplementedError):
        my_atom(np.array([1.0, 2.0, 3.0]))
```
**Fix:** Replace with `test_returns_result` that calls the atom and asserts the output type. Make sure the test input matches what the atom expects (e.g., 2D for adjacency matrices, 4+ points for spline fits).

### Duplicate registry imports
```python
from ageoa.ghost.registry import register_atom
from ageoa.ghost.registry import register_atom as _register_atom  # type: ignore
register_atom = cast(Callable[...], _register_atom)
```
**Fix:** Keep only the first import. Remove the alias and cast.

### No-op preconditions
```python
@icontract.require(lambda: True, "no preconditions for zero-parameter initializer")
```
**Fix:** Per INGESTION.md section 4.6, these provide zero validation. For zero-parameter initializers, use a postcondition-only pattern or add a meaningful structural check on the return value.
