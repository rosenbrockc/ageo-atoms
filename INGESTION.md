# Atom Ingestion Guide

This document specifies the exact procedure for wrapping an existing library
function (numpy, scipy, etc.) as a verified atom in the `ageoa` package.
Every atom must satisfy the rules below so that the
[ageo-matcher](https://github.com/rosenbrockc/ageo-matcher) pipeline can
index, match, assemble, type-check, and export it correctly.

---

## Table of contents

1. [Terminology](#1-terminology)
2. [File placement](#2-file-placement)
3. [Signature rules](#3-signature-rules)
4. [Preconditions (`@require`)](#4-preconditions-require)
5. [Postconditions (`@ensure`)](#5-postconditions-ensure)
6. [Helper functions](#6-helper-functions)
7. [Ghost Witness system](#7-ghost-witness-system)
8. [Docstrings](#8-docstrings)
9. [Module exports](#9-module-exports)
10. [CDG (Computational Dependency Graph) schema](#10-cdg-computational-dependency-graph-schema)
11. [Complete example: `scipy.linalg.lu_factor`](#11-complete-example-scipylinalglu_factor)
12. [Required tests per atom](#12-required-tests-per-atom)
13. [Test template](#13-test-template)
14. [Checklist](#14-checklist)
15. [DSP-specific contract patterns](#15-dsp-specific-contract-patterns)
16. [Bayesian atom patterns](#16-bayesian-atom-patterns)
17. [Skeleton atoms](#17-skeleton-atoms)

---

## 1. Terminology

| Term | Meaning |
|---|---|
| **Atom** | A thin Python wrapper around a single library function, decorated with icontract `@require` / `@ensure` and registered via `@register_atom`. |
| **Precondition** | An `@icontract.require` decorator that guards inputs. Analogous to a Lean hypothesis. |
| **Postcondition** | An `@icontract.ensure` decorator that asserts properties of the return value. Analogous to a Lean proof obligation. |
| **Ghost Witness** | A lightweight function that accepts and returns abstract metadata types (never real data). The ghost simulator executes witnesses in topological order to validate graph wiring before any heavy computation runs. |
| **Abstract type** | A Pydantic model from `ageoa.ghost.abstract` that carries only metadata (shape, dtype, domain, etc.) — never actual sample data. |
| **CDG** | Computational Dependency Graph — a JSON structure of `AlgorithmicNode` objects linked by `DependencyEdge` objects. Describes how atoms compose into a pipeline. |
| **Declaration** | The `ageom.types.Declaration` object the matcher produces when it indexes this atom. Contains `name`, `type_signature`, `docstring`, `raw_code`, `source_lib`, and `prover="python"`. |
| **Skeleton** | `raise NotImplementedError("Skeleton for future ingestion.")` — a placeholder atom that has contracts and a witness but no implementation yet. The Python equivalent of Lean's `sorry`. |

---

## 2. File placement

```
ageoa/
  {library}/           # top-level library name, e.g. numpy, scipy
    __init__.py        # re-exports public atoms
    {submodule}.py     # groups atoms by upstream submodule
    witnesses.py       # ghost witness functions for this domain
  ghost/
    abstract.py        # abstract metadata types
    registry.py        # @register_atom decorator and global REGISTRY
    simulator.py       # ghost graph simulator
    witnesses.py       # shared cross-domain witnesses
```

Mirror the upstream package structure. `numpy.linalg.solve` becomes
`ageoa/numpy/linalg.py : solve`. This ensures the fully-qualified atom name
(`ageoa.numpy.linalg.solve`) is predictable and matches what the indexer
produces.

Witness functions live in a `witnesses.py` file alongside the atoms they
serve — either in the same domain package (e.g., `ageoa/scipy/witnesses.py`)
or in `ageoa/ghost/witnesses.py` for cross-domain witnesses.

---

## 3. Signature rules

The matcher's `PythonDeclarationSource` walks the AST to extract a
`type_signature` string like `(a: np.ndarray, b: np.ndarray) -> np.ndarray`.
This signature is the primary key for embedding-based retrieval and
type-checking. Getting it wrong makes the atom invisible to the matcher.

### 3.1 Use concrete types, never `Any`

**Bad:**
```python
def dot(a: Any, b: Any, out: Any = None) -> Any:
```

**Good:**
```python
def dot(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
```

`Any` erases all type information. mypy cannot validate code that calls
`dot(a, b)` if the return type is `Any`, so the type-check step of the
verification loop becomes a no-op. Use the most specific type that the
upstream function actually accepts.

When the upstream function is genuinely polymorphic (e.g., accepts both
scalars and arrays), use `Union` or `overload`:

```python
from typing import Union
def norm(x: np.ndarray, ord: int | float | None = None) -> Union[np.floating, np.ndarray]:
```

### 3.2 No `*args` in the public signature

The indexer walks `ast.FunctionDef.args.args` (positional parameters). It
does **not** read `args.vararg`. A function like:

```python
def rand(*size: int) -> np.ndarray:   # indexed as () -> np.ndarray
```

produces an empty parameter list, making it unmatchable. Rewrite to use an
explicit parameter:

```python
def rand(size: int | tuple[int, ...] | None = None) -> np.ndarray:
    if size is None:
        return np.random.rand()
    if isinstance(size, int):
        return np.random.rand(size)
    return np.random.rand(*size)
```

### 3.3 No `**kwargs` in the public signature

The indexer does not read `args.kwarg`. Any parameter hidden behind
`**kwargs` is invisible to the matcher. Expose every parameter you intend
to support as an explicit keyword argument with a type annotation and
default value:

```python
# Bad: **kwargs is invisible
def array(object: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:

# Good: explicit parameters
def array(
    object: np.ndarray | list | tuple,
    dtype: np.dtype | type | None = None,
    copy: bool | None = None,
    order: str | None = None,
) -> np.ndarray:
```

### 3.4 Keyword-only arguments are fine but must be annotated

The indexer currently reads `args.args` only. A planned fix will also walk
`args.kwonlyargs`. In the meantime, any parameter that must be keyword-only
should still carry a type annotation so it is ready when the indexer is
updated:

```python
def vstack(
    tup: Sequence[np.ndarray],
    *,
    dtype: np.dtype | None = None,
    casting: str = "same_kind",
) -> np.ndarray:
```

### 3.5 Return type is mandatory

Every atom must have a `-> T` return annotation. Without it the
`type_signature` will lack a return type and mypy cannot validate callers.

---

## 4. Preconditions (`@require`)

Preconditions guard inputs. They must:

1. **Be a single lambda that references only the function's parameters.**
   The lambda's parameter names must exactly match a subset of the
   function's parameters (icontract binds by name).

2. **Include a human-readable description string** as the second argument.
   This string propagates into `Declaration.raw_code` and is shown to the
   LLM repair agent when a contract is violated.

3. **Be non-redundant.** Do not add a precondition that is a strict subset
   of another. For example, if you have
   `@require(lambda a: _is_square_2d(a))` and `_is_square_2d` already
   checks `a.ndim == 2`, do not add a separate
   `@require(lambda a: a.ndim == 2)`.

4. **Not call the function under test or any function with side effects.**

5. **Reference only helpers defined *above* the decorator in the file.**
   icontract evaluates lambdas lazily (at call time, not decoration time),
   so forward references happen to work. But they break static analysis
   tools, confuse readers, and will fail if icontract ever changes its
   evaluation strategy. Always define helpers before the first function
   that uses them.

6. **Never use no-op placeholders.** Contracts like `lambda c1, c2: True`
   provide zero validation. Every `@require` must check something
   meaningful.

### Precondition patterns

| Scenario | Pattern |
|---|---|
| Non-null guard | `@require(lambda x: x is not None, "x must not be None")` |
| Type guard | `@require(lambda data: isinstance(data, np.ndarray), "data must be ndarray")` |
| Shape guard | `@require(lambda a: a.ndim == 2, "a must be 2D")` |
| Non-empty | `@require(lambda a: a.shape[0] > 0, "a must be non-empty")` |
| Finite values | `@require(lambda a: np.isfinite(a).all(), "a must contain only finite values")` |
| Dimensional match | `@require(lambda a, b: a.shape[1] == b.shape[0], "inner dimensions must match")` |
| Square matrix | `@require(lambda a: a.shape[0] == a.shape[1], "a must be square")` |
| Range guard | `@require(lambda low, high: low <= high, "low must be <= high")` |
| Complex logic | Extract to a named helper (see [section 6](#6-helper-functions)) |

### Decorator ordering (critical)

icontract evaluates `@require` decorators **bottom-up** — the decorator
closest to the function `def` runs **first**. This means safety checks
must be innermost so they short-circuit before property checks that would
crash on invalid input:

```python
# CORRECT: isinstance runs first (innermost), isfinite runs last (outermost)
@register_atom(witness_fn)
@icontract.require(lambda data: np.isfinite(data).all(), "data must be finite")
@icontract.require(lambda data: data.shape[0] > 0, "data must be non-empty")
@icontract.require(lambda data: data.ndim >= 1, "data must be at least 1D")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be ndarray")
def my_atom(data: np.ndarray) -> np.ndarray:
```

**If you put `isfinite` closest to `def`, passing `None` will crash with
`TypeError` instead of raising `icontract.ViolationError`.**

---

## 5. Postconditions (`@ensure`)

Postconditions assert properties of the return value. They are the Python
analogue of a proof obligation: they state *what* the function guarantees,
not *how* it computes the result.

**Every atom must have at least one `@ensure` decorator.** An atom with
only preconditions proves nothing about its output — it is equivalent
to a Lean theorem with hypotheses but no conclusion.

### `@ensure` calling convention

The first parameter of the lambda is always `result` (the return value).
Subsequent parameters are bound to the function's input parameters by name:

```python
@icontract.ensure(
    lambda result, a, b: result.shape[0] == a.shape[1],
    "result has correct leading dimension",
)
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
```

### Postcondition patterns

| Scenario | Pattern |
|---|---|
| Output shape | `@ensure(lambda result, a: result.shape == a.shape, "shape preserved")` |
| Output dtype | `@ensure(lambda result: result.dtype == np.float64, "result is float64")` |
| Output range | `@ensure(lambda result: np.all(result >= 0), "result is non-negative")` |
| Consistency | `@ensure(lambda result, a, b: np.allclose(a @ result, b), "a @ result == b")` |
| Dimensionality | `@ensure(lambda result, a: result.ndim == a.ndim, "dimensionality preserved")` |
| Non-empty | `@ensure(lambda result: result.size > 0, "result is non-empty")` |

### When a postcondition is too expensive

Some postconditions (like `np.allclose(a @ result, b)` for `solve`) perform
a full matrix multiply on every call. For these, use icontract's `enabled`
flag to allow runtime opt-out:

```python
import os
_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"

@icontract.ensure(
    lambda result, a, b: np.allclose(a @ result, b, atol=1e-6),
    "solution satisfies a @ x == b",
    enabled=_SLOW_CHECKS,
)
```

**All expensive postconditions must be gated by `_SLOW_CHECKS`.** This
includes round-trip checks (`np.allclose(np.square(result), x)`) and
value-level consistency checks. Shape-only postconditions are O(1) and
should always be enabled.

The matcher's mypy-based verifier does not execute postconditions (it only
type-checks), so this flag has no effect on the verification pipeline. The
flag is for runtime use only.

### Shape-only postconditions are always cheap

Asserting `result.shape == (n, m)` is O(1) and should always be enabled.
Prefer shape postconditions as the mandatory minimum; add value-level
postconditions as an additional layer.

---

## 6. Helper functions

Complex precondition logic should be extracted into a named helper function
prefixed with `_`:

```python
def _is_square_2d(a: np.ndarray) -> bool:
    """Check that a is a 2D square matrix."""
    return a.ndim == 2 and a.shape[0] == a.shape[1]

@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
def inv(a: np.ndarray) -> np.ndarray:
    ...
```

Rules:
- The helper must be defined **above** all functions that reference it.
- The helper name must start with `_` (the indexer skips `_`-prefixed
  functions, so helpers will not be indexed as atoms).
- The helper must be a pure function (no side effects, no I/O).
- The helper's parameters must have type annotations.

---

## 7. Ghost Witness system

Every atom must be registered with a ghost witness via the `@register_atom`
decorator. This is how the matcher's ghost simulator validates graph wiring.

### 7.1 Registration

```python
from ageoa.ghost.registry import register_atom
from ageoa.my_domain.witnesses import witness_my_atom

@register_atom(witness_my_atom)
@icontract.require(...)
@icontract.ensure(...)
def my_atom(data: np.ndarray) -> np.ndarray:
    ...
```

`@register_atom` must be the **outermost** decorator (above all
`@icontract` decorators). It stores the implementation, witness, docstring,
both signatures (witness and heavy), and module name in the global
`REGISTRY`.

An optional `name` keyword overrides the registry key:
```python
@register_atom(witness_fn, name="scipy.stats.describe")
```

### 7.2 Witness function rules

Witness functions must:
1. Accept and return **abstract types only** — never `np.ndarray` or
   concrete data
2. Have **type annotations** on all parameters and return type (the
   registry reads `witness.__annotations__`)
3. Propagate shape/dtype metadata faithfully
4. Raise `ValueError` on structural violations (shape mismatch, domain
   mismatch, etc.)
5. Be **pure** — no side effects, no I/O, no randomness

### 7.3 Available abstract types

All types are defined in `ageoa.ghost.abstract`:

| Type | Use case | Key fields |
|---|---|---|
| `AbstractSignal` | DSP atoms (time/freq domain) | `shape`, `dtype`, `sampling_rate`, `domain`, `units` |
| `AbstractArray` | Generic array atoms | `shape`, `dtype`, `is_sorted`, `min_val`, `max_val` |
| `AbstractScalar` | Scalar-returning atoms | `dtype`, `min_val`, `max_val`, `is_index` |
| `AbstractMatrix` | Symbolic-dimension matrices | `shape` (tuple of strings), `dtype` |
| `AbstractBeatPool` | Beat detection pipelines | `size`, `is_calibrated`, `calibration_threshold` |
| `AbstractDistribution` | Bayesian atoms | `family`, `event_shape`, `batch_shape`, `support`, `is_discrete` |
| `AbstractRNGState` | Stochastic atoms | `seed`, `consumed`, `is_split` |
| `AbstractMCMCTrace` | MCMC chain atoms | `n_samples`, `n_chains`, `param_dims`, `warmup_steps`, `accept_rate` |
| `AbstractFilterCoefficients` | Filter design atoms | `order`, `btype`, `format`, `is_stable` |
| `AbstractGraphMeta` | Graph signal processing | `n_nodes`, `is_symmetric` |

### 7.4 Witness examples

**Simple array atom:**
```python
def witness_my_sort(data: AbstractArray) -> AbstractArray:
    """Witness for my_sort."""
    return AbstractArray(shape=data.shape, dtype=data.dtype, is_sorted=True)
```

**DSP atom (domain transition):**
```python
def witness_fft(sig: AbstractSignal) -> AbstractSignal:
    """Witness for fft — time domain to frequency domain."""
    sig.assert_domain("time")
    return AbstractSignal(
        shape=sig.shape,
        dtype="complex128",
        sampling_rate=sig.sampling_rate,
        domain="freq",
        units=sig.units,
    )
```

**Scalar-returning atom:**
```python
def witness_run_simulation(model, claim, seed: int, trials: int,
                           anti: bool, simulator_name: str) -> AbstractScalar:
    """Witness for run_simulation."""
    return AbstractScalar(dtype="float64")
```

### 7.5 One witness per atom

Every atom must have its **own** witness function. Do not reuse another
atom's witness (e.g., do not bind `hamilton_segmenter` to
`witness_ssf_segmenter`). Even if the abstract behavior is identical,
each atom needs a distinct witness for traceability.

### 7.6 Ghost simulator integration

The ghost simulator in `ageom/synthesizer/ghost_sim.py` has a hard-coded
list of atom modules that it imports to trigger `@register_atom`
decorators. When adding a new domain package:

1. Ensure all atoms are importable via the package's `__init__.py`
2. The ingester's ghost simulation bridge will auto-discover registered atoms

The simulator also maintains:
- **`_PARAM_DEFAULTS`**: heuristic scalar values for witness testing
  (e.g., `order=4`, `n=1024`, `fs=44100.0`). New atoms with non-standard
  scalar parameters may need entries here.
- **`_ATOM_ERROR_FACTORS`**: per-atom error expansion multipliers for
  precision gradient propagation. Atoms in precision-sensitive chains
  need entries here.

---

## 8. Docstrings

Every atom must have a Google-style docstring with `Args:` and `Returns:`
sections. The docstring is stored in `Declaration.docstring` and used for
embedding-based semantic search. A missing or vague docstring degrades
retrieval quality.

```python
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the linear system a @ x = b for x.

    Computes the exact solution of the well-determined linear matrix
    equation using Lower-Upper triangular decomposition (LU) with
    partial pivoting.

    Args:
        a: Coefficient matrix, shape (n, n). Must be square and
            non-singular.
        b: Ordinate values, shape (n,) or (n, k).

    Returns:
        Solution array x with shape matching b.
    """
```

Rules:
- First line: imperative summary of what the function computes.
- Mention mathematical operation or algorithm by name when applicable.
- **Spell out all abbreviations on first use.** Write "Finite Impulse
  Response (FIR)" not just "FIR". Write "Discrete Fourier Transform (DFT)"
  not just "DFT". The abbreviation may be used without expansion after
  the first occurrence.
- **Explain domain-specific jargon inline.** If a term like
  "Hermitian-symmetric" or "antithetic variates" appears, add a brief
  parenthetical explanation:
  `"Hermitian-symmetric (i.e., symmetric under conjugation)"`.
- Args section: one entry per parameter, including shape and constraints.
- Returns section: describe the output shape and semantics.
- Do not repeat the icontract decorators in prose. The contracts are
  machine-readable; the docstring is for humans and embedding models.

---

## 9. Module exports

Every atom must be listed in its module's `__init__.py`:

```python
# ageoa/numpy/__init__.py
from .linalg import solve, inv, det, norm
from .arrays import array, zeros, dot, vstack, reshape
from . import linalg
from . import random

__all__ = [
    "array", "zeros", "dot", "vstack", "reshape",
    "linalg", "random",
]
```

Also list submodule imports in the top-level `ageoa/__init__.py`:

```python
from . import numpy

__all__ = ["numpy"]
```

### Why this matters

The matcher runs `PythonDeclarationSource.get_declarations_from_package("ageoa")`
which calls `pkgutil.walk_packages`. If a module is not importable (missing
from `__init__.py` or has a top-level import error), its atoms will be
silently skipped.

**Additionally**, `@register_atom` decorators fire at import time. If an
atom is defined in a submodule (e.g., `controls.py`) but not imported in
`__init__.py`, the ghost simulator will not see it. Ensure all atoms —
including those in non-standard submodules like `controls.py`,
`num_methods.py`, or `montecarlo.py` — are imported somewhere in the
package init chain.

---

## 10. CDG (Computational Dependency Graph) schema

When atoms compose into a pipeline, the pipeline is described by a CDG
JSON file. CDGs are validated by the matcher's handoff system.

### 10.1 Schema

A CDG file contains:

```json
{
  "nodes": [ ... ],
  "edges": [ ... ],
  "metadata": { "source": "ingester", "class_name": "..." }
}
```

### 10.2 AlgorithmicNode fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `node_id` | string | yes | Unique identifier |
| `parent_id` | string or null | yes | null for root node |
| `name` | string | yes | Human-readable name |
| `description` | string | yes | **Must be non-empty for atomic leaves** |
| `concept_type` | string | yes | One of the `ConceptType` enum values (see below) |
| `inputs` | list[IOSpec] | yes | **Must be non-empty for atomic leaves** (strict validation) |
| `outputs` | list[IOSpec] | yes | **Must be non-empty for atomic leaves** (strict validation) |
| `status` | string | yes | `"decomposed"` for parent, `"atomic"` for leaves |
| `children` | list[string] | yes | child node_ids for decomposed nodes; empty for atomic |
| `depth` | int | yes | 0 for root, 1 for children, etc. |
| `type_signature` | string | yes | **Must be non-empty for atomic leaves**; format: `(param: type) -> return_type` |
| `matched_primitive` | string or null | no | Matched library primitive name |
| `is_optional` | bool | no | Default false. Config-gated branches |
| `is_opaque` | bool | no | Default false. Deep learning boundary |
| `is_external` | bool | no | Default false. External tool call |
| `parallelizable` | bool | no | Default false |
| `conceptual_summary` | string | no | Domain-agnostic summary for cross-domain retrieval |
| `critic_notes` | string | no | |
| `decomposition_rationale` | string | no | |

### 10.3 IOSpec fields

```json
{
  "name": "filtered",
  "type_desc": "np.ndarray",
  "constraints": "1D bandpass-filtered ECG signal"
}
```

**Constraints must be non-empty** for atomic leaves. Empty constraint
strings indicate missing documentation.

### 10.4 DependencyEdge fields

| Field | Type | Notes |
|---|---|---|
| `source_id` | string | Must reference an existing node |
| `target_id` | string | Must reference an existing node |
| `output_name` | string | Which output port of source |
| `input_name` | string | Which input port of target |
| `source_type` | string | Type of data flowing out |
| `target_type` | string | Expected type at target |
| `requires_glue` | bool | True if source/target types differ |

### 10.5 ConceptType values

Standard: `sorting`, `searching`, `divide_and_conquer`, `greedy`,
`dynamic_programming`, `graph_traversal`, `graph_optimization`,
`string_matching`, `geometry`, `arithmetic`, `number_theory`,
`combinatorics`, `algebra`, `analysis`, `set_theory`, `signal_transform`,
`signal_filter`, `graph_signal_processing`, `neural_network`, `custom`,
`external_tool`.

Bayesian: `sampler`, `log_prob`, `posterior_update`,
`variational_inference`, `prior_init`, `prior_distribution`,
`likelihood_evaluation`, `probabilistic_oracle`, `oracle_gradient`,
`mcmc_kernel`, `mcmc_proposal`, `vi_elbo`, `sequential_filter`,
`smc_reweight`, `message_passing`, `conjugate_update`.

### 10.6 Structural rules

1. **No circular edges.** The graph must be a DAG (directed acyclic graph).
   Template extraction cannot depend on peak correction if peak correction
   depends on template extraction.
2. **No duplicate edges.** Each (source_id, target_id, output_name,
   input_name) tuple must be unique. Do not create parallel edges with
   different type annotations for the same data flow.
3. **No orphan nodes.** Every atomic leaf must be reachable from the root
   via BFS traversal of edges. Disconnected nodes indicate missing edges.
4. **Use consistent types.** All edges between the same source/target pair
   should use the same type system. Do not mix `np.ndarray` and
   `ECGPipelineState` for the same data flow.
5. **Type_signature format.** Use named-parameter format:
   `(param: type) -> return_type`, not `Callable[[type], type]`.
6. **All optional boolean fields should be present** on all nodes for
   consistency, even if set to default values.

### 10.7 Handoff validation

The matcher runs two levels of validation:

**Basic** (`validate_handoff`):
- Every atomic leaf has non-empty `description`
- Every atomic leaf has non-empty `type_signature`
- No non-atomic leaves remaining

**Strict** (`validate_handoff_strict`):
- Type signature syntax validated per prover
- Edge type compatibility: `source_type` must match `target_type`
- Graph connectivity: BFS from roots; no orphan nodes
- IOSpec arity: atomic nodes must have both inputs AND outputs

---

## 11. Complete example: `scipy.linalg.lu_factor`

### Step 1: Read the upstream signature

```python
# scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
# Returns: (lu, piv)
#   lu : ndarray, shape (n, n)
#   piv : ndarray, shape (n,)
```

### Step 2: Write the witness

```python
# ageoa/scipy/witnesses.py

from ageoa.ghost.abstract import AbstractArray

def witness_scipy_lu_factor(a: AbstractArray) -> tuple:
    """Witness for lu_factor — validates square input, returns shape metadata."""
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square 2D matrix")
    n = a.shape[0]
    lu = AbstractArray(shape=(n, n), dtype=a.dtype)
    piv = AbstractArray(shape=(n,), dtype="int32")
    return (lu, piv)
```

### Step 3: Write the atom

```python
# ageoa/scipy/linalg.py

import numpy as np
import scipy.linalg
import icontract

from ageoa.ghost.registry import register_atom
from ageoa.scipy.witnesses import witness_scipy_lu_factor


def _is_square_2d(a: np.ndarray) -> bool:
    """Check that a is a 2D square matrix."""
    return a.ndim == 2 and a.shape[0] == a.shape[1]


@register_atom(witness_scipy_lu_factor)
@icontract.require(lambda a: a.ndim == 2, "a must be 2D")
@icontract.require(lambda a: _is_square_2d(a), "a must be square")
@icontract.ensure(
    lambda result, a: result[0].shape == a.shape,
    "Lower-Upper triangular decomposition (LU) factor has same shape as input",
)
@icontract.ensure(
    lambda result, a: result[1].shape == (a.shape[0],),
    "pivot array has length n",
)
def lu_factor(
    a: np.ndarray,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pivoted Lower-Upper triangular decomposition (LU) of a square matrix.

    The decomposition satisfies A = P @ L @ U where P is a permutation
    matrix derived from the pivot indices.

    Args:
        a: Square matrix to decompose, shape (n, n).
        overwrite_a: Whether to overwrite data in a (may improve
            performance).
        check_finite: Whether to check that the input contains only
            finite numbers.

    Returns:
        Tuple of (lu, piv) where lu is the LU factor matrix of shape
        (n, n) and piv is the pivot index array of shape (n,).
    """
    return scipy.linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)
```

### Step 4: Check the atom against the rules

- [x] Concrete types (`np.ndarray`, `bool`, `tuple[...]`), no `Any`
- [x] No `*args` or `**kwargs`
- [x] Return type annotation present
- [x] At least one `@require`
- [x] At least one `@ensure` (two: shape of lu, shape of piv)
- [x] `@register_atom` is outermost decorator
- [x] Witness uses abstract types only, has type annotations
- [x] Helper `_is_square_2d` defined above its first use
- [x] No redundant contracts
- [x] Google-style docstring with Args and Returns
- [x] Abbreviation "LU" spelled out as "Lower-Upper triangular decomposition (LU)"
- [x] Description strings on all decorators
- [x] Expensive postconditions (if any) gated by `_SLOW_CHECKS`

---

## 12. Required tests per atom

Every atom must have a test class that validates five categories. The test
file lives in `tests/` and is named `test_{submodule}.py` (e.g.,
`tests/test_linalg.py`).

**Tests must be organized into classes** named `TestAtomName` (e.g.,
`TestLuFactor`). Do not use standalone functions.

**Tests must use `icontract.ViolationError`** for precondition violation
assertions, never bare `Exception`.

### Category 1: Positive path (correct inputs produce correct output)

Call the atom with valid inputs and assert the output is numerically correct
(or structurally correct for non-numeric functions).

```python
def test_positive_basic(self):
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    lu, piv = ag_linalg.lu_factor(a)
    b = np.array([1.0, 2.0])
    x = scipy.linalg.lu_solve((lu, piv), b)
    assert np.allclose(a @ x, b)
```

### Category 2: Precondition violation (bad inputs raise `ViolationError`)

For every `@require`, provide at least one input that violates it and assert
`icontract.ViolationError` is raised.

```python
def test_require_2d(self):
    with pytest.raises(icontract.ViolationError, match="must be 2D"):
        ag_linalg.lu_factor(np.array([1, 2, 3]))

def test_require_square(self):
    with pytest.raises(icontract.ViolationError, match="must be square"):
        ag_linalg.lu_factor(np.array([[1, 2, 3], [4, 5, 6]]))
```

### Category 3: Postcondition verification (output shapes/properties hold)

Call the atom and explicitly assert the properties stated in `@ensure`.
This serves as a redundant check that the postcondition lambdas are correct.

```python
def test_postcondition_shapes(self):
    a = np.eye(3)
    lu, piv = ag_linalg.lu_factor(a)
    assert lu.shape == a.shape            # mirrors @ensure #1
    assert piv.shape == (a.shape[0],)     # mirrors @ensure #2
```

### Category 4: Edge cases

Test boundary conditions relevant to the function:

```python
def test_1x1(self):
    a = np.array([[5.0]])
    lu, piv = ag_linalg.lu_factor(a)
    assert lu.shape == (1, 1)
    assert piv.shape == (1,)

def test_singular(self):
    """Singular matrix: upstream may raise LinAlgError or return garbage.
    The atom should not crash with a confusing contract error."""
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    lu, piv = ag_linalg.lu_factor(a)
    assert lu.shape == a.shape
```

### Category 5: Upstream parity

Verify the atom produces the same output as calling the upstream function
directly. This catches cases where the wrapper accidentally changes
defaults or drops parameters.

```python
def test_matches_upstream(self):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4))
    lu_atom, piv_atom = ag_linalg.lu_factor(a)
    lu_raw, piv_raw = scipy.linalg.lu_factor(a)
    np.testing.assert_array_equal(lu_atom, lu_raw)
    np.testing.assert_array_equal(piv_atom, piv_raw)
```

---

## 13. Test template

Copy this template for each new atom, replacing `lu_factor` with the
atom name and adjusting inputs/assertions.

```python
"""Tests for ageoa.scipy.linalg.lu_factor atom."""

import numpy as np
import pytest
import icontract
import scipy.linalg

import ageoa.scipy.linalg as ag_linalg


class TestLuFactor:
    """Tests for the lu_factor atom."""

    # -- Category 1: Positive path --

    def test_positive_basic(self):
        a = np.array([[2.0, 1.0], [1.0, 3.0]])
        lu, piv = ag_linalg.lu_factor(a)
        b = np.array([1.0, 2.0])
        x = scipy.linalg.lu_solve((lu, piv), b)
        assert np.allclose(a @ x, b)

    # -- Category 2: Precondition violations --

    def test_require_2d(self):
        with pytest.raises(icontract.ViolationError, match="2D"):
            ag_linalg.lu_factor(np.array([1, 2, 3]))

    def test_require_square(self):
        with pytest.raises(icontract.ViolationError, match="square"):
            ag_linalg.lu_factor(np.array([[1, 2, 3], [4, 5, 6]]))

    # -- Category 3: Postcondition verification --

    def test_postcondition_shapes(self):
        a = np.eye(3)
        lu, piv = ag_linalg.lu_factor(a)
        assert lu.shape == a.shape
        assert piv.shape == (a.shape[0],)

    # -- Category 4: Edge cases --

    def test_1x1(self):
        a = np.array([[5.0]])
        lu, piv = ag_linalg.lu_factor(a)
        assert lu.shape == (1, 1)

    # -- Category 5: Upstream parity --

    def test_matches_upstream(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 4))
        lu_atom, piv_atom = ag_linalg.lu_factor(a)
        lu_raw, piv_raw = scipy.linalg.lu_factor(a)
        np.testing.assert_array_equal(lu_atom, lu_raw)
        np.testing.assert_array_equal(piv_atom, piv_raw)
```

---

## 14. Checklist

Before submitting a new atom, verify every item:

```
Signature:
  [ ] All parameters have type annotations (no bare names)
  [ ] No use of `Any` — use concrete or Union types
  [ ] No `*args` — use explicit parameter with Union type
  [ ] No `**kwargs` — list each supported kwarg explicitly
  [ ] Return type annotation is present
  [ ] Keyword-only arguments (after `*`) are annotated

Contracts:
  [ ] At least one @icontract.require decorator
  [ ] At least one @icontract.ensure decorator
  [ ] Every decorator has a human-readable description string
  [ ] No redundant contracts (each covers a distinct condition)
  [ ] No no-op contracts (lambda ...: True)
  [ ] Lambda parameter names exactly match function parameter names
  [ ] @ensure lambdas use `result` as the first parameter
  [ ] Decorator ordering: isinstance innermost, isfinite outermost
  [ ] Expensive postconditions gated by _SLOW_CHECKS

Ghost Witness:
  [ ] @register_atom(witness_fn) is outermost decorator
  [ ] Witness function uses abstract types only (no np.ndarray)
  [ ] Witness function has type annotations on all params and return
  [ ] Witness is unique to this atom (not reused from another)
  [ ] Witness propagates shape/dtype faithfully

Helpers:
  [ ] All helper functions are defined ABOVE their first use
  [ ] All helper function names start with `_`
  [ ] All helper functions have type-annotated parameters

Docstring:
  [ ] Google-style with Args: and Returns: sections
  [ ] First line is an imperative summary
  [ ] All abbreviations spelled out on first use
  [ ] Domain-specific jargon explained inline
  [ ] Each parameter documents shape and constraints

Module registration:
  [ ] Function is imported in the subpackage __init__.py
  [ ] Function is listed in __all__
  [ ] Import chain reaches ageoa/__init__.py

CDG (if pipeline):
  [ ] No circular edges
  [ ] No duplicate edges
  [ ] No orphan nodes
  [ ] All atomic leaves have non-empty description, type_signature, inputs, outputs
  [ ] All IOSpec constraints are non-empty
  [ ] type_signature uses (param: type) -> type format
  [ ] All optional boolean fields present on all nodes

Tests (all five categories, class-based):
  [ ] Positive path: correct inputs produce correct output
  [ ] Precondition violations: one test per @require, using icontract.ViolationError
  [ ] Postcondition verification: explicitly assert @ensure properties
  [ ] Edge cases: boundary sizes, empty inputs, degenerate cases
  [ ] Upstream parity: output matches direct library call
```

---

## 15. DSP-specific contract patterns

Signal processing atoms have domain-specific contract patterns that go
beyond the standard shape/type postconditions. This section documents the
three main patterns used across the DSP atom families.

### 15.1 Epsilon-Metric Round-Trip

Invertible transform pairs (FFT/IFFT, DCT/IDCT, GFT/IGFT) must satisfy
a round-trip property: applying the forward transform followed by the
inverse must reconstruct the original signal within floating-point
tolerance.

**Pattern:**

```python
import os
_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"

def _roundtrip_close(original: np.ndarray, reconstructed: np.ndarray, atol: float = 1e-10) -> bool:
    return bool(np.allclose(original, reconstructed, atol=atol))

@icontract.ensure(
    lambda result, a, n, axis, norm: _roundtrip_close(
        np.asarray(a),
        np.fft.ifft(result, n=n, axis=axis, norm=norm),
    ),
    "Round-trip IFFT(FFT(x)) must approximate x",
    enabled=_SLOW_CHECKS,
)
def fft(a, n=None, axis=-1, norm=None) -> np.ndarray: ...
```

**Tolerances:**
- `atol=1e-10` for float64 inputs (default)
- `atol=1e-5` for float32 inputs
- Always gated by `_SLOW_CHECKS` (`AGEOA_SLOW_CHECKS=1` env var)

**Applies to:** `fft`/`ifft`, `rfft`/`irfft`, `dct`/`idct`,
`graph_fourier_transform`/`inverse_graph_fourier_transform`

### 15.2 Stability (Filter Design)

Digital Infinite Impulse Response (IIR) filter design atoms must produce
stable filters. A discrete-time filter is stable if and only if all poles
of its transfer function lie strictly inside the unit circle in the z-plane.

**Pattern:**

```python
def _poles_inside_unit_circle(a: np.ndarray) -> bool:
    roots = np.roots(a)
    return bool(np.all(np.abs(roots) < 1.0))

@icontract.ensure(
    lambda result: _poles_inside_unit_circle(result[1]),
    "Designed filter must be stable (poles inside unit circle)",
    enabled=_SLOW_CHECKS,
)
def butter(N, Wn, ...) -> tuple[np.ndarray, np.ndarray]: ...
```

**Notes:**
- Applies to `ba` (transfer function) format output only
- Finite Impulse Response (FIR) filters are trivially stable (denominator
  is `[1]`), so this check is not needed for `firwin`
- Second-Order Sections (SOS) format is stable by construction
- Always gated by `_SLOW_CHECKS`

**Applies to:** `butter`, `cheby1`, `cheby2` (all IIR design atoms)

### 15.3 Total Variation Reduction (GSP)

Graph signal processing smoothing operations (heat diffusion, graph
low-pass filtering) must reduce or preserve the total variation of the
signal. Total variation on a graph is defined as `TV(x) = x^T L x` where
`L` is the graph Laplacian.

**Pattern:**

```python
def _total_variation(L: scipy.sparse.spmatrix, x: np.ndarray) -> float:
    Lx = L.dot(x)
    return float(x.dot(Lx))

@icontract.ensure(
    lambda result, L, x: _total_variation(L, result) <= _total_variation(L, x) + 1e-8,
    "Heat diffusion must reduce total variation (smoothing)",
    enabled=_SLOW_CHECKS,
)
def heat_kernel_diffusion(L, x, t, k=None) -> np.ndarray: ...
```

**Notes:**
- The `+ 1e-8` tolerance accounts for floating-point accumulation
- Applies only to smoothing/low-pass operations, not to general
  graph spectral filters (which may amplify high-frequency components)
- Always gated by `_SLOW_CHECKS`

**Applies to:** `heat_kernel_diffusion`

---

## 16. Bayesian atom patterns

Bayesian and probabilistic inference atoms have specialized witness
requirements based on their `concept_type`. The matcher's ingester
generates these automatically, but hand-authored Bayesian atoms must
follow the same patterns.

### 16.1 Concept types and witness patterns

| Concept Type | Witness Input | Witness Output | Notes |
|---|---|---|---|
| `PRIOR_INIT` | parameters | `AbstractDistribution` | Must set correct `family` and `event_shape` |
| `LOG_PROB` | `AbstractDistribution` + observation | `AbstractScalar` | |
| `SAMPLER` | `AbstractDistribution` + `AbstractRNGState` | sample + advanced `AbstractRNGState` | Must call `rng.advance()` |
| `POSTERIOR_UPDATE` | prior `AbstractDistribution` + data | posterior `AbstractDistribution` | |
| `CONJUGATE_UPDATE` | prior + likelihood `AbstractDistribution` | posterior `AbstractDistribution` | Must call `prior.assert_conjugate_to(likelihood)` |
| `VARIATIONAL_INFERENCE` | model, guide | `AbstractScalar` (ELBO) | ELBO provenance metadata checked |
| `MESSAGE_PASSING` | incoming messages | outgoing message | Must be memoizable; 4 sub-node pattern |

### 16.2 Conjugate pairs

The ghost simulator validates conjugate updates against known pairs:

| Likelihood | Conjugate Prior |
|---|---|
| `normal` | `normal` (known variance) |
| `normal` | `inverse_wishart` (unknown variance) |
| `normal` | `gamma` (precision parameterization) |
| `bernoulli` | `beta` |
| `categorical` | `dirichlet` |
| `poisson` | `gamma` |
| `exponential` | `gamma` |

### 16.3 Message-passing atoms

Factor graph atoms using the `MESSAGE_PASSING` concept type require
four sub-node witness types:
1. **Variable-to-factor message** witness
2. **Factor-to-variable message** witness
3. **Marginal computation** witness
4. **Memo state** witness (for convergence detection)

The ghost simulator runs these iteratively until convergence or a maximum
iteration bound is reached.

### 16.4 Oracle isolation

MCMC/sampling atoms that call external log-density functions are treated
as oracles. Oracle nodes must be stateless and are restricted to allowed
abstract types by the ghost simulator.

---

## 17. Skeleton atoms

Skeleton atoms are placeholders for future ingestion. They have contracts
and witnesses but raise `NotImplementedError` instead of computing a
result.

```python
@register_atom(witness_my_skeleton)
@icontract.require(lambda data: np.isfinite(data).all(), "data must contain only finite values")
@icontract.require(lambda data: data.shape[0] > 0, "data must be non-empty")
@icontract.require(lambda data: data.ndim >= 1, "data must be at least 1D")
@icontract.require(lambda data: isinstance(data, np.ndarray), "data must be ndarray")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim >= 1, "result must be at least 1D")
def my_skeleton(data: np.ndarray) -> np.ndarray:
    """Brief description of what this atom will compute.

    Args:
        data: Input array.

    Returns:
        Processed output array.
    """
    raise NotImplementedError("Skeleton for future ingestion.")
```

Rules for skeletons:
- **Must have meaningful contracts** — not just `data is not None`. Add
  `isinstance`, `ndim`, `shape`, and `isfinite` checks as appropriate.
- **Must have a witness** registered via `@register_atom`.
- **Must have a proper docstring** with Args/Returns sections.
- **Tests must verify** that `NotImplementedError` is raised with valid
  input, and that precondition violations raise `ViolationError`.
- **Decorator ordering applies** — same rules as regular atoms.
