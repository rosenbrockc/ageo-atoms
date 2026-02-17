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
7. [Docstrings](#7-docstrings)
8. [Module exports](#8-module-exports)
9. [Complete example: `scipy.linalg.lu_factor`](#9-complete-example-scipylinalglu_factor)
10. [Required tests per atom](#10-required-tests-per-atom)
11. [Test template](#11-test-template)
12. [Checklist](#12-checklist)

---

## 1. Terminology

| Term | Meaning |
|---|---|
| **Atom** | A thin Python wrapper around a single library function, decorated with icontract `@require` / `@ensure`. |
| **Precondition** | An `@icontract.require` decorator that guards inputs. Analogous to a Lean hypothesis. |
| **Postcondition** | An `@icontract.ensure` decorator that asserts properties of the return value. Analogous to a Lean proof obligation. |
| **Declaration** | The `ageom.types.Declaration` object the matcher produces when it indexes this atom. Contains `name`, `type_signature`, `docstring`, `raw_code`, `source_lib`, and `prover="python"`. |
| **Sorry** | `raise NotImplementedError(...)` in generated Python skeletons &mdash; the Python equivalent of Lean's `sorry`. |

---

## 2. File placement

```
ageoa/
  {library}/           # top-level library name, e.g. numpy, scipy
    __init__.py        # re-exports public atoms
    {submodule}.py     # groups atoms by upstream submodule
```

Mirror the upstream package structure. `numpy.linalg.solve` becomes
`ageoa/numpy/linalg.py : solve`. This ensures the fully-qualified atom name
(`ageoa.numpy.linalg.solve`) is predictable and matches what the indexer
produces.

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

### Precondition patterns

| Scenario | Pattern |
|---|---|
| Non-null guard | `@require(lambda x: x is not None, "x must not be None")` |
| Shape guard | `@require(lambda a: a.ndim == 2, "a must be 2D")` |
| Dimensional match | `@require(lambda a, b: a.shape[1] == b.shape[0], "inner dimensions must match")` |
| Square matrix | `@require(lambda a: a.shape[0] == a.shape[1], "a must be square")` |
| Range guard | `@require(lambda low, high: low <= high, "low must be <= high")` |
| Type guard | `@require(lambda shape: isinstance(shape, (int, tuple)), "shape must be int or tuple")` |
| Complex logic | Extract to a named helper (see [section 6](#6-helper-functions)) |

---

## 5. Postconditions (`@ensure`)

Postconditions assert properties of the return value. They are the Python
analogue of a proof obligation: they state *what* the function guarantees,
not *how* it computes the result.

**Every atom must have at least one `@ensure` decorator.** An atom with
only preconditions proves nothing about its output &mdash; it is equivalent
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

## 7. Docstrings

Every atom must have a Google-style docstring with `Args:` and `Returns:`
sections. The docstring is stored in `Declaration.docstring` and used for
embedding-based semantic search. A missing or vague docstring degrades
retrieval quality.

```python
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the linear system a @ x = b for x.

    Computes the exact solution of the well-determined linear matrix
    equation using LU decomposition with partial pivoting.

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
- Args section: one entry per parameter, including shape and constraints.
- Returns section: describe the output shape and semantics.
- Do not repeat the icontract decorators in prose. The contracts are
  machine-readable; the docstring is for humans and embedding models.

---

## 8. Module exports

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

---

## 9. Complete example: `scipy.linalg.lu_factor`

### Step 1: Read the upstream signature

```python
# scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)
# Returns: (lu, piv)
#   lu : ndarray, shape (n, n)
#   piv : ndarray, shape (n,)
```

### Step 2: Write the atom

```python
# ageoa/scipy/linalg.py

import numpy as np
import scipy.linalg
import icontract


def _is_square_2d(a: np.ndarray) -> bool:
    """Check that a is a 2D square matrix."""
    return a.ndim == 2 and a.shape[0] == a.shape[1]


@icontract.require(lambda a: a.ndim == 2, "a must be 2D")
@icontract.require(lambda a: _is_square_2d(a), "a must be square")
@icontract.ensure(
    lambda result, a: result[0].shape == a.shape,
    "LU factor has same shape as input",
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
    """Compute pivoted LU decomposition of a square matrix.

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

### Step 3: Check the atom against the rules

- [x] Concrete types (`np.ndarray`, `bool`, `tuple[...]`), no `Any`
- [x] No `*args` or `**kwargs`
- [x] Return type annotation present
- [x] At least one `@require`
- [x] At least one `@ensure` (two: shape of lu, shape of piv)
- [x] Helper `_is_square_2d` defined above its first use
- [x] No redundant contracts
- [x] Google-style docstring with Args and Returns
- [x] Description strings on all decorators

---

## 10. Required tests per atom

Every atom must have a test function (or set of test functions) that
validates five categories. The test file lives in `tests/` and is named
`test_{submodule}.py` (e.g., `tests/test_linalg.py`).

### Category 1: Positive path (correct inputs produce correct output)

Call the atom with valid inputs and assert the output is numerically correct
(or structurally correct for non-numeric functions).

```python
def test_lu_factor_positive():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    lu, piv = ageoa.scipy.linalg.lu_factor(a)
    # Reconstruct and verify
    from scipy.linalg import lu_solve
    b = np.array([1.0, 2.0])
    x = lu_solve((lu, piv), b)
    assert np.allclose(a @ x, b)
```

### Category 2: Precondition violation (bad inputs raise `ViolationError`)

For every `@require`, provide at least one input that violates it and assert
`icontract.ViolationError` is raised.

```python
def test_lu_factor_require_2d():
    with pytest.raises(icontract.ViolationError, match="must be 2D"):
        ageoa.scipy.linalg.lu_factor(np.array([1, 2, 3]))

def test_lu_factor_require_square():
    with pytest.raises(icontract.ViolationError, match="must be square"):
        ageoa.scipy.linalg.lu_factor(np.array([[1, 2, 3], [4, 5, 6]]))
```

### Category 3: Postcondition verification (output shapes/properties hold)

Call the atom and explicitly assert the properties stated in `@ensure`.
This serves as a redundant check that the postcondition lambdas are correct.

```python
def test_lu_factor_postcondition_shapes():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    lu, piv = ageoa.scipy.linalg.lu_factor(a)
    assert lu.shape == a.shape            # mirrors @ensure #1
    assert piv.shape == (a.shape[0],)     # mirrors @ensure #2
```

### Category 4: Edge cases

Test boundary conditions relevant to the function:

```python
def test_lu_factor_1x1():
    a = np.array([[5.0]])
    lu, piv = ageoa.scipy.linalg.lu_factor(a)
    assert lu.shape == (1, 1)
    assert piv.shape == (1,)

def test_lu_factor_singular():
    """Singular matrix: upstream may raise LinAlgError or return garbage.
    The atom should not crash with a confusing contract error."""
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    # lu_factor does not raise on singular matrices (unlike solve),
    # so this should succeed without ViolationError.
    lu, piv = ageoa.scipy.linalg.lu_factor(a)
    assert lu.shape == a.shape
```

### Category 5: Upstream parity

Verify the atom produces the same output as calling the upstream function
directly. This catches cases where the wrapper accidentally changes
defaults or drops parameters.

```python
def test_lu_factor_matches_upstream():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4))
    lu_atom, piv_atom = ageoa.scipy.linalg.lu_factor(a)
    lu_raw, piv_raw = scipy.linalg.lu_factor(a)
    np.testing.assert_array_equal(lu_atom, lu_raw)
    np.testing.assert_array_equal(piv_atom, piv_raw)
```

---

## 11. Test template

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

## 12. Checklist

Before submitting a new atom, verify every item:

```
Signature:
  [ ] All parameters have type annotations (no bare names)
  [ ] No use of `Any` &mdash; use concrete or Union types
  [ ] No `*args` &mdash; use explicit parameter with Union type
  [ ] No `**kwargs` &mdash; list each supported kwarg explicitly
  [ ] Return type annotation is present
  [ ] Keyword-only arguments (after `*`) are annotated

Contracts:
  [ ] At least one @icontract.require decorator
  [ ] At least one @icontract.ensure decorator
  [ ] Every decorator has a human-readable description string
  [ ] No redundant contracts (each covers a distinct condition)
  [ ] Lambda parameter names exactly match function parameter names
  [ ] @ensure lambdas use `result` as the first parameter

Helpers:
  [ ] All helper functions are defined ABOVE their first use
  [ ] All helper function names start with `_`
  [ ] All helper functions have type-annotated parameters

Docstring:
  [ ] Google-style with Args: and Returns: sections
  [ ] First line is an imperative summary
  [ ] Each parameter documents shape and constraints

Module registration:
  [ ] Function is imported in the subpackage __init__.py
  [ ] Function is listed in __all__

Tests (all five categories):
  [ ] Positive path: correct inputs produce correct output
  [ ] Precondition violations: one test per @require
  [ ] Postcondition verification: explicitly assert @ensure properties
  [ ] Edge cases: boundary sizes, empty inputs, degenerate cases
  [ ] Upstream parity: output matches direct library call
```
