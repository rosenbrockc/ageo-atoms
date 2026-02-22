# Atom Authoring Rules (Agent Reference)

Compact rule set for the `ageoa` package. See `INGESTION.md` for full rationale and examples.

---

## File layout

```
ageoa/{domain}/
  __init__.py        # re-export all public atoms + __all__
  {submodule}.py     # atoms grouped by upstream module
  witnesses.py       # one witness function per atom
```

Mirror upstream package structure. `numpy.linalg.solve` → `ageoa/numpy/linalg.py : solve`.

---

## Atom structure

```python
@register_atom(witness_fn)                          # outermost
@icontract.require(lambda x: expensive_check(x), "msg")  # runs last
@icontract.require(lambda x: x.shape[0] > 0, "msg")
@icontract.require(lambda x: x.ndim >= 1, "msg")
@icontract.require(lambda x: isinstance(x, np.ndarray), "msg")  # runs first (innermost)
@icontract.ensure(lambda result, x: result.shape == x.shape, "msg")
def my_atom(x: np.ndarray) -> np.ndarray:
    """Imperative summary. Spell out abbreviations: Discrete Fourier Transform (DFT).

    Explain jargon inline: "Hermitian-symmetric (symmetric under conjugation)".

    Args:
        x: Description with shape and constraints.

    Returns:
        Description with shape and semantics.
    """
    return upstream_lib.func(x)
```

---

## Hard rules

### Signatures
- All params and return type annotated. No `Any`, no `*args`, no `**kwargs`.

### Contracts
- At least one `@require` and one `@ensure` per atom.
- Every decorator has a human-readable description string.
- Lambda param names must exactly match function param names.
- `@ensure` lambdas use `result` as first param.
- No no-ops (`lambda x: True`). Every contract must check something.

### Decorator ordering (bottom-up execution)
Innermost `@require` (closest to `def`) runs **first**. Order:
1. `isinstance` (innermost — runs first, prevents TypeError)
2. `ndim` / `shape`
3. `isfinite` / expensive checks (outermost — runs last)
4. `@register_atom(witness_fn)` above all `@icontract` decorators

### Expensive postconditions
Gate with `enabled=_SLOW_CHECKS`:
```python
_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"
```
Shape-only postconditions are O(1) — always enabled.

### Helpers
- Prefix with `_` (indexer skips `_`-prefixed names).
- Define above first use. Pure functions only. Type-annotated params.

### Docstrings
- Google-style: `Args:` and `Returns:` sections mandatory.
- Spell out abbreviations on first use: "Finite Impulse Response (FIR)".
- Explain jargon inline: "antithetic variates (a variance-reduction technique...)".

### Module exports
- Every atom imported in `__init__.py` and listed in `__all__`.
- `@register_atom` fires at import time — unimported atoms are invisible to the ghost simulator.

---

## Ghost Witness

Every atom needs a unique witness registered via `@register_atom(witness_fn)`.

### Witness rules
- Accept/return **abstract types only** (never `np.ndarray`).
- Type annotations on all params and return type (registry reads `__annotations__`).
- Pure: no side effects, no I/O.
- One witness per atom — never reuse another atom's witness.

### Abstract types (`ageoa.ghost.abstract`)

| Type | Fields | Use |
|---|---|---|
| `AbstractArray` | `shape, dtype, is_sorted, min_val, max_val` | Generic arrays |
| `AbstractSignal` | `shape, dtype, sampling_rate, domain, units` | DSP atoms |
| `AbstractScalar` | `dtype, min_val, max_val, is_index` | Scalar returns |
| `AbstractMatrix` | `shape (str,str), dtype` | Symbolic dims |
| `AbstractDistribution` | `family, event_shape, batch_shape, support` | Bayesian |
| `AbstractRNGState` | `seed, consumed, is_split` | Stochastic |
| `AbstractMCMCTrace` | `n_samples, n_chains, param_dims, warmup_steps` | MCMC |
| `AbstractFilterCoefficients` | `order, btype, format, is_stable` | Filter design |
| `AbstractGraphMeta` | `n_nodes, is_symmetric` | Graph signal |
| `AbstractBeatPool` | `size, is_calibrated` | Beat detection |

---

## CDG schema

CDG JSON = `{ "nodes": [...], "edges": [...], "metadata": {...} }`.

### Node required fields
`node_id`, `parent_id`, `name`, `description`, `concept_type`, `inputs` (list[IOSpec]), `outputs` (list[IOSpec]), `status` ("decomposed" | "atomic"), `children`, `depth`, `type_signature`.

### Atomic leaf requirements
- `description`, `type_signature` must be non-empty.
- `inputs` and `outputs` must be non-empty.
- `type_signature` format: `(param: type) -> return_type` (not `Callable[...]`).
- IOSpec `constraints` must be non-empty.

### Edge fields
`source_id`, `target_id`, `output_name`, `input_name`, `source_type`, `target_type`, `requires_glue`.

### Structural rules
- **No circular edges** — must be a DAG.
- **No duplicate edges** — unique (source, target, output, input) tuples.
- **No orphan nodes** — all leaves reachable from root via BFS.
- **Consistent types** — don't mix type systems on the same data flow.
- **All optional booleans present** — `is_optional`, `is_opaque`, `is_external`, `parallelizable`, `conceptual_summary`.

---

## Bayesian atoms

| Concept type | Witness returns | Extra rule |
|---|---|---|
| `PRIOR_INIT` | `AbstractDistribution` | Set `family` + `event_shape` |
| `LOG_PROB` | `AbstractScalar` | |
| `SAMPLER` | sample + `AbstractRNGState` | Call `rng.advance()` |
| `POSTERIOR_UPDATE` | `AbstractDistribution` | |
| `CONJUGATE_UPDATE` | `AbstractDistribution` | Call `prior.assert_conjugate_to(likelihood)` |
| `VARIATIONAL_INFERENCE` | `AbstractScalar` (ELBO) | |
| `MESSAGE_PASSING` | outgoing message | Memoizable; 4 sub-node pattern |

Conjugate pairs: normal-normal, normal-inverse_wishart, normal-gamma, bernoulli-beta, categorical-dirichlet, poisson-gamma, exponential-gamma.

---

## Skeleton atoms

Placeholder atoms that raise `NotImplementedError("Skeleton for future ingestion.")`.

- Must have meaningful contracts (isinstance, ndim, shape, isfinite) — not just `data is not None`.
- Must have a registered witness and proper docstring.
- Same decorator ordering rules as regular atoms.

---

## Tests (class-based, 5 categories)

```python
class TestMyAtom:
    def test_positive_basic(self):        # 1. correct inputs → correct output
    def test_require_ndim(self):          # 2. bad input → icontract.ViolationError
    def test_postcondition_shape(self):   # 3. explicitly assert @ensure properties
    def test_edge_case_1x1(self):         # 4. boundary sizes, degenerate inputs
    def test_matches_upstream(self):      # 5. output == direct library call
```

- Always `pytest.raises(icontract.ViolationError)`, never `pytest.raises(Exception)`.
- Skeleton tests: verify `NotImplementedError` with valid input + `ViolationError` with invalid input.

---

## DSP contract patterns

| Pattern | When | Gate |
|---|---|---|
| Round-trip: `IFFT(FFT(x)) ≈ x` | Invertible transform pairs | `_SLOW_CHECKS` |
| Stability: poles inside unit circle | IIR filter design (`butter`, `cheby1/2`) | `_SLOW_CHECKS` |
| Total variation reduction: `TV(result) ≤ TV(input)` | Graph smoothing (`heat_kernel_diffusion`) | `_SLOW_CHECKS` |
