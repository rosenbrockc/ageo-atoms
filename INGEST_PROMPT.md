# Ingestion Agent Prompt

Primary task: run `ageom ingest` to ingest a Python class into the atom framework.
Secondary task: verify and validate the generated atoms, witnesses, CDG, and tests.

---

## Task 1: Run the Ingester

### Command

```bash
ageom ingest <source_file> --class <ClassName> --output <output_dir> [options]
```

### Required arguments

| Arg | Description |
|---|---|
| `<source_file>` | Path to the Python source file containing the class |
| `--class <ClassName>` | Name of the class to ingest |
| `--output <dir>` | Output directory for generated files (default: `output/<ClassName>`) |

### Optional arguments

| Flag | Description |
|---|---|
| `--procedural` | Use deterministic procedural extraction instead of LLM chunking |
| `--llm-provider <p>` | Override LLM provider (`anthropic`, `llama_cpp`, `claude_cli`, `codex_cli`, `gemini_cli`) |
| `--llm-model <m>` | Override LLM model |
| `--trace` | Write pipeline event trace to `{output_dir}/trace.jsonl` |

### Example

```bash
# Ingest the EDAProcessor class from biosppy
ageom ingest ~/personal/ageo-atoms/ageoa/biosppy/eda.py \
  --class EDAProcessor \
  --output ~/personal/ageo-atoms/ageoa/biosppy

# Procedural mode (no LLM, deterministic)
ageom ingest path/to/source.py --class MyClass --procedural --output output/MyClass
```

### Output files

The ingester generates these files in `<output_dir>/`:

| File | Contents |
|---|---|
| `atoms.py` | Atom wrapper functions with `@register_atom`, `@icontract` decorators |
| `witnesses.py` | Ghost witness functions using abstract types |
| `state_models.py` | State model classes (if the pipeline is stateful) |
| `cdg.json` | Conceptual Dependency Graph describing the pipeline |
| `matches.json` | Match results against indexed library functions (if FAISS index loaded) |
| `trace.jsonl` | Pipeline event trace (if `--trace` flag used) |

### Ingester summary output

After ingestion completes, the CLI prints:

```
Ingestion complete:
  CDG: N nodes, M edges
  Matches: K
  mypy passed: True/False
  Ghost sim passed: True/False
  Output: <output_dir>/
```

Both `mypy passed` and `Ghost sim passed` should be `True`. If either is `False`, proceed to Task 2 to diagnose and fix.

---

## Task 2: Verify and Validate

After ingestion, validate that every generated artifact meets the requirements below. Fix any violations before considering the ingestion complete.

### 2.1 Atom requirements (`atoms.py`)

#### Signatures
- All parameters and return type annotated. No `Any`, no `*args`, no `**kwargs`.
- Return type annotation present on every atom.
- Use concrete types (`np.ndarray`, `bool`, `tuple[...]`), not `Any`.

#### Decorator ordering (bottom-up execution)
`@icontract.require` decorators closest to `def` run **first**. Order must be:
1. `isinstance` checks (innermost, closest to `def` — runs first, prevents TypeError)
2. `ndim` / `shape` checks
3. `isfinite` / expensive checks (outermost — runs last)
4. `@register_atom(witness_fn)` above all `@icontract` decorators (outermost)

```python
@register_atom(witness_fn)                                    # outermost
@icontract.require(lambda x: np.isfinite(x).all(), "msg")    # runs last
@icontract.require(lambda x: x.shape[0] > 0, "msg")
@icontract.require(lambda x: x.ndim >= 1, "msg")
@icontract.require(lambda x: isinstance(x, np.ndarray), "msg")  # runs first
@icontract.ensure(lambda result, x: result.shape == x.shape, "msg")
def my_atom(x: np.ndarray) -> np.ndarray:
```

#### Contract rules
- At least one `@require` and one `@ensure` per atom.
- Every decorator has a human-readable description string.
- Lambda parameter names must exactly match function parameter names.
- `@ensure` lambdas use `result` as first parameter.
- No no-ops (`lambda x: True`). Every contract must check something meaningful.
- No redundant contracts (each covers a distinct condition).

#### Expensive postconditions
Gate with `enabled=_SLOW_CHECKS`:
```python
_SLOW_CHECKS = os.environ.get("AGEOA_SLOW_CHECKS", "0") == "1"
```
Shape-only postconditions are O(1) — always enabled.

#### Helper functions
- Prefix with `_` (indexer skips `_`-prefixed names).
- Define above first use. Pure functions only. Type-annotated parameters.

#### Docstrings
- Google-style with `Args:` and `Returns:` sections mandatory.
- First line: imperative summary of what the function computes.
- Spell out abbreviations on first use: "Finite Impulse Response (FIR)".
- Explain jargon inline: "Hermitian-symmetric (symmetric under conjugation)".

#### Skeleton atoms
Placeholder atoms that raise `NotImplementedError("Skeleton for future ingestion.")`:
- Must have meaningful contracts (isinstance, ndim, shape, isfinite) — not just `data is not None`.
- Must have a registered witness and proper docstring.
- Same decorator ordering rules as regular atoms.

### 2.2 Witness requirements (`witnesses.py`)

- Accept and return **abstract types only** (never `np.ndarray`).
- Type annotations on all parameters and return type (registry reads `__annotations__`).
- Pure: no side effects, no I/O.
- One witness per atom — never reuse another atom's witness.
- Witness propagates shape/dtype metadata faithfully.
- Raises `ValueError` on structural violations (shape mismatch, domain mismatch).

#### Available abstract types (`ageoa.ghost.abstract`)

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

### 2.3 CDG requirements (`cdg.json`)

CDG JSON structure: `{ "nodes": [...], "edges": [...], "metadata": {...} }`.

#### Atomic leaf requirements
- `description` must be non-empty.
- `type_signature` must be non-empty; format: `(param: type) -> return_type` (not `Callable[...]`).
- `inputs` and `outputs` must be non-empty.
- All IOSpec `constraints` must be non-empty.

#### Node required fields
`node_id`, `parent_id`, `name`, `description`, `concept_type`, `inputs` (list[IOSpec]), `outputs` (list[IOSpec]), `status` ("decomposed" | "atomic"), `children`, `depth`, `type_signature`.

#### Optional boolean fields (must be present on all nodes)
`is_optional`, `is_opaque`, `is_external`, `parallelizable`, `conceptual_summary`.

#### Structural rules
- **No circular edges** — must be a DAG.
- **No duplicate edges** — unique (source_id, target_id, output_name, input_name) tuples.
- **No orphan nodes** — all atomic leaves reachable from root via BFS.
- **Consistent types** — don't mix type systems on the same data flow.

#### Edge fields
`source_id`, `target_id`, `output_name`, `input_name`, `source_type`, `target_type`, `requires_glue`.

### 2.4 Module exports

- Every atom imported in the domain's `__init__.py` and listed in `__all__`.
- `@register_atom` fires at import time — unimported atoms are invisible to the ghost simulator.
- Import chain must reach `ageoa/__init__.py`.

### 2.5 Verification commands

```bash
# Type-check the generated atoms
mypy <output_dir>/atoms.py

# Run ghost simulation (via the ingester's built-in check)
# The ingester reports "Ghost sim passed: True/False" automatically.

# After placing files in the ageoa package, verify imports work
python -c "import ageoa.<domain>"

# Upsert CDG into Neo4j graph store (optional)
ageom upsert-cdg <repo_path> --repo-name <domain>
```

---

## Validation Checklist

Run through this checklist after every ingestion:

```
Ingester output:
  [ ] mypy passed: True
  [ ] Ghost sim passed: True
  [ ] CDG has expected number of nodes and edges

Atoms (atoms.py):
  [ ] All parameters and return types annotated (no Any, no *args, no **kwargs)
  [ ] @register_atom(witness_fn) is outermost decorator on every atom
  [ ] At least one @require and one @ensure per atom
  [ ] Every decorator has a human-readable description string
  [ ] Lambda param names match function param names exactly
  [ ] @ensure lambdas use `result` as first parameter
  [ ] No no-op contracts (lambda x: True)
  [ ] Decorator ordering: isinstance innermost, isfinite outermost
  [ ] Expensive postconditions gated by _SLOW_CHECKS
  [ ] Helpers prefixed with _ and defined above first use
  [ ] Google-style docstrings with Args: and Returns:
  [ ] Abbreviations spelled out on first use

Witnesses (witnesses.py):
  [ ] Accept and return abstract types only (no np.ndarray)
  [ ] Type annotations on all params and return type
  [ ] One unique witness per atom (no reuse)
  [ ] Pure functions (no side effects, no I/O)
  [ ] Shape/dtype metadata propagated faithfully

CDG (cdg.json):
  [ ] All atomic leaves have non-empty description, type_signature, inputs, outputs
  [ ] All IOSpec constraints are non-empty
  [ ] type_signature uses (param: type) -> type format
  [ ] All optional boolean fields present on all nodes
  [ ] No circular edges (DAG)
  [ ] No duplicate edges
  [ ] No orphan nodes (all leaves reachable from root)
  [ ] Consistent types on edges

Module exports:
  [ ] All atoms imported in __init__.py and listed in __all__
  [ ] Import chain reaches ageoa/__init__.py
```

---

## DSP Contract Patterns

| Pattern | When | Gate |
|---|---|---|
| Round-trip: `IFFT(FFT(x)) ≈ x` | Invertible transform pairs | `_SLOW_CHECKS` |
| Stability: poles inside unit circle | IIR filter design (`butter`, `cheby1/2`) | `_SLOW_CHECKS` |
| Total variation reduction: `TV(result) ≤ TV(input)` | Graph smoothing (`heat_kernel_diffusion`) | `_SLOW_CHECKS` |

## Bayesian Atom Patterns

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
