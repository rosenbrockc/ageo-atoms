# Ingestion Agent Prompt

Primary task: run `ageom ingest` to ingest source code into the atom framework.
Secondary task: verify and validate the generated atoms, witnesses, CDG, and tests.

The ingester supports **Python, Rust, C++, and Julia** source files. The same `ageom ingest` command is used for all languages — the ingester auto-detects the language from the file extension and selects the appropriate parser. All languages produce the same output: Python atom wrappers (with FFI bindings for non-Python sources), ghost witnesses, and a CDG.

---

## Task 1: Run the Ingester

### Command

```bash
ageom ingest <source_file> --class <ClassName> --output <output_dir> [options]
```

**Always use `ageom ingest`** regardless of source language. The ingester detects the language from the file extension and applies the correct parser.

### Supported languages and file extensions

| Language | Extensions | Parser | FFI binding |
|---|---|---|---|
| Python | `.py` | Python AST (+ JAXpr for JAX code) | None (native) |
| Rust | `.rs` | Tree-sitter | `ctypes` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` | Tree-sitter | `ctypes` |
| Julia | `.jl` | Tree-sitter | `juliacall` |

The language is determined automatically from the file extension. You do not need to specify the language explicitly.

### Required arguments

| Arg | Description |
|---|---|
| `<source_file>` | Path to the source file (`.py`, `.rs`, `.jl`, `.cpp`, etc.) |
| `--class <ClassName>` | Name of the class, struct, module, or top-level entry to ingest |
| `--output <dir>` | Output directory for generated files (default: `output/<ClassName>`) |

### Optional arguments

| Flag | Description |
|---|---|
| `--procedural` | Use deterministic procedural extraction (SSA-based) instead of LLM chunking. Works for all languages. |
| `--llm-provider <p>` | Override LLM provider (`anthropic`, `llama_cpp`, `claude_cli`, `codex_cli`, `gemini_cli`) |
| `--llm-model <m>` | Override LLM model |
| `--trace` | Write pipeline event trace to `{output_dir}/trace.jsonl` |

### Examples by language

```bash
# ── Python ──
# Ingest a Python class (LLM-based chunking)
ageom ingest ~/ageo-atoms/ageoa/biosppy/eda.py \
  --class EDAProcessor \
  --output ~/ageo-atoms/ageoa/biosppy

# Python procedural mode (no LLM, deterministic SSA extraction)
ageom ingest path/to/pipeline.py --class MyPipeline --procedural --output output/MyPipeline

# ── Rust ──
# Ingest a Rust struct (tree-sitter parsing, ctypes FFI bindings generated)
ageom ingest ~/ageo-atoms/ageoa/rust_robotics/src/lib.rs \
  --class RobotController \
  --output ~/ageo-atoms/ageoa/rust_robotics

# Rust procedural mode
ageom ingest path/to/lib.rs --class my_module --procedural --output output/my_module

# ── Julia ──
# Ingest a Julia module (tree-sitter parsing, juliacall FFI bindings generated)
ageom ingest ~/ageo-atoms/ageoa/tempo_jl/src/Tempo.jl \
  --class TempoModule \
  --output ~/ageo-atoms/ageoa/tempo_jl

# Julia procedural mode (extracts free/module-level functions)
ageom ingest path/to/module.jl --class MyModule --procedural --output output/MyModule

# ── C++ ──
# Ingest a C++ class (tree-sitter parsing, ctypes FFI bindings generated)
ageom ingest ~/ageo-atoms/ageoa/molecular_docking/src/docking.cpp \
  --class DockingEngine \
  --output ~/ageo-atoms/ageoa/molecular_docking

# C++ procedural mode
ageom ingest path/to/solver.cpp --class Solver --procedural --output output/Solver
```

### Recursive decomposition (max-depth)

The ingester supports recursive decomposition of complex atoms into sub-atoms. This is controlled by two config settings:

| Config | Default | Purpose |
|---|---|---|
| `AGEOM_INGESTER_MAX_DEPTH` | `1` | Max CDG depth. `1` = flat (no recursion). `2`+ enables recursive decomposition. |
| `AGEOM_INGESTER_DECOMPOSE_LINE_THRESHOLD` | `30` | Method line count that triggers sub-decomposition. |

**Recursive decomposition works for all languages.** After the language-specific extractor produces a `RawDataFlowGraph`, the decomposition pipeline operates on that language-agnostic representation. An atom is considered complex (and recursed into) when any of:
- Combined method source exceeds the line threshold
- Methods call 3+ internal sub-functions
- Any method body is a `NotImplementedError` skeleton stub

To enable recursive decomposition, set `AGEOM_INGESTER_MAX_DEPTH` in `.env` or export it:

```bash
# Enable 3-level deep recursive decomposition
export AGEOM_INGESTER_MAX_DEPTH=3

# Then run ingestion as normal (any language)
ageom ingest path/to/source.rs --class MyStruct --output output/MyStruct
```

When `max_depth > 1`, the CDG will contain decomposed parent nodes with `status: "decomposed"` and `children` referencing their sub-atoms. Leaf nodes remain `status: "atomic"`. The `depth` field on each node tracks its level in the decomposition tree (0 = root).

### How non-Python ingestion works

For Rust, C++, and Julia sources, the ingester pipeline is:

1. **Extraction** — Tree-sitter parses the source into a language-agnostic `RawDataFlowGraph`. Language-specific features are detected:
   - **Rust**: trait bounds, lifetime annotations, oracle function detection
   - **Julia**: typed dispatch (multiple dispatch signatures), Bijectors.jl constraints
   - **C++**: function pointer patterns, template specializations
2. **Chunking** — LLM-based (default) or procedural (SSA-based with `--procedural`). Identical for all languages since it operates on the `RawDataFlowGraph`.
3. **Recursive decomposition** — If `max_depth > 1`, complex atoms are recursively split. Language-agnostic.
4. **Code generation** — Python atom wrappers are generated with:
   - `@register_atom` and `@icontract` decorators (same as Python atoms)
   - FFI import block for the source language (`ctypes` for Rust/C++, `juliacall` for Julia)
   - Skeleton bodies that call through to the foreign implementation
5. **Verification** — mypy type-checking and ghost simulation run on the generated Python wrappers.

### Output files

The ingester generates these files in `<output_dir>/`:

| File | Contents |
|---|---|
| `atoms.py` | Python atom wrappers with `@register_atom`, `@icontract` decorators. For non-Python sources, includes FFI bindings. |
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

After ingestion, validate that every generated artifact meets the requirements below. Fix any violations before considering the ingestion complete. These requirements apply equally to atoms generated from any source language — the output is always Python.

### 2.1 Atom requirements (`atoms.py`)

#### Signatures
- All parameters and return type annotated. No `Any`, no `*args`, no `**kwargs`.
- Return type annotation present on every atom.
- Use concrete types (`np.ndarray`, `bool`, `tuple[...]`), not `Any`.

#### FFI atoms (Rust, C++, Julia)
- FFI import block must be present at the top of the file (`ctypes` for Rust/C++, `juliacall` for Julia).
- Atom bodies call through to the foreign implementation via the FFI layer.
- Skeleton atoms raise `NotImplementedError("Skeleton for future ingestion.")` when the FFI binding is not yet available.
- All the same contract, witness, and docstring rules apply as for native Python atoms.

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

#### Recursive decomposition validation
When `max_depth > 1`, verify:
- Decomposed parent nodes have `status: "decomposed"` and non-empty `children` lists.
- Leaf nodes have `status: "atomic"` and empty `children` lists.
- `depth` field is consistent: root = 0, children = parent depth + 1.
- No leaf exceeds `max_depth - 1`.
- Parent-child edges exist for every decomposed node.

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
# Type-check the generated atoms (all languages produce Python wrappers)
mypy <output_dir>/atoms.py

# Run ghost simulation (via the ingester's built-in check)
# The ingester reports "Ghost sim passed: True/False" automatically.

# After placing files in the ageoa package, verify imports work
python -c "import ageoa.<domain>"

# For non-Python atoms, verify FFI bindings load
python -c "from ageoa.<domain>.atoms import <atom_name>"

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

Source language:
  [ ] Correct parser was selected (check trace or output for language detection)
  [ ] For non-Python: FFI import block present in atoms.py
  [ ] For non-Python: atom bodies call through FFI or raise NotImplementedError

Recursive decomposition (if max_depth > 1):
  [ ] Decomposed nodes have status: "decomposed" and non-empty children
  [ ] Leaf nodes have status: "atomic" and empty children
  [ ] depth field is consistent (root=0, children=parent+1)
  [ ] No leaf exceeds max_depth - 1

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
