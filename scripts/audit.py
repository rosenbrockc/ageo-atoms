#!/usr/bin/env python3
"""Audit all ingested atom directories against INGEST_PROMPT.md requirements.

Usage:
    python scripts/audit.py [--verbose]
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "ageoa"
GHOST_ABSTRACT = BASE / "ghost" / "abstract.py"
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ABSTRACT_TYPES = {
    "AbstractArray", "AbstractSignal", "AbstractScalar", "AbstractMatrix",
    "AbstractDistribution", "AbstractRNGState", "AbstractMCMCTrace",
    "AbstractFilterCoefficients", "AbstractGraphMeta", "AbstractBeatPool",
}

HEAVY_IMPORTS = {"torch", "jax", "haiku", "networkx", "tensorflow"}

BANNED_ANNOTATIONS = {"Any", "any"}


@dataclass
class Violation:
    rule: str
    message: str


@dataclass
class FileAudit:
    path: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def fail(self, rule: str, msg: str) -> None:
        self.violations.append(Violation(rule, msg))


# ---------------------------------------------------------------------------
# Atoms audit
# ---------------------------------------------------------------------------

def audit_atoms(atoms_path: Path) -> FileAudit:
    audit = FileAudit(str(atoms_path))
    text = atoms_path.read_text()

    # R11: Must parse
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        audit.fail("A-PARSE", f"SyntaxError line {e.lineno}: {e.msg}")
        return audit

    lines = text.splitlines()

    # Collect top-level functions only (atoms, not class methods)
    functions: list[ast.FunctionDef] = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        and not n.name.startswith("witness_") and not n.name == "register_atom"
    ]

    for func in functions:
        fname = func.name

        # --- R1: Type annotations ---
        for arg in func.args.args:
            ann = ast.dump(arg.annotation) if arg.annotation else None
            ann_src = ast.get_source_segment(text, arg.annotation) if arg.annotation else None
            if ann is None:
                audit.fail("A-TYPE", f"{fname}: param '{arg.arg}' has no type annotation")
            elif ann_src and ann_src.strip() in BANNED_ANNOTATIONS:
                audit.fail("A-TYPE", f"{fname}: param '{arg.arg}' annotated as Any")

        # Check return annotation
        if func.returns is None:
            audit.fail("A-TYPE", f"{fname}: missing return type annotation")
        else:
            ret_src = ast.get_source_segment(text, func.returns) or ""
            if ret_src.strip() in BANNED_ANNOTATIONS:
                audit.fail("A-TYPE", f"{fname}: return type is Any")

        # --- R2/R3/R5/R6/R7: Decorators ---
        has_register = False
        has_require = False
        has_ensure = False
        has_kwargs_ensure = False
        register_is_outermost = False

        for i, dec in enumerate(func.decorator_list):
            dec_src = ""
            if hasattr(dec, "lineno"):
                dec_line = dec.lineno - 1
                if 0 <= dec_line < len(lines):
                    dec_src = lines[dec_line].strip()

            # Check register_atom
            if "register_atom" in dec_src and "icontract" not in dec_src:
                has_register = True
                if i == 0:
                    register_is_outermost = True

            if "icontract.require" in dec_src:
                has_require = True

            if "icontract.ensure" in dec_src:
                has_ensure = True
                if "**kwargs" in dec_src:
                    has_kwargs_ensure = True

        if not has_register:
            audit.fail("A-REG", f"{fname}: missing @register_atom decorator")
        elif not register_is_outermost:
            audit.fail("A-REG", f"{fname}: @register_atom is not the outermost decorator")

        if not has_require:
            audit.fail("A-CONTRACT", f"{fname}: missing @icontract.require")
        if not has_ensure:
            audit.fail("A-CONTRACT", f"{fname}: missing @icontract.ensure")
        if has_kwargs_ensure:
            audit.fail("A-KWARGS", f"{fname}: @ensure uses **kwargs instead of named params")

        # --- R10: Docstrings ---
        docstring = ast.get_docstring(func)
        if not docstring:
            audit.fail("A-DOC", f"{fname}: missing docstring")
        elif "Args:" not in docstring and len(func.args.args) > 1:
            audit.fail("A-DOC", f"{fname}: docstring missing 'Args:' section")

    # --- Heavy imports ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in HEAVY_IMPORTS:
                    audit.fail("A-IMPORT", f"heavy import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in HEAVY_IMPORTS:
                audit.fail("A-IMPORT", f"heavy import from: {node.module}")

    return audit


# ---------------------------------------------------------------------------
# Witnesses audit
# ---------------------------------------------------------------------------

def audit_witnesses(witnesses_path: Path) -> FileAudit:
    audit = FileAudit(str(witnesses_path))
    text = witnesses_path.read_text()

    # R7: Must parse
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        audit.fail("W-PARSE", f"SyntaxError line {e.lineno}: {e.msg}")
        return audit

    # Collect witness functions
    functions = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("witness_")
    ]

    if not functions:
        audit.fail("W-MISSING", "no witness functions found")
        return audit

    # --- Heavy imports ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in HEAVY_IMPORTS:
                    audit.fail("W-IMPORT", f"heavy import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in HEAVY_IMPORTS:
                audit.fail("W-IMPORT", f"heavy import from: {node.module}")

    for func in functions:
        fname = func.name

        # --- R2: Type annotations ---
        for arg in func.args.args:
            if arg.annotation is None:
                audit.fail("W-TYPE", f"{fname}: param '{arg.arg}' has no type annotation")

        if func.returns is None:
            audit.fail("W-TYPE", f"{fname}: missing return type annotation")

        # --- R4: Purity (check for side effects) ---
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                call_src = ""
                if isinstance(node.func, ast.Attribute):
                    call_src = node.func.attr
                elif isinstance(node.func, ast.Name):
                    call_src = node.func.id
                if call_src in ("print", "open", "write", "logging"):
                    audit.fail("W-PURE", f"{fname}: impure call to '{call_src}'")

        # --- Check return is not None literal ---
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value is not None:
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    audit.fail("W-NONE", f"{fname}: returns None instead of abstract type")

    return audit


# ---------------------------------------------------------------------------
# CDG audit
# ---------------------------------------------------------------------------

def audit_cdg(cdg_path: Path) -> FileAudit:
    audit = FileAudit(str(cdg_path))

    try:
        cdg = json.loads(cdg_path.read_text())
    except json.JSONDecodeError as e:
        audit.fail("C-PARSE", f"Invalid JSON: {e}")
        return audit

    nodes = cdg.get("nodes", [])
    edges = cdg.get("edges", [])
    node_map = {n["node_id"]: n for n in nodes}

    # --- Atomic leaf requirements ---
    for node in nodes:
        nid = node.get("node_id", "?")
        status = node.get("status", "")

        if status == "atomic":
            if not node.get("description"):
                audit.fail("C-LEAF", f"{nid}: empty description")
            if not node.get("type_signature"):
                audit.fail("C-LEAF", f"{nid}: empty type_signature")
            ts = node.get("type_signature", "")
            if "Callable[" in ts:
                audit.fail("C-LEAF", f"{nid}: type_signature uses Callable[...] format")
            if not node.get("inputs"):
                audit.fail("C-LEAF", f"{nid}: empty inputs")
            if not node.get("outputs"):
                audit.fail("C-LEAF", f"{nid}: empty outputs")

            # Check IOSpec constraints
            for io_key in ("inputs", "outputs"):
                for spec in node.get(io_key, []):
                    c = spec.get("constraints", "")
                    if c == "" or c == [] or c is None:
                        audit.fail("C-CONSTRAINT",
                                   f"{nid}.{io_key}.{spec.get('name','?')}: empty constraints")

        # --- Decomposed node checks ---
        if status == "decomposed":
            if not node.get("children"):
                audit.fail("C-DECOMP", f"{nid}: decomposed but no children")
        if status == "atomic":
            if node.get("children"):
                audit.fail("C-DECOMP", f"{nid}: atomic but has children")

    # --- Depth consistency ---
    for node in nodes:
        nid = node.get("node_id", "?")
        pid = node.get("parent_id")
        if pid and pid in node_map:
            parent = node_map[pid]
            expected_depth = parent.get("depth", 0) + 1
            actual_depth = node.get("depth", -1)
            if actual_depth != expected_depth:
                audit.fail("C-DEPTH",
                           f"{nid}: depth={actual_depth}, expected {expected_depth}")

    # --- Self-loops ---
    for e in edges:
        if e.get("source_id") == e.get("target_id"):
            audit.fail("C-SELFLOOP",
                       f"self-loop: {e['source_id']} -> {e['target_id']}")

    # --- Duplicate edges ---
    edge_keys = set()
    for e in edges:
        key = (e.get("source_id"), e.get("target_id"),
               e.get("output_name"), e.get("input_name"))
        if key in edge_keys:
            audit.fail("C-DUP", f"duplicate edge: {key}")
        edge_keys.add(key)

    # --- Cycle detection (topological sort) ---
    in_degree: dict[str, int] = defaultdict(int)
    adj: dict[str, list[str]] = defaultdict(list)
    all_edge_nodes: set[str] = set()
    for e in edges:
        s, t = e.get("source_id", ""), e.get("target_id", "")
        if s == t:
            continue  # skip self-loops already reported
        adj[s].append(t)
        in_degree[t] += 1
        all_edge_nodes.add(s)
        all_edge_nodes.add(t)

    for n in all_edge_nodes:
        in_degree.setdefault(n, 0)

    queue = deque([n for n in all_edge_nodes if in_degree[n] == 0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for t in adj[node]:
            in_degree[t] -= 1
            if in_degree[t] == 0:
                queue.append(t)

    if all_edge_nodes and visited < len(all_edge_nodes):
        cycle_nodes = {n for n in all_edge_nodes if in_degree[n] > 0}
        audit.fail("C-CYCLE", f"DAG has cycle involving {len(cycle_nodes)} nodes: "
                   + ", ".join(sorted(cycle_nodes)[:5]))

    # --- Orphan check (all atomic leaves reachable from root) ---
    root_ids = {n["node_id"] for n in nodes if n.get("depth", 0) == 0}
    parent_child: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        pid = n.get("parent_id")
        if pid:
            parent_child[pid].append(n["node_id"])

    reachable: set[str] = set()
    bfs_q = deque(root_ids)
    while bfs_q:
        nid = bfs_q.popleft()
        if nid in reachable:
            continue
        reachable.add(nid)
        bfs_q.extend(parent_child.get(nid, []))

    for n in nodes:
        if n.get("status") == "atomic" and n["node_id"] not in reachable:
            audit.fail("C-ORPHAN", f"{n['node_id']}: unreachable from root")

    # --- Optional boolean fields ---
    required_bools = {"is_optional", "is_opaque", "is_external", "parallelizable"}
    for n in nodes:
        nid = n.get("node_id", "?")
        missing = required_bools - set(n.keys())
        if missing:
            audit.fail("C-FIELDS", f"{nid}: missing fields: {sorted(missing)}")

    return audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_atom_dirs() -> list[Path]:
    dirs = []
    for atoms_py in sorted(BASE.rglob("atoms.py")):
        d = atoms_py.parent
        # Skip the ghost module and top-level domain files (only audit sub-packages)
        if "ghost" in d.parts:
            continue
        dirs.append(d)
    return dirs


def print_audit(label: str, audit: FileAudit) -> None:
    status = "\033[32mPASS\033[0m" if audit.passed else "\033[31mFAIL\033[0m"
    print(f"  {status}  {label}")
    if VERBOSE and not audit.passed:
        for v in audit.violations:
            print(f"         [{v.rule}] {v.message}")


def main() -> None:
    dirs = discover_atom_dirs()
    print(f"Auditing {len(dirs)} atom directories in {BASE}\n")

    atom_results: list[FileAudit] = []
    witness_results: list[FileAudit] = []
    cdg_results: list[FileAudit] = []

    rule_counts: dict[str, int] = defaultdict(int)

    for d in dirs:
        name = d.name
        atoms_path = d / "atoms.py"
        witnesses_path = d / "witnesses.py"
        cdg_path = d / "cdg.json"

        a = audit_atoms(atoms_path) if atoms_path.exists() else FileAudit(str(atoms_path), [Violation("A-MISSING", "atoms.py not found")])
        w = audit_witnesses(witnesses_path) if witnesses_path.exists() else FileAudit(str(witnesses_path), [Violation("W-MISSING", "witnesses.py not found")])
        c = audit_cdg(cdg_path) if cdg_path.exists() else FileAudit(str(cdg_path), [Violation("C-MISSING", "cdg.json not found")])

        atom_results.append(a)
        witness_results.append(w)
        cdg_results.append(c)

        for v in a.violations + w.violations + c.violations:
            rule_counts[v.rule] += 1

    # --- Print results by category ---
    print("=" * 60)
    print("ATOMS (atoms.py)")
    print("=" * 60)
    a_pass = 0
    for d, a in zip(dirs, atom_results):
        print_audit(d.relative_to(BASE), a)
        if a.passed:
            a_pass += 1
    print(f"\n  {a_pass}/{len(dirs)} pass\n")

    print("=" * 60)
    print("WITNESSES (witnesses.py)")
    print("=" * 60)
    w_pass = 0
    for d, w in zip(dirs, witness_results):
        print_audit(d.relative_to(BASE), w)
        if w.passed:
            w_pass += 1
    print(f"\n  {w_pass}/{len(dirs)} pass\n")

    print("=" * 60)
    print("CDGs (cdg.json)")
    print("=" * 60)
    c_pass = 0
    for d, c in zip(dirs, cdg_results):
        print_audit(d.relative_to(BASE), c)
        if c.passed:
            c_pass += 1
    print(f"\n  {c_pass}/{len(dirs)} pass\n")

    # --- Rule violation summary ---
    print("=" * 60)
    print("VIOLATION SUMMARY")
    print("=" * 60)
    rule_descriptions = {
        "A-PARSE": "atoms.py does not parse (SyntaxError)",
        "A-TYPE": "Missing/banned type annotation (Any)",
        "A-REG": "@register_atom missing or not outermost",
        "A-CONTRACT": "Missing @require or @ensure",
        "A-KWARGS": "@ensure uses **kwargs",
        "A-DOC": "Missing/incomplete docstring",
        "A-IMPORT": "Heavy library import in atoms",
        "W-PARSE": "witnesses.py does not parse (SyntaxError)",
        "W-MISSING": "No witness functions found",
        "W-TYPE": "Witness param missing type annotation",
        "W-IMPORT": "Heavy library import in witnesses",
        "W-PURE": "Impure function call in witness",
        "W-NONE": "Witness returns None",
        "C-PARSE": "cdg.json invalid JSON",
        "C-LEAF": "Atomic leaf missing required field",
        "C-CONSTRAINT": "IOSpec with empty constraints",
        "C-DECOMP": "Status/children inconsistency",
        "C-DEPTH": "Depth field inconsistency",
        "C-SELFLOOP": "Self-loop edge",
        "C-DUP": "Duplicate edge",
        "C-CYCLE": "Cycle in DAG",
        "C-ORPHAN": "Unreachable atomic leaf",
        "C-FIELDS": "Missing optional boolean fields",
    }

    if rule_counts:
        for rule in sorted(rule_counts, key=lambda r: -rule_counts[r]):
            desc = rule_descriptions.get(rule, rule)
            count = rule_counts[rule]
            print(f"  {count:4d}  {rule:16s}  {desc}")
    else:
        print("  No violations found!")

    total_v = sum(rule_counts.values())
    print(f"\n  Total: {total_v} violations across {len(dirs)} directories")
    print(f"  Atoms: {a_pass}/{len(dirs)} pass | "
          f"Witnesses: {w_pass}/{len(dirs)} pass | "
          f"CDGs: {c_pass}/{len(dirs)} pass")

    # Exit code
    sys.exit(0 if total_v == 0 else 1)


if __name__ == "__main__":
    main()
