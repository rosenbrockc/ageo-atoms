"""AST-based deterministic state-fidelity checks."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .paths import ROOT
from .semantics import utc_now, write_evidence_section

QUERY_LIKE_TOKENS = ("predict", "transform", "query", "get", "read")


def _read_module(module_path: str) -> tuple[Path, str]:
    path = ROOT / module_path
    return path, path.read_text()


def _find_function(tree: ast.AST, wrapper_symbol: str, wrapper_line: int | None) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wrapper_symbol
    ]
    if not candidates:
        return None
    if wrapper_line is None:
        return candidates[0]
    for node in candidates:
        if node.lineno == wrapper_line:
            return node
    return candidates[0]


def _state_param_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for arg in function.args.args + function.args.kwonlyargs:
        annotation = ast.unparse(arg.annotation) if arg.annotation is not None else ""
        if "state" in arg.arg.lower() or "State" in annotation:
            names.add(arg.arg)
    return names


def _load_state_fields(wrapper_path: Path) -> set[str]:
    candidates = []
    for pattern in ("state_models.py", "*_state.py", "state.py"):
        candidates.extend(sorted(wrapper_path.parent.glob(pattern)))
    fields: set[str] = set()
    for candidate in candidates:
        try:
            tree = ast.parse(candidate.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                        fields.add(child.target.id)
    return fields


def _extract_model_copy_updates(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str]]:
    update_keys: set[str] = set()
    state_reads: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "model_copy":
                for keyword in node.keywords:
                    if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                        for key in keyword.value.keys:
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                update_keys.add(key.value)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if "state" in node.value.id.lower():
                state_reads.add(node.attr)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if "state" in node.value.id.lower() and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                state_reads.add(node.slice.value)
    return update_keys, state_reads


def analyze_state_fidelity(record: dict[str, Any]) -> dict[str, Any]:
    """Analyze stateful wrappers for suspicious state update patterns."""
    source_refs = [{"path": record["module_path"], "line": record.get("wrapper_line")}]
    if not record.get("stateful"):
        return {
            "status": "not_applicable",
            "findings": [],
            "notes": ["Atom is not marked stateful in the manifest."],
            "source_refs": source_refs,
            "state_fields": [],
            "updated_fields": [],
        }

    try:
        wrapper_path, source = _read_module(record["module_path"])
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unknown",
            "findings": ["STATE_UNKNOWN"],
            "notes": [f"Failed to parse wrapper module: {type(exc).__name__}: {exc}"],
            "source_refs": source_refs,
            "state_fields": [],
            "updated_fields": [],
        }

    function = _find_function(tree, record["wrapper_symbol"], record.get("wrapper_line"))
    if function is None:
        return {
            "status": "unknown",
            "findings": ["STATE_UNKNOWN"],
            "notes": ["Wrapper function was not found in the parsed module."],
            "source_refs": source_refs,
            "state_fields": [],
            "updated_fields": [],
        }

    state_params = _state_param_names(function)
    declared_fields = _load_state_fields(wrapper_path)
    update_keys, state_reads = _extract_model_copy_updates(function)
    findings: list[str] = []
    notes: list[str] = []

    if declared_fields and update_keys - declared_fields:
        findings.append("STATE_FABRICATED_FIELD")
        notes.append(f"Update fields {sorted(update_keys - declared_fields)} are not declared in a companion state model.")

    lower_name = record["wrapper_symbol"].lower()
    if any(token in lower_name for token in QUERY_LIKE_TOKENS) and update_keys:
        findings.append("STATE_QUERY_MUTATION_CONFUSION")
        notes.append("Query-like wrapper mutates state via model_copy(update=...).")

    if any(token in lower_name for token in QUERY_LIKE_TOKENS) and state_params and not state_reads:
        findings.append("STATE_REHYDRATION_MISSING")
        notes.append("Query-like wrapper accepts state but does not appear to read any state fields.")

    if any(code in findings for code in ("STATE_FABRICATED_FIELD", "STATE_QUERY_MUTATION_CONFUSION")):
        status = "fail"
    elif findings:
        status = "partial"
    elif update_keys or state_reads:
        status = "pass"
    else:
        status = "unknown"
        findings.append("STATE_UNKNOWN")
        notes.append("No recognizable state update or state-read pattern was found.")

    return {
        "status": status,
        "findings": sorted(set(findings)),
        "notes": notes,
        "source_refs": source_refs,
        "state_fields": sorted(declared_fields),
        "updated_fields": sorted(update_keys),
        "state_reads": sorted(state_reads),
    }


def write_state_fidelity(record: dict[str, Any]) -> dict[str, Any]:
    """Persist one state-fidelity evidence section."""
    section = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "atom_id": record["atom_id"],
        **analyze_state_fidelity(record),
    }
    write_evidence_section(record["atom_id"], "state_fidelity", section)
    return section
