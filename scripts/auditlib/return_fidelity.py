"""AST-based deterministic return-fidelity checks."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .paths import ROOT
from .semantics import utc_now, write_evidence_section


def _read_module(module_path: str) -> str:
    return (ROOT / module_path).read_text()


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


def _name_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    current = []
    for char in text:
        if char.isalnum():
            current.append(char.lower())
        else:
            if current:
                token = "".join(current)
                if len(token) >= 3:
                    tokens.add(token)
                current = []
    if current:
        token = "".join(current)
        if len(token) >= 3:
            tokens.add(token)
    return tokens


def _collect_assigned_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str], dict[str, set[str]]]:
    assigned: set[str] = set()
    call_targets: set[str] = set()
    attr_writes: dict[str, set[str]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            value_is_call = isinstance(node.value, ast.Call)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
                    if value_is_call:
                        call_targets.add(target.id)
                elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    attr_writes.setdefault(target.value.id, set()).add(target.attr)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                assigned.add(target.id)
                if isinstance(node.value, ast.Call):
                    call_targets.add(target.id)
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                attr_writes.setdefault(target.value.id, set()).add(target.attr)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name):
                attr_writes.setdefault(node.target.value.id, set()).add(node.target.attr)
    return assigned, call_targets, attr_writes


def _return_names(expression: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def _return_attrs(expression: ast.AST) -> list[tuple[str, str]]:
    attrs: list[tuple[str, str]] = []
    for node in ast.walk(expression):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attrs.append((node.value.id, node.attr))
    return attrs


def analyze_return_fidelity(record: dict[str, Any]) -> dict[str, Any]:
    """Analyze one wrapper function for suspicious return-shape drift."""
    source_refs = [{"path": record["module_path"], "line": record.get("wrapper_line")}]
    try:
        source = _read_module(record["module_path"])
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001 - evidence path wants failure details
        section = {
            "status": "unknown",
            "findings": ["RETURN_UNKNOWN"],
            "notes": [f"Failed to parse wrapper module: {type(exc).__name__}: {exc}"],
            "source_refs": source_refs,
            "returned_names": [],
            "call_assignment_targets": [],
        }
        return section

    function = _find_function(tree, record["wrapper_symbol"], record.get("wrapper_line"))
    if function is None:
        return {
            "status": "unknown",
            "findings": ["RETURN_UNKNOWN"],
            "notes": ["Wrapper function was not found in the parsed module."],
            "source_refs": source_refs,
            "returned_names": [],
            "call_assignment_targets": [],
        }

    params = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    assigned, call_targets, attr_writes = _collect_assigned_names(function)
    findings: list[str] = []
    notes: list[str] = []
    returned_names: set[str] = set()

    doc_tokens = _name_tokens(record.get("docstring_summary"))
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        names = _return_names(node.value)
        attrs = _return_attrs(node.value)
        returned_names.update(names)

        if call_targets and not (names & call_targets):
            findings.append("RETURN_IGNORES_UPSTREAM_VALUE")
            notes.append("Return expression does not mention any variable assigned from a call result.")

        for root, attr in attrs:
            if root in call_targets and attr not in attr_writes.get(root, set()):
                findings.append("RETURN_FABRICATED_ATTRIBUTE")
                notes.append(f"Return reads attribute `{root}.{attr}` without a local write anchor.")

        if isinstance(node.value, ast.Dict):
            dict_keys = [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            if dict_keys and not set().union(*(_name_tokens(key) for key in dict_keys)).issubset(doc_tokens):
                findings.append("RETURN_DERIVED_ARTIFACT_UNDOCUMENTED")
                notes.append("Return dict introduces keys that are not reflected in the docstring summary.")

    findings = sorted(set(findings))
    if any(code in findings for code in ("RETURN_FABRICATED_ATTRIBUTE", "RETURN_IGNORES_UPSTREAM_VALUE")):
        status = "fail"
    elif findings:
        status = "partial"
    else:
        status = "pass"

    return {
        "status": status,
        "findings": findings,
        "notes": notes,
        "source_refs": source_refs,
        "returned_names": sorted(returned_names),
        "call_assignment_targets": sorted(call_targets),
    }


def write_return_fidelity(record: dict[str, Any]) -> dict[str, Any]:
    """Persist one return-fidelity evidence section."""
    section = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "atom_id": record["atom_id"],
        **analyze_return_fidelity(record),
    }
    write_evidence_section(record["atom_id"], "return_fidelity", section)
    return section
