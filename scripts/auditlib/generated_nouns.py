"""Generated-noun heuristics for deterministic semantic auditing."""

from __future__ import annotations

import ast
import re
from typing import Any

from .io import read_json
from .paths import AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH, ROOT
from .semantics import load_atom_evidence, utc_now, write_evidence_section


def _split_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {token for token in re.split(r"[^A-Za-z0-9]+", expanded.lower()) if len(token) >= 3}


def load_generated_nouns_allowlist() -> set[str]:
    """Load the noun allowlist; missing files degrade to an empty allowlist."""
    if not AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH.exists():
        return set()
    payload = read_json(AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH)
    terms = payload.get("allowlisted_nouns", [])
    return {term.lower() for term in terms if isinstance(term, str)}


def _find_function(tree: ast.AST, wrapper_symbol: str, wrapper_line: int | None) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wrapper_symbol:
            if wrapper_line is None or node.lineno == wrapper_line:
                return node
    return None


def _candidate_nouns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[dict[str, set[str]], set[str]]:
    candidates = {"output": set(), "state": set()}
    all_tokens: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    tokens = _split_tokens(key.value)
                    candidates["output"].update(tokens)
                    all_tokens.update(tokens)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model_copy":
            for keyword in node.keywords:
                if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            tokens = _split_tokens(key.value)
                            candidates["state"].update(tokens)
                            all_tokens.update(tokens)
    return candidates, all_tokens


def analyze_generated_nouns(record: dict[str, Any]) -> dict[str, Any]:
    """Flag generated output/state nouns with low textual upstream support."""
    source_refs = [
        {"path": record["module_path"], "line": record.get("wrapper_line")},
        {"path": str(AUDIT_GENERATED_NOUNS_ALLOWLIST_PATH)},
    ]
    try:
        source = (ROOT / record["module_path"]).read_text()
        tree = ast.parse(source)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unknown",
            "findings": [],
            "notes": [f"Failed to parse wrapper module: {type(exc).__name__}: {exc}"],
            "source_refs": source_refs,
            "candidate_nouns": [],
            "undocumented_nouns": [],
        }

    function = _find_function(tree, record["wrapper_symbol"], record.get("wrapper_line"))
    if function is None:
        return {
            "status": "unknown",
            "findings": [],
            "notes": ["Wrapper function was not found in the parsed module."],
            "source_refs": source_refs,
            "candidate_nouns": [],
            "undocumented_nouns": [],
        }

    candidates, all_tokens = _candidate_nouns(function)
    if not all_tokens:
        return {
            "status": "not_applicable",
            "findings": [],
            "notes": ["No output dict keys or model_copy(update=...) keys were found."],
            "source_refs": source_refs,
            "candidate_nouns": [],
            "undocumented_nouns": [],
        }

    allowlist = load_generated_nouns_allowlist()
    evidence = load_atom_evidence(record["atom_id"])
    mapping = evidence.get("upstream_mapping", {}) if isinstance(evidence, dict) else {}
    corpus = set()
    corpus |= _split_tokens(record.get("wrapper_symbol"))
    corpus |= _split_tokens(record.get("docstring_summary"))
    for value in record.get("upstream_symbols", {}).values():
        if isinstance(value, str):
            corpus |= _split_tokens(value)
    for value in mapping.values() if isinstance(mapping, dict) else ():
        if isinstance(value, str):
            corpus |= _split_tokens(value)

    findings: list[str] = []
    notes: list[str] = []
    undocumented: set[str] = set()
    allowlisted: set[str] = set()

    for token in sorted(candidates["output"]):
        if token in allowlist:
            allowlisted.add(token)
            continue
        if token not in corpus:
            findings.append("NOUN_UNDOCUMENTED_OUTPUT")
            undocumented.add(token)
    for token in sorted(candidates["state"]):
        if token in allowlist:
            allowlisted.add(token)
            continue
        if token not in corpus:
            findings.append("NOUN_UNDOCUMENTED_STATE")
            undocumented.add(token)

    if allowlisted:
        findings.append("NOUN_ALLOWLISTED_DERIVATION")
        notes.append(f"Allowlisted derived nouns: {sorted(allowlisted)}")

    if all_tokens:
        aligned = len(all_tokens - undocumented)
        if aligned / max(len(all_tokens), 1) < 0.5 and len(all_tokens) >= 2:
            findings.append("NOUN_LOW_UPSTREAM_ALIGNMENT")
            notes.append("Less than half of the generated noun tokens align with upstream/docstring terms.")

    findings = sorted(set(findings))
    if any(code in findings for code in ("NOUN_UNDOCUMENTED_OUTPUT", "NOUN_UNDOCUMENTED_STATE")):
        status = "partial"
    elif findings:
        status = "pass"
    else:
        status = "pass"

    return {
        "status": status,
        "findings": findings,
        "notes": notes,
        "source_refs": source_refs,
        "candidate_nouns": sorted(all_tokens),
        "undocumented_nouns": sorted(undocumented),
    }


def write_generated_nouns(record: dict[str, Any]) -> dict[str, Any]:
    """Persist one generated-nouns evidence section."""
    section = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "atom_id": record["atom_id"],
        **analyze_generated_nouns(record),
    }
    write_evidence_section(record["atom_id"], "generated_nouns", section)
    return section
