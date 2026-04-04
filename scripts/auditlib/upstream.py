"""Upstream mapping and signature extraction helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import inspect
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .io import ensure_dir
from .models import UpstreamMapping
from .paths import ATOM_MANIFEST_PATH, AUDIT_DIR, THIRD_PARTY_DIR

_MPLCONFIGDIR = AUDIT_DIR / "mplconfig"
ensure_dir(_MPLCONFIGDIR)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


@lru_cache(maxsize=1)
def load_atom_manifest() -> dict[str, UpstreamMapping]:
    """Load the repo-local atom manifest."""
    raw = yaml.safe_load(ATOM_MANIFEST_PATH.read_text()) or []
    mapping: dict[str, UpstreamMapping] = {}
    for item in raw:
        atom_key = str(item.get("atom", "")).strip()
        upstream = item.get("upstream") or {}
        mapping[atom_key] = UpstreamMapping(
            repo=upstream.get("repo"),
            module=upstream.get("module"),
            function=upstream.get("function"),
            language=upstream.get("language"),
            notes=upstream.get("notes") or item.get("notes"),
        )
    return mapping


def get_upstream_mapping(atom_key: str) -> UpstreamMapping | None:
    """Look up an atom key in the deterministic atom manifest."""
    return load_atom_manifest().get(atom_key)


@lru_cache(maxsize=None)
def get_repo_revision(repo_name: str | None) -> str | None:
    """Read a vendored repo git revision if present."""
    if not repo_name or repo_name == "~":
        return None
    repo_path = THIRD_PARTY_DIR / repo_name
    if not repo_path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=None)
def get_installed_package_version(module_name: str | None) -> str | None:
    """Resolve an installed package version for import-based upstream mappings."""
    if not module_name:
        return None
    root = module_name.split(".", 1)[0]
    try:
        return importlib.metadata.version(root)
    except Exception:
        return None


def _module_to_candidate_paths(repo_name: str, module_name: str) -> list[Path]:
    parts = module_name.split(".")
    repo_root = THIRD_PARTY_DIR / repo_name
    candidates = [
        repo_root / Path(*parts).with_suffix(".py"),
        repo_root / Path(*parts) / "__init__.py",
    ]
    if len(parts) > 1:
        candidates.append(repo_root / Path(*parts[1:]).with_suffix(".py"))
        candidates.append(repo_root / Path(*parts[1:]) / "__init__.py")
    if repo_name.endswith(".jl") and module_name:
        candidates.append(repo_root / "src" / f"{parts[-1]}.jl")
    candidates.append(repo_root / "src" / Path(*parts).with_suffix(".rs"))
    if len(parts) > 1:
        candidates.append(repo_root / "src" / Path(*parts[1:]).with_suffix(".rs"))
    return candidates


def resolve_vendored_module_path(mapping: UpstreamMapping) -> Path | None:
    """Resolve a vendored source path for a mapped upstream symbol."""
    if not mapping.repo or not mapping.module or not mapping.language:
        return None
    if mapping.language not in {"python", "rust"}:
        return None
    for candidate in _module_to_candidate_paths(mapping.repo, mapping.module):
        if candidate.exists():
            return candidate
    return None


def _annotation_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_signature_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        args.append(
            {
                "name": arg.arg,
                "required": default is None,
                "kind": "positional_or_keyword",
                "annotation": _annotation_text(arg.annotation),
            }
        )
    if node.args.vararg is not None:
        args.append(
            {
                "name": node.args.vararg.arg,
                "required": False,
                "kind": "vararg",
                "annotation": _annotation_text(node.args.vararg.annotation),
            }
        )
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        args.append(
            {
                "name": arg.arg,
                "required": default is None,
                "kind": "keyword_only",
                "annotation": _annotation_text(arg.annotation),
            }
        )
    if node.args.kwarg is not None:
        args.append(
            {
                "name": node.args.kwarg.arg,
                "required": False,
                "kind": "kwargs",
                "annotation": _annotation_text(node.args.kwarg.annotation),
            }
        )
    return {
        "parameter_names": [arg["name"] for arg in args if arg["kind"] not in {"vararg", "kwargs"}],
        "required_parameter_names": [arg["name"] for arg in args if arg["required"]],
        "parameters": args,
        "return_annotation": _annotation_text(node.returns),
    }


def _extract_from_ast(path: Path, qualname: str) -> dict[str, Any] | None:
    text = path.read_text()
    tree = ast.parse(text)
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return _extract_signature_from_function(node)
        return None
    if len(parts) == 2:
        class_name, method_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                        return _extract_signature_from_function(child)
    return None


def _extract_with_inspect(mapping: UpstreamMapping) -> dict[str, Any] | None:
    if not mapping.module or not mapping.function:
        return None
    try:
        module = importlib.import_module(mapping.module)
    except Exception:
        return None
    target: Any = module
    try:
        for part in mapping.function.split("."):
            target = getattr(target, part)
    except Exception:
        return None
    try:
        sig = inspect.signature(target)
    except Exception:
        return None
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        kind_name = param.kind.name.lower()
        if kind_name == "positional_or_keyword":
            kind = "positional_or_keyword"
        elif kind_name == "var_positional":
            kind = "var_positional"
        elif kind_name == "var_keyword":
            kind = "var_keyword"
        elif kind_name == "keyword_only":
            kind = "keyword_only"
        elif kind_name == "positional_only":
            kind = "positional_only"
        else:
            kind = kind_name
        required = param.default is inspect._empty and kind not in {"var_positional", "var_keyword"}
        params.append(
            {
                "name": name,
                "required": required,
                "kind": kind,
                "annotation": None if param.annotation is inspect._empty else repr(param.annotation),
            }
        )
    return {
        "parameter_names": [param["name"] for param in params if param["kind"] not in {"var_positional", "var_keyword"}],
        "required_parameter_names": [param["name"] for param in params if param["required"]],
        "parameters": params,
        "return_annotation": None if sig.return_annotation is inspect._empty else repr(sig.return_annotation),
    }


def _split_rust_top_level(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    angle = paren = bracket = brace = 0
    for char in text:
        if char == "," and angle == 0 and paren == 0 and bracket == 0 and brace == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue
        current.append(char)
        if char == "<":
            angle += 1
        elif char == ">":
            angle = max(0, angle - 1)
        elif char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_rust_signature(signature_text: str) -> dict[str, Any] | None:
    open_paren = signature_text.find("(")
    if open_paren == -1:
        return None
    paren_depth = 0
    close_paren = -1
    for idx in range(open_paren, len(signature_text)):
        char = signature_text[idx]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                close_paren = idx
                break
    if close_paren == -1:
        return None
    params_text = signature_text[open_paren + 1 : close_paren]
    return_text = signature_text[close_paren + 1 :].strip()
    if "{" in return_text:
        return_text = return_text.split("{", 1)[0].strip()
    if return_text.startswith("->"):
        return_annotation = return_text[2:].strip() or None
    else:
        return_annotation = None

    params: list[dict[str, Any]] = []
    for raw_param in _split_rust_top_level(params_text):
        param = raw_param.strip()
        if not param:
            continue
        normalized = param.replace("pub ", "").strip()
        if normalized in {"self", "&self", "&mut self", "mut self"}:
            continue
        if ":" not in normalized:
            continue
        name_text, annotation_text = normalized.split(":", 1)
        name = name_text.strip().removeprefix("mut ").strip()
        if not name:
            continue
        params.append(
            {
                "name": name,
                "required": True,
                "kind": "positional_or_keyword",
                "annotation": annotation_text.strip() or None,
            }
        )

    return {
        "parameter_names": [param["name"] for param in params],
        "required_parameter_names": [param["name"] for param in params if param["required"]],
        "parameters": params,
        "return_annotation": return_annotation,
    }


def _extract_from_rust(path: Path, qualname: str) -> dict[str, Any] | None:
    text = path.read_text()
    parts = qualname.replace("::", ".").split(".")
    target_impl = parts[0] if len(parts) == 2 else None
    target_fn = parts[-1]
    lines = text.splitlines()

    def _strip_leading_rust_generics(text: str) -> str:
        normalized = text.strip()
        if not normalized.startswith("<"):
            return normalized
        depth = 0
        for idx, char in enumerate(normalized):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    return normalized[idx + 1 :].strip()
        return normalized

    def _impl_target_from_header(header: str) -> str | None:
        normalized = " ".join(header.split())
        if not normalized.startswith("impl") or "{" not in normalized:
            return None
        if " for " in normalized:
            after_for = normalized.split(" for ", 1)[1].split("{", 1)[0].strip()
            if not after_for:
                return None
            return after_for.split("<", 1)[0].split()[-1]
        after_impl = _strip_leading_rust_generics(normalized[4:].split("{", 1)[0].strip())
        if not after_impl:
            return None
        return after_impl.split("<", 1)[0].split()[-1]

    def _collect_signature(start_index: int) -> dict[str, Any] | None:
        collected: list[str] = []
        paren_depth = 0
        saw_open = False
        for line in lines[start_index:]:
            collected.append(line.strip())
            paren_depth += line.count("(") - line.count(")")
            saw_open = saw_open or "(" in line
            if saw_open and paren_depth <= 0 and ("{" in line or "where" in line or "->" in line):
                break
        return _parse_rust_signature(" ".join(collected))

    if target_impl is None:
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if stripped.startswith("fn ") or stripped.startswith("pub fn "):
                name = stripped.split("fn ", 1)[1].split("(", 1)[0].split("<", 1)[0].strip()
                if name == target_fn:
                    return _collect_signature(idx)
        return None

    impl_target: str | None = None
    impl_depth = 0
    pending_impl_header: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if impl_target is None:
            if pending_impl_header or stripped.startswith("impl"):
                pending_impl_header.append(stripped)
                if "{" not in stripped:
                    continue
                header_lines = list(pending_impl_header)
                header_target = _impl_target_from_header(" ".join(header_lines))
                pending_impl_header = []
                impl_depth = sum(segment.count("{") - segment.count("}") for segment in header_lines)
                if header_target == target_impl:
                    impl_target = header_target
                else:
                    impl_depth = 0
            continue

        if impl_target is not None:
            if stripped.startswith("fn ") or stripped.startswith("pub fn "):
                name = stripped.split("fn ", 1)[1].split("(", 1)[0].split("<", 1)[0].strip()
                if name == target_fn:
                    return _collect_signature(idx)
            impl_depth += line.count("{") - line.count("}")
            if impl_depth <= 0:
                impl_target = None
                impl_depth = 0
    return None


def resolve_upstream_signature(mapping: UpstreamMapping) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Resolve a deterministic upstream signature."""
    if not mapping.module or not mapping.function or not mapping.language:
        return None, None, None
    vendored_path = resolve_vendored_module_path(mapping)
    if mapping.language == "python" and vendored_path is not None:
        signature = _extract_from_ast(vendored_path, mapping.function)
        if signature is not None:
            return signature, "vendored_ast", str(vendored_path)
    if mapping.language == "rust" and vendored_path is not None:
        signature = _extract_from_rust(vendored_path, mapping.function)
        if signature is not None:
            return signature, "vendored_rust", str(vendored_path)
    if mapping.language == "python":
        inspected = _extract_with_inspect(mapping)
        if inspected is not None:
            return inspected, "inspect", mapping.module
    if mapping.language != "python":
        return None, None, "non_python_upstream"
    return None, None, "signature_unavailable"
