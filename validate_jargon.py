#!/usr/bin/env python3
"""Validate docstrings for jargon density.

Produces a deterministic "jargon probability" score (0.0–1.0) for every
docstring in the ageoa/ tree. No ML model required — uses a weighted
combination of four heuristic signals:

  1. Acronym density        – ratio of ALL-CAPS tokens (≥2 chars) to total tokens
  2. Rare-word ratio        – fraction of words absent from a common-English word list
  3. Flesch–Kincaid grade   – standard readability metric (higher = harder)
  4. Unexplained-acronym    – acronyms that never appear expanded nearby

Usage:
    python validate_jargon.py                     # scan ageoa/
    python validate_jargon.py path/to/atoms.py    # scan one file
    python validate_jargon.py --threshold 0.6     # only show score ≥ 0.6
    python validate_jargon.py --json              # machine-readable output
    python validate_jargon.py --ci                # exit 1 if any score ≥ threshold
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Common English words (top ~3000 + programming terms).
# We embed a small set inline so the script has zero dependencies beyond stdlib.
# Source: merged from Ogden's Basic English + tech/programming vocabulary.
# ---------------------------------------------------------------------------

_COMMON_WORDS: set[str] | None = None


def _load_common_words() -> set[str]:
    global _COMMON_WORDS
    if _COMMON_WORDS is not None:
        return _COMMON_WORDS

    # We use a two-tier approach:
    # Tier 1: ~1000 most common English words (hardcoded, small)
    # Tier 2: /usr/share/dict/words if available (macOS/Linux)
    base = {
        "a", "about", "above", "across", "act", "add", "after", "again", "against",
        "all", "almost", "along", "already", "also", "always", "an", "and", "another",
        "any", "apply", "are", "area", "argument", "arguments", "array", "as", "at",
        "back", "base", "based", "batch", "be", "because", "been", "before", "begin",
        "being", "below", "between", "both", "but", "by", "call", "called", "can",
        "case", "change", "check", "class", "close", "code", "column", "come", "common",
        "complete", "compute", "computed", "computes", "condition", "contain", "contains",
        "control", "convert", "copy", "correct", "correspond", "corresponding", "could",
        "count", "create", "created", "creates", "current", "data", "default", "define",
        "defined", "defines", "description", "detect", "detected", "determine", "dict",
        "did", "different", "dimension", "dimensional", "dimensions", "do", "does",
        "done", "down", "during", "each", "effect", "element", "elements", "else",
        "empty", "end", "ensure", "equal", "error", "even", "every", "example",
        "except", "execute", "exist", "expected", "expression", "false", "field",
        "file", "final", "find", "first", "float", "following", "for", "form",
        "format", "found", "from", "full", "function", "general", "generate",
        "generated", "generates", "get", "give", "given", "global", "go", "good",
        "got", "group", "had", "handle", "has", "have", "he", "help", "her", "here",
        "high", "his", "hold", "how", "however", "if", "implement", "implementation",
        "implements", "import", "in", "include", "includes", "including", "index",
        "indices", "information", "initial", "initialize", "initialized", "initializes",
        "input", "inputs", "instance", "instead", "int", "integer", "internal", "into",
        "is", "it", "item", "items", "its", "just", "keep", "key", "kind", "know",
        "large", "last", "later", "left", "length", "less", "let", "level", "like",
        "line", "list", "load", "local", "long", "look", "low", "made", "main",
        "make", "many", "map", "mapping", "match", "maximum", "may", "mean", "means",
        "method", "might", "minimum", "model", "module", "more", "most", "much",
        "must", "name", "named", "need", "new", "next", "no", "node", "none", "normal",
        "not", "note", "nothing", "now", "null", "number", "object", "of", "off",
        "old", "on", "once", "one", "only", "open", "operation", "option", "optional",
        "or", "order", "original", "other", "otherwise", "our", "out", "output",
        "over", "own", "pair", "parameter", "parameters", "part", "particular", "pass",
        "passed", "path", "pattern", "per", "perform", "performs", "place", "point",
        "position", "possible", "present", "previous", "print", "problem", "process",
        "processed", "produce", "produced", "provides", "public", "put", "query",
        "raise", "range", "rate", "rather", "read", "real", "record", "reference",
        "remaining", "remove", "repeat", "replace", "represent", "representing",
        "represents", "request", "required", "result", "results", "return", "returned",
        "returns", "right", "row", "rows", "rule", "run", "running", "same", "sample",
        "save", "say", "second", "see", "sequence", "series", "set", "setting",
        "several", "shape", "should", "show", "side", "signal", "similar", "simple",
        "since", "single", "size", "small", "so", "some", "sort", "sorted", "source",
        "space", "specific", "specified", "standard", "start", "state", "static",
        "step", "steps", "still", "stop", "store", "stored", "string", "structure",
        "such", "support", "system", "table", "take", "target", "test", "text", "than",
        "that", "the", "their", "them", "then", "there", "these", "they", "thing",
        "this", "those", "though", "through", "time", "to", "together", "too", "top",
        "total", "transform", "true", "try", "tuple", "turn", "two", "type", "under",
        "unique", "unit", "until", "up", "update", "updated", "updates", "upon", "us",
        "use", "used", "user", "uses", "using", "valid", "value", "values", "variable",
        "version", "very", "view", "want", "was", "way", "we", "well", "were", "what",
        "when", "where", "whether", "which", "while", "who", "whole", "why", "will",
        "window", "with", "within", "without", "word", "words", "work", "working",
        "would", "write", "written", "year", "yet", "you", "your", "zero",
        # Programming / math fundamentals
        "abs", "algorithm", "append", "binary", "bit", "bits", "boolean", "bound",
        "buffer", "byte", "bytes", "cache", "callback", "char", "coefficient",
        "collection", "concatenate", "config", "configuration", "constant", "constructor",
        "coordinate", "coordinates", "cumulative", "decode", "delete", "delta",
        "denominator", "depth", "derivative", "diagonal", "dictionary", "digit",
        "directory", "disable", "division", "double", "dtype", "duplicate", "edge",
        "enable", "encode", "entry", "enum", "epsilon", "equation", "evaluate",
        "evaluation", "event", "exception", "factor", "failure", "feature", "filter",
        "flag", "flatten", "frequency", "graph", "hash", "header", "heap", "height",
        "histogram", "hook", "identifier", "ignore", "image", "immutable", "increment",
        "inherit", "init", "inner", "insert", "integral", "interpolate", "interval",
        "inverse", "iterate", "iteration", "iterator", "json", "label", "lambda",
        "layer", "layout", "leaf", "limit", "linear", "log", "logarithm", "loop",
        "magnitude", "mask", "matrix", "max", "merge", "message", "metadata", "metric",
        "min", "modulo", "multiply", "mutable", "mutex", "namespace", "negative",
        "nested", "network", "normalize", "normalized", "numerator", "numeric",
        "numerical", "offset", "operand", "operator", "optimize", "optimizer", "origin",
        "overflow", "override", "packet", "padding", "parent", "parse", "parser",
        "partial", "payload", "peek", "percent", "percentage", "permutation", "pipe",
        "pixel", "placeholder", "pointer", "polygon", "pool", "pop", "port", "positive",
        "prefix", "prime", "priority", "probability", "product", "profile", "progress",
        "prompt", "property", "protocol", "push", "queue", "quote", "radius", "random",
        "ratio", "raw", "recursive", "reduce", "regex", "register", "registry",
        "relative", "remainder", "render", "replace", "repository", "reset", "resize",
        "resolve", "response", "restore", "retry", "reverse", "root", "rotate",
        "rotation", "round", "runtime", "scalar", "scale", "scan", "schedule",
        "schema", "scope", "score", "scroll", "search", "seed", "segment", "select",
        "separator", "serial", "serialize", "server", "session", "shape", "shift",
        "shuffle", "sign", "signed", "skeleton", "slice", "snapshot", "socket",
        "solve", "span", "sparse", "spawn", "split", "sqrt", "square", "stack",
        "static", "status", "stride", "strip", "submit", "subset", "subtract", "suffix",
        "sum", "swap", "switch", "symbol", "sync", "syntax", "tag", "tail", "task",
        "template", "tensor", "terminal", "threshold", "timeout", "timestamp", "token",
        "trace", "track", "transpose", "traverse", "tree", "trigger", "trim", "truncate",
        "tuple", "type", "underflow", "undo", "unicode", "union", "unsigned", "unset",
        "validate", "validation", "variance", "vector", "verbose", "verify", "vertex",
        "virtual", "volume", "wait", "walk", "weight", "weights", "width", "wildcard",
        "wrap", "wrapper", "xml", "yield",
        # Common numpy / array terms
        "ndarray", "ndim", "numpy", "axis", "axes", "broadcast", "chunk", "column",
        "contiguous", "copy", "dimension", "dot", "elementwise", "flat", "index",
        "indexing", "mean", "median", "meshgrid", "ones", "outer", "pad", "reshape",
        "shape", "size", "slice", "sort", "squeeze", "stack", "std", "stride", "sum",
        "tile", "transpose", "view", "zeros",
    }

    # Try to augment with system dictionary
    dict_path = Path("/usr/share/dict/words")
    if dict_path.exists():
        try:
            text = dict_path.read_text()
            for w in text.splitlines():
                w = w.strip().lower()
                if len(w) >= 2 and w.isalpha():
                    base.add(w)
        except OSError:
            pass

    _COMMON_WORDS = base
    return _COMMON_WORDS


# ---------------------------------------------------------------------------
# Syllable counter (heuristic, for Flesch–Kincaid)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    # Remove trailing silent-e
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")  # 2+ uppercase chars
_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def acronym_density(text: str) -> float:
    """Fraction of tokens that look like unexpanded acronyms."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    acronyms = _ACRONYM_RE.findall(text)
    return len(acronyms) / len(words)


def rare_word_ratio(text: str) -> float:
    """Fraction of words not in the common-English + programming vocabulary."""
    common = _load_common_words()
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    rare = sum(1 for w in words if w not in common)
    return rare / len(words)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch–Kincaid grade level. Higher = harder to read."""
    sentences = re.split(r"[.!?:]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    total_syllables = sum(_count_syllables(w) for w in words)
    avg_words_per_sent = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    grade = 0.39 * avg_words_per_sent + 11.8 * avg_syllables_per_word - 15.59
    return max(grade, 0.0)


def unexplained_acronym_ratio(text: str) -> float:
    """Fraction of acronyms that are never expanded (no parenthetical or dash explanation)."""
    acronyms = set(_ACRONYM_RE.findall(text))
    if not acronyms:
        return 0.0
    unexplained = 0
    for acr in acronyms:
        # Check for patterns like "Electrocardiogram (ECG)" or "ECG — electrocardiogram"
        pattern = re.compile(
            rf"\([^)]*{re.escape(acr)}[^)]*\)"   # inside parens
            rf"|{re.escape(acr)}\s*[-—–]\s*\w",   # followed by dash + word
            re.IGNORECASE,
        )
        if not pattern.search(text):
            unexplained += 1
    return unexplained / len(acronyms)


# ---------------------------------------------------------------------------
# Combined jargon score
# ---------------------------------------------------------------------------

# Weights (sum to 1.0). Tuned so a perfectly clear docstring scores ~0.1
# and a jargon-heavy one scores ~0.8+.
_WEIGHTS = {
    "acronym_density": 0.20,
    "rare_word_ratio": 0.35,
    "flesch_kincaid_normalized": 0.25,
    "unexplained_acronym_ratio": 0.20,
}


def jargon_score(text: str) -> dict:
    """Compute jargon probability for a block of text.

    Returns a dict with component scores and a combined 'score' in [0, 1].
    The score is deterministic — same input always produces same output.
    """
    ad = acronym_density(text)
    rwr = rare_word_ratio(text)
    fk = flesch_kincaid_grade(text)
    uar = unexplained_acronym_ratio(text)

    # Normalize FK grade to 0–1 range. Grade 0 → 0.0, grade 20+ → 1.0
    fk_norm = min(fk / 20.0, 1.0)

    combined = (
        _WEIGHTS["acronym_density"] * min(ad * 5, 1.0)  # scale: 20% acronyms → 1.0
        + _WEIGHTS["rare_word_ratio"] * min(rwr * 2, 1.0)  # scale: 50% rare → 1.0
        + _WEIGHTS["flesch_kincaid_normalized"] * fk_norm
        + _WEIGHTS["unexplained_acronym_ratio"] * uar
    )
    # Clamp to [0, 1]
    combined = max(0.0, min(1.0, combined))

    return {
        "score": round(combined, 3),
        "acronym_density": round(ad, 3),
        "rare_word_ratio": round(rwr, 3),
        "flesch_kincaid_grade": round(fk, 1),
        "unexplained_acronym_ratio": round(uar, 3),
    }


# ---------------------------------------------------------------------------
# AST-based docstring extraction
# ---------------------------------------------------------------------------

@dataclass
class DocstringInfo:
    file: str
    func_name: str
    line: int
    docstring: str
    scores: dict = field(default_factory=dict)


def extract_docstrings(filepath: Path) -> list[DocstringInfo]:
    """Extract all function/class/module docstrings from a Python file."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results: list[DocstringInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = ast.get_docstring(node)
            if ds and ds.strip() not in (
                "Auto-generated verified atom wrapper.",
                "Auto-generated atom wrappers following the ageoa pattern.",
            ):
                results.append(DocstringInfo(
                    file=str(filepath),
                    func_name=node.name,
                    line=node.lineno,
                    docstring=ds,
                ))

    return results


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_python_files(root: Path) -> list[Path]:
    """Find all atoms.py and witnesses.py files under root."""
    patterns = ["**/atoms.py", "**/witnesses.py"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score docstrings for jargon density (0 = clear, 1 = opaque).",
    )
    parser.add_argument(
        "paths", nargs="*", default=["ageoa"],
        help="Files or directories to scan (default: ageoa/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Minimum score to display (default: 0.4)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all docstrings regardless of score",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: exit 1 if any docstring exceeds threshold",
    )
    args = parser.parse_args()

    # Collect files
    all_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            all_files.append(path)
        elif path.is_dir():
            all_files.extend(find_python_files(path))
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)

    if not all_files:
        print("No files found.", file=sys.stderr)
        return 1

    # Score all docstrings
    all_docs: list[DocstringInfo] = []
    for f in all_files:
        docs = extract_docstrings(f)
        for d in docs:
            d.scores = jargon_score(d.docstring)
        all_docs.extend(docs)

    if not all_docs:
        print("No docstrings found.", file=sys.stderr)
        return 0

    # Filter
    threshold = 0.0 if args.all else args.threshold
    flagged = [d for d in all_docs if d.scores["score"] >= threshold]
    flagged.sort(key=lambda d: d.scores["score"], reverse=True)

    # Output
    if args.json_output:
        out = []
        for d in flagged:
            out.append({
                "file": d.file,
                "function": d.func_name,
                "line": d.line,
                **d.scores,
                "docstring_preview": d.docstring[:120],
            })
        print(json.dumps(out, indent=2))
    else:
        # Summary header
        avg_score = sum(d.scores["score"] for d in all_docs) / len(all_docs)
        n_high = sum(1 for d in all_docs if d.scores["score"] >= 0.6)
        print(f"Scanned {len(all_docs)} docstrings across {len(all_files)} files")
        print(f"Average jargon score: {avg_score:.3f}")
        print(f"High-jargon (≥0.6):   {n_high}/{len(all_docs)}")
        print()

        if flagged:
            print(f"{'Score':>5}  {'File':40}  {'Function':35}  Preview")
            print("-" * 120)
            for d in flagged:
                preview = d.docstring.replace("\n", " ")[:60]
                relfile = d.file
                print(f"{d.scores['score']:5.3f}  {relfile:40}  {d.func_name:35}  {preview}")
        else:
            print("All docstrings below threshold — looking good!")

    # CI exit code
    if args.ci:
        violations = [d for d in all_docs if d.scores["score"] >= args.threshold]
        if violations:
            print(
                f"\nCI FAIL: {len(violations)} docstring(s) exceed "
                f"jargon threshold {args.threshold}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
