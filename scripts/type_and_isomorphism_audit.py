#!/usr/bin/env python3
"""Type standardization and subgraph isomorphism audit for ageo-atoms CDGs.

Usage:
    python scripts/type_and_isomorphism_audit.py [--verbose] [--json] [--threshold 0.3] [--top-k 20]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent / "ageoa"

# ── ANSI helpers ──────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Normalization constants ───────────────────────────────────────────────

NORM_MAP: dict[str, str] = {
    # Array variants → canonical
    "ndarray":                          "np.ndarray",
    "Array-like":                       "array-like",
    "array<float>":                     "np.ndarray[float]",
    "array<int>":                       "np.ndarray[int]",
    "array<object>":                    "np.ndarray",
    "Array[float]":                     "np.ndarray[float]",
    "Array[int]":                       "np.ndarray[int]",
    "Array[particle]":                  "np.ndarray",
    "Array[observation_or_prediction]": "np.ndarray",
    "np.ndarray of shape (N, 2)":       "np.ndarray",
    "1D numeric array":                 "np.ndarray",
    "1D boolean array":                 "np.ndarray[bool]",
    "2D numeric array [n,2]":           "np.ndarray",
    "list of 1D numeric arrays":        "list[np.ndarray]",
    "boolean array-like":               "np.ndarray[bool]",
    "FeatureMatrix (np.ndarray[float])": "np.ndarray[float]",
    "IndexVector (np.ndarray[int])":    "np.ndarray[int]",
    # Primitive casing
    "Float":    "float",
    "Integer":  "int",
    "Boolean":  "bool",
    "String":   "str",
    "string":   "str",
    "Number":   "float",
    "Object":   "object",
    "Scalar":   "float",
    # Opaque
    "any":      "Any",
}

KNOWN_PRIMITIVES = {
    "float", "int", "bool", "str", "None", "Any", "object",
    "dict", "list", "tuple", "set", "bytes", "complex",
    "np.ndarray", "array-like", "array",
    "np.ndarray[float]", "np.ndarray[int]", "np.ndarray[bool]",
    "pd.DataFrame", "matplotlib.Axes", "matplotlib.axes.Axes",
    "matplotlib.figure.Figure", "Path",
}

PROSE_THRESHOLD_WORDS = 4

# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class CDGData:
    atom_name: str
    path: Path
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


@dataclass
class TypeOccurrence:
    atom: str
    node_id: str
    io_direction: str
    port_name: str
    raw: str
    normalized: str
    norm_rule: str | None
    classification: str


@dataclass
class CDGTypeAudit:
    atom_name: str
    occurrences: list[TypeOccurrence]
    norm_applied: Counter
    non_normalizable: list[TypeOccurrence]
    health_score: float
    class_dist: Counter


@dataclass
class DecomposedSubgraph:
    atom: str
    node_id: str
    node_name: str
    n_children: int
    topo_hash: str
    child_concept_types: list[str]
    degree_seq: list[tuple[int, int]]


@dataclass
class SubgraphPair:
    a: DecomposedSubgraph
    b: DecomposedSubgraph
    hash_match: bool
    jaccard: float
    a_types: dict[str, int]
    b_types: dict[str, int]


# ── Data loading ──────────────────────────────────────────────────────────

def discover_atom_dirs() -> list[Path]:
    dirs = []
    for cdg_json in sorted(BASE.rglob("cdg.json")):
        d = cdg_json.parent
        if "ghost" in d.parts:
            continue
        dirs.append(d)
    return dirs


def load_all_cdgs() -> list[CDGData]:
    result = []
    for d in discover_atom_dirs():
        try:
            raw = json.loads((d / "cdg.json").read_text())
            result.append(CDGData(
                atom_name=str(d.relative_to(BASE)),
                path=d / "cdg.json",
                nodes=raw.get("nodes", []),
                edges=raw.get("edges", []),
            ))
        except json.JSONDecodeError as exc:
            print(f"  WARNING: {d.name}/cdg.json parse error: {exc}", file=sys.stderr)
    return result


# ── Part 1: Type normalization & classification ───────────────────────────

def normalize_type_desc(td: str) -> tuple[str, str | None]:
    """Normalize a type_desc string. Returns (normalized, rule_name_or_None)."""
    td = td.strip()
    if td in NORM_MAP:
        return NORM_MAP[td], "direct_map"

    # null notation: | null -> | None
    step = re.sub(r"\|\s*null\b", "| None", td)
    if step != td:
        return step, "null_to_None"

    # Optional[X] -> X | None
    m = re.fullmatch(r"Optional\[(.+)\]", td)
    if m:
        return f"{m.group(1)} | None", "optional_expand"

    return td, None


def classify_type_desc(td: str) -> str:
    """Classify a normalized type_desc."""
    if td in KNOWN_PRIMITIVES:
        return "primitive"
    has_structural = bool(re.search(r"[<>\[\]|{}()]", td))
    if has_structural:
        return "structural"
    if re.fullmatch(r"[A-Z][A-Za-z0-9_]+", td):
        return "domain_type"
    words = td.split()
    if len(words) >= PROSE_THRESHOLD_WORDS:
        return "prose"
    if len(words) >= 2:
        return "short_phrase"
    return "other"


def audit_types_per_cdg(cdg: CDGData) -> CDGTypeAudit:
    occurrences: list[TypeOccurrence] = []
    norm_applied: Counter = Counter()

    for node in cdg.nodes:
        for io_key in ("inputs", "outputs"):
            for spec in node.get(io_key, []):
                td = spec.get("type_desc", "").strip()
                if not td:
                    continue
                normalized, rule = normalize_type_desc(td)
                cls = classify_type_desc(normalized)
                if rule:
                    norm_applied[rule] += 1
                occurrences.append(TypeOccurrence(
                    atom=cdg.atom_name,
                    node_id=node.get("node_id", "?"),
                    io_direction=io_key,
                    port_name=spec.get("name", "?"),
                    raw=td,
                    normalized=normalized,
                    norm_rule=rule,
                    classification=cls,
                ))

    non_norm = [o for o in occurrences
                if o.norm_rule is None and o.classification in ("prose", "other", "short_phrase")]
    healthy = sum(1 for o in occurrences
                  if o.classification in ("primitive", "domain_type", "structural"))
    score = healthy / len(occurrences) if occurrences else 1.0
    class_dist = Counter(o.classification for o in occurrences)

    return CDGTypeAudit(
        atom_name=cdg.atom_name,
        occurrences=occurrences,
        norm_applied=norm_applied,
        non_normalizable=non_norm,
        health_score=score,
        class_dist=class_dist,
    )


def build_cross_cdg_vocab(audits: list[CDGTypeAudit]) -> list[tuple[str, str, int, set[str]]]:
    """Returns sorted list of (atom_a, atom_b, shared_count, shared_types)."""
    type_sets: dict[str, set[str]] = {}
    for a in audits:
        types = set()
        for o in a.occurrences:
            if o.classification in ("primitive", "domain_type", "structural"):
                types.add(o.normalized)
        type_sets[a.atom_name] = types

    pairs = []
    names = sorted(type_sets)
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            shared = type_sets[na] & type_sets[nb]
            if shared:
                pairs.append((na, nb, len(shared), shared))
    pairs.sort(key=lambda p: -p[2])
    return pairs


# ── Part 2: Topo-hash & isomorphism ──────────────────────────────────────

def topo_hash(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], root_id: str) -> str:
    """Exact reimplementation of _topo_hash from ageo-matcher/ageom/graph_store.py."""
    children = [n for n in nodes if n.get("parent_id") == root_id]
    child_ids = {c["node_id"] for c in children}
    sibling_edges = [
        e for e in edges
        if e["source_id"] in child_ids and e["target_id"] in child_ids
    ]
    degree_seq: list[tuple[int, int]] = []
    for cid in sorted(child_ids):
        in_deg = sum(1 for e in sibling_edges if e["target_id"] == cid)
        out_deg = sum(1 for e in sibling_edges if e["source_id"] == cid)
        degree_seq.append((in_deg, out_deg))
    raw = str(sorted(degree_seq))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def collect_decomposed_subgraphs(cdgs: list[CDGData]) -> list[DecomposedSubgraph]:
    result = []
    for cdg in cdgs:
        node_map = {n["node_id"]: n for n in cdg.nodes}
        for node in cdg.nodes:
            if node.get("status") != "decomposed":
                continue
            nid = node["node_id"]
            children = [n for n in cdg.nodes if n.get("parent_id") == nid]
            if not children:
                continue
            child_ids = {c["node_id"] for c in children}
            sibling_edges = [
                e for e in cdg.edges
                if e["source_id"] in child_ids and e["target_id"] in child_ids
            ]
            degree_seq = []
            for cid in sorted(child_ids):
                in_d = sum(1 for e in sibling_edges if e["target_id"] == cid)
                out_d = sum(1 for e in sibling_edges if e["source_id"] == cid)
                degree_seq.append((in_d, out_d))
            h = topo_hash(cdg.nodes, cdg.edges, nid)
            child_concept_types = [c.get("concept_type", "custom") for c in children]
            result.append(DecomposedSubgraph(
                atom=cdg.atom_name,
                node_id=nid,
                node_name=node.get("name", nid),
                n_children=len(children),
                topo_hash=h,
                child_concept_types=child_concept_types,
                degree_seq=degree_seq,
            ))
    return result


def jaccard_multiset(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    intersection = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return intersection / union if union > 0 else 0.0


def is_degenerate(sg: DecomposedSubgraph) -> bool:
    return all(d == (0, 0) for d in sg.degree_seq)


def run_pairwise_comparison(
    subgraphs: list[DecomposedSubgraph],
    jaccard_threshold: float = 0.3,
    top_k: int = 20,
) -> tuple[list[SubgraphPair], list[SubgraphPair]]:
    exact_matches: list[SubgraphPair] = []
    similar: list[SubgraphPair] = []

    n = len(subgraphs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = subgraphs[i], subgraphs[j]
            hm = a.topo_hash == b.topo_hash
            jac = jaccard_multiset(a.child_concept_types, b.child_concept_types)
            pair = SubgraphPair(
                a, b, hm, jac,
                dict(Counter(a.child_concept_types)),
                dict(Counter(b.child_concept_types)),
            )
            if hm:
                exact_matches.append(pair)
            if jac >= jaccard_threshold:
                similar.append(pair)

    similar.sort(key=lambda p: -p.jaccard)
    return exact_matches, similar[:top_k]


# ── Reporting ─────────────────────────────────────────────────────────────

def report_type_audit(
    audits: list[CDGTypeAudit],
    vocab_pairs: list[tuple[str, str, int, set[str]]],
    verbose: bool,
) -> dict[str, Any]:
    print(f"\n{BOLD}{'=' * 60}")
    print("TYPE STANDARDIZATION AUDIT")
    print(f"{'=' * 60}{RESET}\n")

    # Aggregate normalization rules
    total_norm: Counter = Counter()
    raw_to_canonical: dict[str, str] = {}
    for a in audits:
        total_norm += a.norm_applied
        for o in a.occurrences:
            if o.norm_rule and o.raw != o.normalized:
                raw_to_canonical[o.raw] = o.normalized

    print(f"{BOLD}Normalization Rules Applied:{RESET}")
    if raw_to_canonical:
        for raw, canon in sorted(raw_to_canonical.items(), key=lambda x: x[0].lower()):
            count = sum(1 for a in audits for o in a.occurrences
                        if o.raw == raw and o.norm_rule)
            atoms = sorted({o.atom for a in audits for o in a.occurrences
                            if o.raw == raw and o.norm_rule})
            print(f"  {DIM}{raw}{RESET} -> {CYAN}{canon}{RESET}  "
                  f"({count} occ in {', '.join(atoms)})")
    else:
        print("  (none)")

    total_rules = sum(total_norm.values())
    print(f"\n  Total transformations: {total_rules}")
    for rule, cnt in total_norm.most_common():
        print(f"    {rule}: {cnt}")

    # Non-normalizable types
    all_non_norm: list[TypeOccurrence] = []
    for a in audits:
        all_non_norm.extend(a.non_normalizable)

    # Deduplicate by (raw, atom)
    seen: set[tuple[str, str]] = set()
    unique_non_norm: list[TypeOccurrence] = []
    for o in all_non_norm:
        key = (o.raw, o.atom)
        if key not in seen:
            seen.add(key)
            unique_non_norm.append(o)

    print(f"\n{BOLD}Non-Normalizable Types ({len(unique_non_norm)} unique):{RESET}")
    if verbose:
        for cls in ("prose", "short_phrase", "other"):
            items = [o for o in unique_non_norm if o.classification == cls]
            if items:
                print(f"  [{cls}] ({len(items)}):")
                for o in sorted(items, key=lambda x: (x.atom, x.raw)):
                    print(f"    {DIM}{o.atom}/{o.node_id}.{o.io_direction}.{o.port_name}:{RESET} "
                          f"{YELLOW}\"{o.raw}\"{RESET}")
    else:
        by_cls = Counter(o.classification for o in unique_non_norm)
        for cls, cnt in by_cls.most_common():
            print(f"  [{cls}]: {cnt} unique types")
        print(f"  {DIM}(use --verbose for full list){RESET}")

    # Per-CDG health table
    print(f"\n{BOLD}Per-CDG Type Health:{RESET}")
    print(f"  {'CDG':<25} {'Score':>6}  {'Total':>5}  "
          f"{'Prim':>5} {'Domain':>6} {'Struct':>6} {'Prose':>5} {'Short':>5} {'Other':>5}")
    print(f"  {'-' * 90}")
    for a in sorted(audits, key=lambda x: x.health_score):
        d = a.class_dist
        color = GREEN if a.health_score >= 0.9 else (YELLOW if a.health_score >= 0.7 else RED)
        print(f"  {a.atom_name:<25} {color}{a.health_score:>5.1%}{RESET}  "
              f"{sum(d.values()):>5}  "
              f"{d.get('primitive', 0):>5} {d.get('domain_type', 0):>6} "
              f"{d.get('structural', 0):>6} {d.get('prose', 0):>5} "
              f"{d.get('short_phrase', 0):>5} {d.get('other', 0):>5}")

    # Cross-CDG vocab
    print(f"\n{BOLD}Cross-CDG Type Compatibility (top-15 by shared types):{RESET}")
    for na, nb, cnt, shared in vocab_pairs[:15]:
        print(f"  {na:<25} <-> {nb:<25}  {cnt:>2} shared")
        if verbose:
            for t in sorted(shared):
                print(f"    {DIM}{t}{RESET}")

    # Overall summary
    total_occ = sum(len(a.occurrences) for a in audits)
    total_healthy = sum(
        sum(1 for o in a.occurrences
            if o.classification in ("primitive", "domain_type", "structural"))
        for a in audits
    )
    all_raw = set()
    all_normed = set()
    for a in audits:
        for o in a.occurrences:
            all_raw.add(o.raw)
            all_normed.add(o.normalized)

    overall = total_healthy / total_occ if total_occ else 1.0
    color = GREEN if overall >= 0.9 else (YELLOW if overall >= 0.7 else RED)
    print(f"\n{BOLD}Overall:{RESET} {color}{overall:.1%}{RESET} healthy "
          f"({total_healthy}/{total_occ} occurrences)")
    print(f"  Unique raw types: {len(all_raw)}  |  "
          f"After normalization: {len(all_normed)}  |  "
          f"Collapsed: {len(all_raw) - len(all_normed)}")

    return {
        "summary": {
            "total_occurrences": total_occ,
            "unique_raw": len(all_raw),
            "unique_normalized": len(all_normed),
            "collapsed": len(all_raw) - len(all_normed),
            "overall_health": round(overall, 4),
        },
        "per_cdg": [
            {"atom": a.atom_name, "health_score": round(a.health_score, 4),
             "total": sum(a.class_dist.values()), "distribution": dict(a.class_dist)}
            for a in sorted(audits, key=lambda x: x.atom_name)
        ],
    }


def report_isomorphism_audit(
    subgraphs: list[DecomposedSubgraph],
    exact_matches: list[SubgraphPair],
    top_similar: list[SubgraphPair],
    verbose: bool,
) -> dict[str, Any]:
    print(f"\n{BOLD}{'=' * 60}")
    print("SUBGRAPH ISOMORPHISM AUDIT")
    print(f"{'=' * 60}{RESET}\n")

    # Hash distribution
    hash_groups: dict[str, list[DecomposedSubgraph]] = defaultdict(list)
    for sg in subgraphs:
        hash_groups[sg.topo_hash].append(sg)

    collision_groups = {h: sgs for h, sgs in hash_groups.items() if len(sgs) >= 2}
    sgs_in_collisions = sum(len(sgs) for sgs in collision_groups.values())

    print(f"{BOLD}Topo-Hash Distribution:{RESET}")
    print(f"  {len(subgraphs)} decomposed subgraphs across {len(set(sg.atom for sg in subgraphs))} CDGs")
    print(f"  {len(hash_groups)} unique hashes  |  "
          f"{len(collision_groups)} collision groups  |  "
          f"{sgs_in_collisions} subgraphs in collisions")

    if collision_groups:
        print(f"\n{BOLD}Hash Collision Groups:{RESET}")
        for h, sgs in sorted(collision_groups.items(), key=lambda x: -len(x[1])):
            degen = all(is_degenerate(sg) for sg in sgs)
            degen_tag = f"  {DIM}[DEGENERATE: no sibling edges]{RESET}" if degen else ""
            print(f"\n  Hash {CYAN}{h}{RESET} ({len(sgs)} subgraphs){degen_tag}")
            for sg in sgs:
                ct = dict(Counter(sg.child_concept_types))
                same_tag = ""
                print(f"    {sg.atom}/{sg.node_name}  "
                      f"({sg.n_children} children)  {DIM}{ct}{RESET}{same_tag}")

    # Exact matches with Jaccard
    if exact_matches:
        print(f"\n{BOLD}Exact Matches ({len(exact_matches)} pairs):{RESET}")
        for pair in exact_matches:
            same = pair.a.atom == pair.b.atom
            tag = f"  {DIM}[SAME CDG]{RESET}" if same else ""
            degen = is_degenerate(pair.a) and is_degenerate(pair.b)
            dtag = f"  {DIM}[DEGENERATE]{RESET}" if degen else ""
            color = GREEN if pair.jaccard >= 0.7 else (YELLOW if pair.jaccard >= 0.4 else RED)
            print(f"\n  {pair.a.atom}/{pair.a.node_name} <-> {pair.b.atom}/{pair.b.node_name}"
                  f"{tag}{dtag}")
            print(f"    Jaccard(concept_types): {color}{pair.jaccard:.3f}{RESET}")
            print(f"    A: {pair.a_types}")
            print(f"    B: {pair.b_types}")

    # Top similar pairs
    print(f"\n{BOLD}Top-{len(top_similar)} Similar Pairs (by Jaccard on concept_types):{RESET}")
    for rank, pair in enumerate(top_similar, 1):
        same = pair.a.atom == pair.b.atom
        tag = f" {DIM}[SAME CDG]{RESET}" if same else ""
        hm = f" {CYAN}[HASH MATCH]{RESET}" if pair.hash_match else ""
        degen = is_degenerate(pair.a) and is_degenerate(pair.b)
        dtag = f" {DIM}[DEGEN]{RESET}" if degen else ""
        color = GREEN if pair.jaccard >= 0.7 else YELLOW

        print(f"\n  {rank:>2}. Jaccard={color}{pair.jaccard:.3f}{RESET}"
              f"  {pair.a.atom}/{pair.a.node_name} <-> "
              f"{pair.b.atom}/{pair.b.node_name}{hm}{dtag}{tag}")
        if verbose or pair.jaccard >= 0.7:
            print(f"      A ({pair.a.n_children} children): {pair.a_types}")
            print(f"      B ({pair.b.n_children} children): {pair.b_types}")

    # Summary
    n_degen_exact = sum(1 for p in exact_matches
                        if is_degenerate(p.a) and is_degenerate(p.b))
    n_strong_exact = len(exact_matches) - n_degen_exact
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Exact hash matches: {len(exact_matches)} "
          f"({n_strong_exact} non-degenerate, {n_degen_exact} degenerate)")
    if top_similar:
        avg_jac = sum(p.jaccard for p in top_similar) / len(top_similar)
        print(f"  Top-{len(top_similar)} avg Jaccard: {avg_jac:.3f}")

    return {
        "summary": {
            "total_subgraphs": len(subgraphs),
            "unique_hashes": len(hash_groups),
            "collision_groups": len(collision_groups),
            "exact_matches": len(exact_matches),
            "non_degenerate_matches": n_strong_exact,
        },
        "collision_groups": {
            h: [{"atom": sg.atom, "node_id": sg.node_id,
                 "n_children": sg.n_children,
                 "concept_types": dict(Counter(sg.child_concept_types)),
                 "is_degenerate": is_degenerate(sg)}
                for sg in sgs]
            for h, sgs in collision_groups.items()
        },
        "top_similar": [
            {"rank": i + 1, "jaccard": round(p.jaccard, 4),
             "hash_match": p.hash_match,
             "a": f"{p.a.atom}/{p.a.node_id}",
             "b": f"{p.b.atom}/{p.b.node_id}",
             "a_types": p.a_types, "b_types": p.b_types}
            for i, p in enumerate(top_similar)
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> tuple[bool, bool, float, int]:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    as_json = "--json" in sys.argv
    threshold = 0.3
    top_k = 20
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
        if arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
    return verbose, as_json, threshold, top_k


def main() -> None:
    verbose, as_json, threshold, top_k = parse_args()

    cdgs = load_all_cdgs()
    if not cdgs:
        print("No CDGs found.", file=sys.stderr)
        sys.exit(1)

    # Part 1: Type audit
    type_audits = [audit_types_per_cdg(cdg) for cdg in cdgs]
    vocab_pairs = build_cross_cdg_vocab(type_audits)

    if not as_json:
        type_data = report_type_audit(type_audits, vocab_pairs, verbose)
    else:
        type_data = _build_type_json(type_audits, vocab_pairs)

    # Part 2: Isomorphism audit
    subgraphs = collect_decomposed_subgraphs(cdgs)
    exact_matches, top_similar = run_pairwise_comparison(subgraphs, threshold, top_k)

    if not as_json:
        iso_data = report_isomorphism_audit(subgraphs, exact_matches, top_similar, verbose)
    else:
        iso_data = _build_iso_json(subgraphs, exact_matches, top_similar)

    if as_json:
        print(json.dumps({"type_audit": type_data, "iso_audit": iso_data}, indent=2))


def _build_type_json(audits: list[CDGTypeAudit],
                     vocab_pairs: list[tuple[str, str, int, set[str]]]) -> dict:
    total_occ = sum(len(a.occurrences) for a in audits)
    all_raw = {o.raw for a in audits for o in a.occurrences}
    all_normed = {o.normalized for a in audits for o in a.occurrences}
    total_healthy = sum(
        sum(1 for o in a.occurrences
            if o.classification in ("primitive", "domain_type", "structural"))
        for a in audits
    )
    return {
        "summary": {
            "total_occurrences": total_occ,
            "unique_raw": len(all_raw),
            "unique_normalized": len(all_normed),
            "overall_health": round(total_healthy / total_occ, 4) if total_occ else 1.0,
        },
        "per_cdg": [
            {"atom": a.atom_name, "health_score": round(a.health_score, 4),
             "distribution": dict(a.class_dist)}
            for a in sorted(audits, key=lambda x: x.atom_name)
        ],
        "cross_cdg_top15": [
            {"a": na, "b": nb, "shared": cnt, "types": sorted(shared)}
            for na, nb, cnt, shared in vocab_pairs[:15]
        ],
    }


def _build_iso_json(subgraphs: list[DecomposedSubgraph],
                    exact_matches: list[SubgraphPair],
                    top_similar: list[SubgraphPair]) -> dict:
    hash_groups: dict[str, list[DecomposedSubgraph]] = defaultdict(list)
    for sg in subgraphs:
        hash_groups[sg.topo_hash].append(sg)
    collision_groups = {h: sgs for h, sgs in hash_groups.items() if len(sgs) >= 2}
    return {
        "summary": {
            "total_subgraphs": len(subgraphs),
            "unique_hashes": len(hash_groups),
            "collision_groups": len(collision_groups),
            "exact_match_pairs": len(exact_matches),
        },
        "collisions": {
            h: [{"atom": sg.atom, "node_id": sg.node_id, "n_children": sg.n_children,
                 "is_degenerate": is_degenerate(sg)}
                for sg in sgs]
            for h, sgs in collision_groups.items()
        },
        "top_similar": [
            {"rank": i + 1, "jaccard": round(p.jaccard, 4),
             "hash_match": p.hash_match,
             "a": f"{p.a.atom}/{p.a.node_id}", "b": f"{p.b.atom}/{p.b.node_id}",
             "a_types": p.a_types, "b_types": p.b_types}
            for i, p in enumerate(top_similar)
        ],
    }


if __name__ == "__main__":
    main()
