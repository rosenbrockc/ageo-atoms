#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGEOA = ROOT / 'ageoa'
MANIFEST_PATH = ROOT / 'data' / 'hyperparams' / 'manifest.json'
DB_PATH = ROOT / 'data' / 'hyperparams' / 'manifest.sqlite'
REFERENCES_REGISTRY_PATH = ROOT / 'data' / 'references' / 'registry.json'

DOMAIN_MAP = {
    'advancedvi': 'probabilistic_inference',
    'algorithms': 'algorithms',
    'alphafold': 'ml_biology',
    'astroflow': 'astronomy_timekeeping',
    'bayes_rs': 'probabilistic_inference',
    'belief_propagation': 'probabilistic_inference',
    'biosppy': 'signal_biosignal',
    'conjugate_priors': 'probabilistic_inference',
    'datadriven': 'scientific_ml',
    'e2e_ppg': 'signal_biosignal',
    'heartpy': 'signal_biosignal',
    'hftbacktest': 'quant_finance',
    'hPDB': 'molecular_biology',
    'institutional_quant_engine': 'quant_finance',
    'jax_advi': 'probabilistic_inference',
    'jFOF': 'scientific_python',
    'kalman_filters': 'state_estimation',
    'mcmc_foundational': 'probabilistic_inference',
    'mint': 'ml_sequence_models',
    'molecular_docking': 'molecular_docking',
    'neurokit2': 'signal_biosignal',
    'numpy': 'scientific_python',
    'particle_filters': 'state_estimation',
    'pasqal': 'quantum_optimization',
    'pronto': 'state_estimation',
    'pulsar': 'astronomy_timekeeping',
    'pulsar_folding': 'astronomy_timekeeping',
    'quant_engine': 'quant_finance',
    'quantfin': 'quant_finance',
    'rust_robotics': 'state_estimation',
    'scipy': 'scientific_python',
    'skyfield': 'astronomy_timekeeping',
    'tempo_jl': 'astronomy_timekeeping',
    'tempo.py': 'astronomy_timekeeping',
}
PLUMBING_EXACT = {
    'algorithm', 'axis', 'backend_flags', 'bounds', 'callback', 'check_finite', 'constrain_fun_dict',
    'constraints', 'copy', 'current_state', 'current_state_model', 'dense_output', 'device', 'dtype', 'events',
    'fs', 'full_output', 'generator', 'hess', 'hessp', 'include_nyquist', 'jac', 'keepdims', 'kernel_spec',
    'log_lik_fun', 'log_prior_fun', 'log_prob_oracle', 'method', 'model', 'model_spec', 'oracle', 'options',
    'order', 'output', 'overwrite_a', 'overwrite_b', 'overwrite_x', 'predicted_state', 'predicted_state_model',
    'prng_key_in', 'prob', 'random_state', 'register', 'return_diag', 'rng', 'rng_in', 'rng_key', 'rng_state_in',
    'sampling_rate', 'seed', 'state', 'state_in', 'state_model', 'subok', 'target', 'target_log_kernel',
    'theta_shape_dict', 'var_param_inits', 'vectorized'
}
PLUMBING_SUBSTRINGS = (
    'backend', 'bound', 'callback', 'constraint', 'device', 'dtype', 'event', 'generator', 'hess', 'jac', 'kernel',
    'lik', 'logp', 'method', 'model', 'oracle', 'order', 'output', 'prior', 'register', 'rng', 'seed', 'spec',
    'state', 'target', 'tensor_fn'
)
DATA_EXACT = {
    'L', 'W', 'a', 'adj', 'b', 'c', 'candidate_bandwidth', 'candidate_matrix', 'candidate_permutation', 'carry_weights',
    'clean_indices', 'coordinates_layout', 'count_dist', 'cov', 'current_iteration_state', 'd', 'data', 'delay_table',
    'detector_1', 'detector_2', 'ecg_cleaned', 'ecg_signal', 'envelope', 'fchan', 'final_register', 'graph', 'initial_positions',
    'initial_register', 'input_data', 'k', 'keys', 'labels', 'lattice_instance', 'mapping', 'mapping_context', 'mat', 'mat_list',
    'mean', 'measurement_counts', 'noise', 'noisy_indices', 'object', 'observation_t', 'onsets', 'pvals', 'ppg', 'prior_state',
    'proposed_particles', 'q', 'rates', 'results', 'returns', 'series', 'signal', 'signals', 'start_state', 'subject',
    'subject_dict', 'subject_idx', 'subjects', 't', 'thresholded_matrix', 'thresholds', 'times', 'trade_qty', 'tup', 'u', 'unexpanded_nodes',
    'unmapping', 'v', 'x', 'y', 'z'
}
CONFIG_RE = re.compile(
    r'(^n$|^k$|^m$|^p$|^df$|^dt$|^t$|^tol$|^eps(abs|rel)?$|threshold|window|order|degree|depth|iter|steps?'
    r'|n_|num|count|neighbors|smoothing|cutoff|freq|band|rate|period|lag|width|height|penalty|alpha|beta|gamma'
    r'|rho|sigma|theta|kappa|lambda|tau|temperature|dropout|regular|trim|ddof|side|sorter|norm$|loc$|scale$|size$'
    r'|shape$|damp|horizon|leapfrog|accept|refractory|search|bc_type|extrapolate|pass_zero|whole|analog|assume_a'
    r'|trans|weight|wvar|maxp1|limlst|limit|points|epsilon|nseg|num_sol|tree_depth|max_iter|step_size|learning'
    r'|momentum|friction|trials|test_size|train_size|correction_period|yaw_slip|disable_period|crossing|discount'
    r'|threshold_quantile|n_steps|n_sims|num_sims|max_degree|lambda_val)',
    re.I,
)
SCALAR_ANN_RE = re.compile(r'(^|\.)(int|float|bool|str|integer|number)(\[.*\])?$', re.I)
IGNORE_CALL_PREFIXES = (
    'NotImplementedError', 'RuntimeError', 'ValueError', 'ageoa.ghost.registry.register_atom', 'bool', 'builtins.', 'collections.', 'dataclasses.', 'dict',
    'enumerate', 'float', 'icontract.', 'int', 'itertools.', 'jax.', 'json.', 'len', 'list', 'math.', 'np.', 'numpy.',
    'print', 'range', 'scipy.', 'set', 'str', 'tuple', 'type', 'typing.', 'zip', 'isinstance'
)


def module_parts_for(path: Path) -> list[str]:
    rel = path.relative_to(AGEOA).with_suffix('')
    parts = list(rel.parts)
    if parts and parts[-1] == 'atoms':
        parts = parts[:-1]
    return parts


def build_import_map(tree: ast.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                mapping[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                mapping[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    return mapping


def resolve_call(target: ast.AST, imports: dict[str, str]) -> str | None:
    if isinstance(target, ast.Name):
        return imports.get(target.id, target.id)
    if isinstance(target, ast.Attribute):
        parts: list[str] = []
        node = target
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(imports.get(node.id, node.id))
            return '.'.join(reversed(parts))
    return None


def annotation_src(node: ast.AST | None) -> str:
    if node is None:
        return ''
    try:
        return ast.unparse(node)
    except Exception:
        return ''


def literal_value(node: ast.AST | None):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def is_registered(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == 'register_atom':
            return True
        if isinstance(target, ast.Attribute) and target.attr == 'register_atom':
            return True
    return False


def infer_param_kind(name: str, annotation: str, optional: bool) -> str:
    lname = name.lower()
    ann = annotation.lower()
    if lname in DATA_EXACT:
        return 'data'
    if lname in PLUMBING_EXACT or any(token in lname for token in PLUMBING_SUBSTRINGS):
        return 'plumbing'
    if CONFIG_RE.search(lname):
        return 'config'
    if SCALAR_ANN_RE.search(ann) and lname not in {'sampling_rate', 'fs'}:
        return 'config'
    if optional and lname not in DATA_EXACT:
        return 'api_control'
    return 'data'


def collect_atoms() -> list[dict]:
    atoms: list[dict] = []
    for path in sorted(AGEOA.rglob('*.py')):
        if path.name == '__init__.py' or path.name.endswith('witnesses.py') or '__pycache__' in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        imports = build_import_map(tree)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith('_') or not is_registered(node):
                continue
            parts = module_parts_for(path)
            atom_fqdn = '.'.join(['ageoa', *parts, node.name])
            atom_id = f'{atom_fqdn}@{path.relative_to(ROOT)}:{node.lineno}'
            args = list(node.args.args)
            defaults = list(node.args.defaults)
            first_default = len(args) - len(defaults)
            params = []
            for idx, arg in enumerate(args):
                params.append({
                    'name': arg.arg,
                    'annotation': annotation_src(arg.annotation),
                    'optional': idx >= first_default,
                    'kwonly': False,
                    'default': literal_value(defaults[idx - first_default]) if idx >= first_default else None,
                })
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                params.append({
                    'name': arg.arg,
                    'annotation': annotation_src(arg.annotation),
                    'optional': default is not None,
                    'kwonly': True,
                    'default': literal_value(default),
                })
            calls = []
            seen = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    ref = resolve_call(sub.func, imports)
                    if not ref or ref in seen or ref.startswith(IGNORE_CALL_PREFIXES):
                        continue
                    seen.add(ref)
                    calls.append(ref)
            top = parts[0] if parts else path.stem
            atoms.append({
                'atom_id': atom_id,
                'atom': atom_fqdn,
                'path': str(path.relative_to(ROOT)),
                'line': node.lineno,
                'module_family': top,
                'family': DOMAIN_MAP.get(top, top),
                'params': params,
                'calls': calls,
            })
    return atoms


def load_existing() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def manual_lookup(existing: dict) -> dict[str, dict]:
    lookup = {}
    for entry in existing.get('reviewed_atoms', []):
        key = entry.get('atom_id') or entry['atom']
        lookup[key] = entry
    return lookup


def attach_impl_locator(entry: dict, atom_meta: dict) -> dict:
    updated = dict(entry)
    updated['atom_id'] = atom_meta['atom_id']
    updated['atom'] = atom_meta['atom']
    updated['path'] = atom_meta['path']
    updated['module_family'] = atom_meta['module_family']
    updated['family'] = updated.get('family', atom_meta['family'])
    updated['scholarly_references'] = list(updated.get('scholarly_references', []))
    provenance = updated.get('provenance', [])
    upstream_refs = [
        {'type': item.get('type'), 'reference': item.get('reference')}
        for item in provenance
        if item.get('type') in {'third_party_source', 'official_api_docs', 'paper_or_method_doc', 'targeted_web_search'}
    ]
    updated['implementation_locator'] = {
        'manifest_key': atom_meta['atom_id'],
        'wrapper_symbol': atom_meta['atom'],
        'wrapper_path': atom_meta['path'],
        'wrapper_line': atom_meta['line'],
        'delegate_calls': atom_meta['calls'],
        'upstream_references': upstream_refs,
    }
    return updated


def generate_entry(atom_meta: dict) -> dict:
    classified = []
    for param in atom_meta['params']:
        enriched = dict(param)
        enriched['inferred_kind'] = infer_param_kind(param['name'], param['annotation'], param['optional'])
        classified.append(enriched)
    config = [p for p in classified if p['inferred_kind'] == 'config']
    api_control = [p for p in classified if p['inferred_kind'] == 'api_control']
    plumbing = [p for p in classified if p['inferred_kind'] == 'plumbing']
    blocked_params = []
    if config:
        status = 'blocked'
        reason = 'Conservative repository-wide pass found behavior-shaping or scalar control parameters, but no source-backed search bounds have been approved yet.'
        prov_note = 'Conservative wrapper audit only. This atom still needs upstream source/doc review before Principal can tune it.'
        for param in config:
            item = {
                'name': param['name'],
                'kind': 'unreviewed',
                'reason': 'Behavior-shaping or scalar control parameter found during conservative wrapper audit; no source-backed search range is approved yet.',
            }
            if param['default'] is not None:
                item['default'] = param['default']
            blocked_params.append(item)
        for param in api_control:
            item = {
                'name': param['name'],
                'kind': 'unreviewed',
                'reason': 'Optional API control is exposed, but the optimizer policy for it has not been source-backed yet.',
            }
            if param['default'] is not None:
                item['default'] = param['default']
            blocked_params.append(item)
    else:
        status = 'fixed'
        reason = 'Conservative repository-wide pass found only data, metadata, or API-plumbing inputs and no approved tunable search space.'
        prov_note = 'Conservative wrapper audit only. No source-backed hyperparameter search space is approved for this atom.'
        for param in api_control + plumbing:
            item = {
                'name': param['name'],
                'kind': 'non_optimizable',
                'reason': 'Library/API control, metadata, or runtime plumbing; not a Principal hyperparameter candidate in the conservative pass.',
            }
            if param['default'] is not None:
                item['default'] = param['default']
            blocked_params.append(item)
    return {
        'atom_id': atom_meta['atom_id'],
        'atom': atom_meta['atom'],
        'path': atom_meta['path'],
        'family': atom_meta['family'],
        'module_family': atom_meta['module_family'],
        'scholarly_references': [],
        'status': status,
        'reason': reason,
        'tunable_params': [],
        'blocked_params': blocked_params,
        'implementation_locator': {
            'manifest_key': atom_meta['atom_id'],
            'wrapper_symbol': atom_meta['atom'],
            'wrapper_path': atom_meta['path'],
            'wrapper_line': atom_meta['line'],
            'delegate_calls': atom_meta['calls'],
            'upstream_references': [],
        },
        'provenance': [
            {
                'type': 'wrapper_code',
                'reference': atom_meta['path'],
                'notes': prov_note,
            }
        ],
        'audit_method': 'conservative_wrapper_review',
    }


def build_manifest(existing: dict, atoms: list[dict]) -> dict:
    manual = manual_lookup(existing)
    reviewed = []
    for atom_meta in atoms:
        key = atom_meta['atom_id'] if atom_meta['atom_id'] in manual else atom_meta['atom']
        if key in manual:
            reviewed.append(attach_impl_locator(manual[key], atom_meta))
        else:
            reviewed.append(generate_entry(atom_meta))
    reviewed.sort(key=lambda item: (item['atom'], item['path'], item.get('implementation_locator', {}).get('wrapper_line', 0)))

    family_summary: dict[str, dict[str, int]] = {}
    for entry in reviewed:
        fam = family_summary.setdefault(entry['family'], {'reviewed_atoms': 0, 'approved': 0, 'fixed': 0, 'blocked': 0, 'deprecated': 0})
        fam['reviewed_atoms'] += 1
        fam[entry['status']] += 1

    delegate_lookup: dict[str, list[str]] = defaultdict(list)
    upstream_lookup: dict[str, list[str]] = defaultdict(list)
    for entry in reviewed:
        impl = entry.get('implementation_locator', {})
        for ref in impl.get('delegate_calls', []):
            delegate_lookup[ref].append(entry['atom_id'])
        for ref in impl.get('upstream_references', []):
            value = ref.get('reference') if isinstance(ref, dict) else ref
            if value:
                upstream_lookup[value].append(entry['atom_id'])

    return {
        'schema_version': '0.4',
        'repo': 'ageo-atoms',
        'canonical_source': 'data/hyperparams/manifest.json',
        'runtime_index': 'data/hyperparams/manifest.sqlite',
        'audit_policy': {
            'source_order': ['wrapper_code', 'third_party_source', 'official_api_docs', 'paper_or_method_doc', 'targeted_web_search'],
            'approval_rule': 'Do not approve a tunable search range unless upstream code/docs/papers justify a safe domain.',
            'generation_rule': 'Canonical JSON is the human-reviewed source of truth. SQLite is a generated runtime/query index.',
        },
        'status_definitions': existing['status_definitions'],
        'family_summary': dict(sorted(family_summary.items())),
        'implementation_index': {
            'delegate_call_count': len(delegate_lookup),
            'upstream_reference_count': len(upstream_lookup),
        },
        'reviewed_atoms': reviewed,
    }


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + '\n')


def create_db(manifest: dict) -> None:
    # Load reference registry for denormalized scholarly_references table
    ref_registry: dict[str, dict] = {}
    if REFERENCES_REGISTRY_PATH.exists():
        reg = json.loads(REFERENCES_REGISTRY_PATH.read_text())
        ref_registry = reg.get('references', {})

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(
        '''
        PRAGMA journal_mode=WAL;
        CREATE TABLE atoms (
          atom_id TEXT PRIMARY KEY,
          fqdn TEXT NOT NULL,
          path TEXT NOT NULL,
          module_family TEXT NOT NULL,
          domain_family TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          audit_method TEXT NOT NULL
        );
        CREATE TABLE implementation_refs (
          atom_id TEXT NOT NULL,
          ref_kind TEXT NOT NULL,
          ref_value TEXT NOT NULL,
          FOREIGN KEY(atom_id) REFERENCES atoms(atom_id)
        );
        CREATE TABLE hyperparams (
          atom_id TEXT NOT NULL,
          name TEXT NOT NULL,
          status TEXT NOT NULL,
          kind TEXT,
          default_value TEXT,
          min_value TEXT,
          max_value TEXT,
          step_value TEXT,
          log_scale INTEGER,
          choices_json TEXT,
          constraints_json TEXT,
          semantic_role TEXT,
          range_source TEXT,
          source_reference TEXT,
          source_confidence TEXT,
          reason TEXT,
          FOREIGN KEY(atom_id) REFERENCES atoms(atom_id)
        );
        CREATE TABLE provenance (
          atom_id TEXT NOT NULL,
          source_type TEXT NOT NULL,
          reference TEXT NOT NULL,
          notes TEXT,
          FOREIGN KEY(atom_id) REFERENCES atoms(atom_id)
        );
        CREATE TABLE manifest_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE scholarly_references (
          atom_id TEXT NOT NULL,
          ref_id TEXT NOT NULL,
          ref_type TEXT NOT NULL,
          doi TEXT,
          url TEXT,
          title TEXT,
          authors_json TEXT,
          year INTEGER,
          venue TEXT,
          match_type TEXT,
          similarity_score REAL,
          confidence TEXT,
          FOREIGN KEY(atom_id) REFERENCES atoms(atom_id)
        );
        CREATE INDEX idx_atoms_fqdn ON atoms(fqdn);
        CREATE INDEX idx_atoms_domain_family ON atoms(domain_family);
        CREATE INDEX idx_atoms_status ON atoms(status);
        CREATE INDEX idx_impl_ref_value ON implementation_refs(ref_value);
        CREATE INDEX idx_hyperparams_atom_name ON hyperparams(atom_id, name);
        CREATE INDEX idx_provenance_atom ON provenance(atom_id);
        CREATE INDEX idx_scholarly_refs_atom ON scholarly_references(atom_id);
        CREATE INDEX idx_scholarly_refs_doi ON scholarly_references(doi);
        CREATE INDEX idx_scholarly_refs_refid ON scholarly_references(ref_id);
        '''
    )
    cur.executemany(
        'INSERT INTO manifest_meta(key, value) VALUES (?, ?)',
        [
            ('schema_version', manifest['schema_version']),
            ('canonical_source', manifest['canonical_source']),
            ('runtime_index', manifest['runtime_index']),
        ],
    )
    for entry in manifest['reviewed_atoms']:
        cur.execute(
            'INSERT INTO atoms(atom_id, fqdn, path, module_family, domain_family, status, reason, audit_method) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                entry['atom_id'], entry['atom'], entry['path'], entry.get('module_family', ''), entry.get('family', ''),
                entry['status'], entry['reason'], entry.get('audit_method', 'unknown')
            ),
        )
        impl = entry.get('implementation_locator', {})
        refs = [
            ('manifest_key', impl.get('manifest_key')),
            ('wrapper_symbol', impl.get('wrapper_symbol')),
            ('wrapper_path', impl.get('wrapper_path')),
        ]
        if impl.get('wrapper_line') is not None:
            refs.append(('wrapper_line', str(impl['wrapper_line'])))
        refs.extend(('delegate_call', value) for value in impl.get('delegate_calls', []))
        for ref in impl.get('upstream_references', []):
            if isinstance(ref, dict):
                refs.append((ref.get('type', 'upstream_reference'), ref.get('reference')))
            else:
                refs.append(('upstream_reference', ref))
        for kind, value in refs:
            if value:
                cur.execute('INSERT INTO implementation_refs(atom_id, ref_kind, ref_value) VALUES (?, ?, ?)', (entry['atom_id'], kind, value))
        for group_name, status in (('tunable_params', 'approved'), ('blocked_params', 'blocked')):
            for item in entry.get(group_name, []):
                cur.execute(
                    'INSERT INTO hyperparams(atom_id, name, status, kind, default_value, min_value, max_value, step_value, log_scale, choices_json, constraints_json, semantic_role, range_source, source_reference, source_confidence, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        entry['atom_id'], item['name'], status, item.get('kind'), json.dumps(item.get('default')), json.dumps(item.get('min_value')),
                        json.dumps(item.get('max_value')), json.dumps(item.get('step')), int(bool(item.get('log_scale'))), json.dumps(item.get('choices')),
                        json.dumps(item.get('constraints')), item.get('semantic_role'), item.get('range_source'), item.get('source_reference'), item.get('source_confidence'), item.get('reason')
                    ),
                )
        for prov in entry.get('provenance', []):
            if prov.get('reference'):
                cur.execute(
                    'INSERT INTO provenance(atom_id, source_type, reference, notes) VALUES (?, ?, ?, ?)',
                    (entry['atom_id'], prov.get('type', 'unknown'), prov['reference'], prov.get('notes')),
                )
        for ref_id in entry.get('scholarly_references', []):
            ref = ref_registry.get(ref_id, {})
            match_meta = ref.get('match_metadata', {})
            cur.execute(
                'INSERT INTO scholarly_references(atom_id, ref_id, ref_type, doi, url, title, authors_json, year, venue, match_type, similarity_score, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    entry['atom_id'], ref_id, ref.get('type', 'unknown'), ref.get('doi'), ref.get('url'),
                    ref.get('title'), json.dumps(ref.get('authors', [])), ref.get('year'), ref.get('venue'),
                    match_meta.get('match_type'), match_meta.get('similarity_score'), match_meta.get('confidence'),
                ),
            )
    conn.commit()
    conn.close()


def main() -> None:
    manifest = build_manifest(load_existing(), collect_atoms())
    write_manifest(manifest)
    create_db(manifest)
    counts = Counter(entry['status'] for entry in manifest['reviewed_atoms'])
    print(json.dumps({
        'atoms': len(manifest['reviewed_atoms']),
        'status_counts': dict(counts),
        'families': len(manifest['family_summary']),
        'db': str(DB_PATH.relative_to(ROOT)),
    }, indent=2))

if __name__ == '__main__':
    main()
