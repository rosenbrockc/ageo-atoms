# Future Ingester Extensions

This file captures lessons from repo-side trust-debt remediation that may warrant
future changes in `../ageo-matcher`.

## 2026-03-31

- Older ingest-derived wrappers still sometimes emit contracts that are stricter
  than the wrapper logic or upstream semantics require, especially for
  array-like parameters. A clear example is the `kalman_filters/static_kf`
  family, where generated preconditions preferred scalar-only checks even though
  the wrapper implementation naturally accepts NumPy array inputs after
  coercion. Future ingester hardening should bias toward conservative
  array-compatible contracts when emitted code normalizes inputs with
  `np.asarray`, `np.atleast_1d`, or `np.atleast_2d`, instead of over-strengthening
  public contracts to scalar-only forms.

- Older ingest-derived witness surfaces can also be semantically mismatched for
  stateful estimator families even when the wrapper bodies are serviceable. A
  concrete example is `pronto/dynamic_stance_estimator`, where witness
  signatures drift into unrelated abstract-distribution shapes instead of
  reflecting the wrapper's state-dict transition semantics. Future ingester
  hardening should keep witness generation conservative for stateful filters and
  estimators, preferring state-shaped abstract surrogates over conceptually
  unrelated witness abstractions.
