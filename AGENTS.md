# AGENTS

- Use the shared virtual environment in [../ageo-matcher/.venv](/Users/conrad/personal/ageo-matcher/.venv) for this repo.
- Run Python commands with `../ageo-matcher/.venv/bin/python`.
- Run tests with `../ageo-matcher/.venv/bin/python -m pytest ...` or `pytest ...` only when the active shell is already using that same environment.
- Do not create or rely on a repo-local `.venv` in `ageo-atoms`.
- Julia-backed tests and audits should use the writable runtime configured by [ageoa_julia_runtime.py](/Users/conrad/personal/ageo-atoms/ageoa_julia_runtime.py).
- For CI or fresh machines, prewarm that runtime once with [scripts/ensure_juliacall_runtime.py](/Users/conrad/personal/ageo-atoms/scripts/ensure_juliacall_runtime.py) before running Julia-backed tests.
