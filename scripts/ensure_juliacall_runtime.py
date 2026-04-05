"""Provision a writable JuliaCall/PythonCall runtime for tests and CI."""

from __future__ import annotations

from ageoa_julia_runtime import prewarm_juliacall_project


def main() -> None:
    cfg = prewarm_juliacall_project()
    print(f"JULIA_EXE={cfg.julia_exe}")
    print(f"PYTHON_JULIAPKG_PROJECT={cfg.project}")
    print(f"PYTHON_JULIACALL_PROJECT={cfg.project}")
    print(f"JULIA_DEPOT_PATH={cfg.depot}")


if __name__ == "__main__":
    main()
