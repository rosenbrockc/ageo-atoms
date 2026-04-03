from __future__ import annotations

from auditlib import runtime_probes

_module = runtime_probes.safe_import_module("ageoa.e2e_ppg.template_matching")
templatefeaturecomputation = _module.templatefeaturecomputation


def test_templatefeaturecomputation_returns_feature_tuple() -> None:
    heart_cycles = [
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.0, 0.4, 1.0, 0.6, 0.0],
        [0.0, 0.6, 1.0, 0.4, 0.0],
    ]

    result = templatefeaturecomputation(heart_cycles)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(value, float) for value in result)
