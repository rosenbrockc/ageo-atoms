"""Focused runtime-probe coverage for rust_robotics family packets."""

from __future__ import annotations

from auditlib import runtime_probes


def _record(atom_name: str, module_import_path: str, wrapper_symbol: str) -> dict[str, object]:
    return {
        "atom_id": f"{atom_name}@ageoa/example.py:1",
        "atom_name": atom_name,
        "module_import_path": module_import_path,
        "module_path": "ageoa/example.py",
        "wrapper_symbol": wrapper_symbol,
        "wrapper_line": 1,
        "skeleton": False,
    }


def _assert_probe_passes(atom_name: str, module_import_path: str, wrapper_symbol: str) -> None:
    probe = runtime_probes.build_runtime_probe(_record(atom_name, module_import_path, wrapper_symbol))
    assert probe["status"] == "pass"
    assert probe["parity_used"] is True
    assert "RUNTIME_PROBE_PASS" in probe["findings"]
    assert "RUNTIME_CONTRACT_NEGATIVE_PASS" in probe["findings"]


def test_runtime_probe_passes_for_rust_robotics_top_level_wrappers() -> None:
    _assert_probe_passes(
        "ageoa.rust_robotics.atoms.n_joint_arm_solver",
        "ageoa.rust_robotics.atoms",
        "n_joint_arm_solver",
    )
    _assert_probe_passes(
        "ageoa.rust_robotics.atoms.dijkstra_path_planning",
        "ageoa.rust_robotics.atoms",
        "dijkstra_path_planning",
    )


def test_runtime_probe_passes_for_rust_robotics_bicycle_helpers() -> None:
    _assert_probe_passes(
        "ageoa.rust_robotics.bicycle_kinematic.constructgeometrymodel",
        "ageoa.rust_robotics.bicycle_kinematic.atoms",
        "constructgeometrymodel",
    )
    _assert_probe_passes(
        "ageoa.rust_robotics.bicycle_kinematic.computelinearizedstatematrices",
        "ageoa.rust_robotics.bicycle_kinematic.atoms",
        "computelinearizedstatematrices",
    )


def test_runtime_probe_passes_for_rust_robotics_longitudinal_dynamics() -> None:
    _assert_probe_passes(
        "ageoa.rust_robotics.longitudinal_dynamics.initialize_model",
        "ageoa.rust_robotics.longitudinal_dynamics.atoms",
        "initialize_model",
    )
    _assert_probe_passes(
        "ageoa.rust_robotics.longitudinal_dynamics.solve_control_for_target_derivative",
        "ageoa.rust_robotics.longitudinal_dynamics.atoms",
        "solve_control_for_target_derivative",
    )


def test_runtime_probe_passes_for_rust_robotics_n_joint_arm_2d() -> None:
    _assert_probe_passes(
        "ageoa.rust_robotics.n_joint_arm_2d.modelspecloadingandsizing",
        "ageoa.rust_robotics.n_joint_arm_2d.atoms",
        "modelspecloadingandsizing",
    )
    _assert_probe_passes(
        "ageoa.rust_robotics.n_joint_arm_2d.dynamicsandlinearizationkernel",
        "ageoa.rust_robotics.n_joint_arm_2d.atoms",
        "dynamicsandlinearizationkernel",
    )
