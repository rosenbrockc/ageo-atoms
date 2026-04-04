from __future__ import annotations

import json

from ageoa.rust_robotics.longitudinal_dynamics.atoms import deserialize_model_spec, initialize_model


def test_initialize_model_returns_typed_vehicle_spec() -> None:
    model = initialize_model(1800.0, 2.5)

    assert model == {
        "mass": 1800.0,
        "area_frontal": 2.5,
        "Cd": 0.3,
        "rho": 1.225,
        "Cr": 0.01,
        "g": 9.81,
    }


def test_deserialize_model_spec_loads_vehicle_spec(tmp_path) -> None:
    path = tmp_path / "vehicle.json"
    path.write_text(json.dumps({"mass": 1600.0, "area_frontal": 2.3}))

    model = deserialize_model_spec(str(path))

    assert model["mass"] == 1600.0
    assert model["area_frontal"] == 2.3
