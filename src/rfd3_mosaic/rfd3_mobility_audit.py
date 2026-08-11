"""Audit execution and bounds of compiler-declared component mobility."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_OBJECTIVE_TERMS = ("total", "junction", "clash", "tilt", "prior")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _single_example(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if len(payload) != 1:
        raise ValueError("Compiled input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled example must be an object")
    return example


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(_finite_number(component) for component in value)
    ):
        return None
    return tuple(float(component) for component in value)


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _scale(
    vector: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(factor * value for value in vector)


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _directional_motion_metrics(
    *,
    record: dict[str, Any],
    subspace: str | None,
    symmetry_axis: Any,
    tolerance: float,
) -> dict[str, Any]:
    restricted = subspace in {
        "radial",
        "radial_axial",
        "radial_rotation",
        "radial_axial_rotation",
    }
    if not restricted:
        return {
            "directional_contract_required": False,
            "directional_contract_valid": True,
        }
    if not isinstance(symmetry_axis, dict):
        return {
            "directional_contract_required": True,
            "directional_contract_valid": False,
            "directional_contract_failure": "missing symmetry axis",
        }
    axis_point = _vector3(symmetry_axis.get("point"))
    axis_direction = _vector3(symmetry_axis.get("direction"))
    translations = record.get("translation_vectors")
    centers = record.get("template_master_centers")
    if (
        axis_point is None
        or axis_direction is None
        or not isinstance(translations, list)
        or not isinstance(centers, list)
        or not translations
        or len(translations) != len(centers)
    ):
        return {
            "directional_contract_required": True,
            "directional_contract_valid": False,
            "directional_contract_failure": (
                "missing finite translation vectors or template centers"
            ),
        }
    axis_norm = _norm(axis_direction)
    if axis_norm <= tolerance:
        return {
            "directional_contract_required": True,
            "directional_contract_valid": False,
            "directional_contract_failure": "degenerate symmetry axis",
        }
    axis_unit = _scale(axis_direction, 1.0 / axis_norm)
    radial_values: list[float] = []
    axial_values: list[float] = []
    tangential_values: list[float] = []
    for raw_translation, raw_center in zip(
        translations,
        centers,
        strict=True,
    ):
        translation = _vector3(raw_translation)
        center = _vector3(raw_center)
        if translation is None or center is None:
            return {
                "directional_contract_required": True,
                "directional_contract_valid": False,
                "directional_contract_failure": (
                    "non-finite translation vector or template center"
                ),
            }
        center_offset = _subtract(center, axis_point)
        radial = _subtract(
            center_offset,
            _scale(axis_unit, _dot(center_offset, axis_unit)),
        )
        radial_norm = _norm(radial)
        if radial_norm <= tolerance:
            return {
                "directional_contract_required": True,
                "directional_contract_valid": False,
                "directional_contract_failure": (
                    "component center lies on the symmetry axis"
                ),
            }
        radial_unit = _scale(radial, 1.0 / radial_norm)
        axial = _dot(translation, axis_unit)
        radial_component = _dot(translation, radial_unit)
        tangential = _subtract(
            _subtract(
                translation,
                _scale(axis_unit, axial),
            ),
            _scale(radial_unit, radial_component),
        )
        radial_values.append(radial_component)
        axial_values.append(axial)
        tangential_values.append(_norm(tangential))

    maximum_axial = max((abs(value) for value in axial_values), default=0.0)
    maximum_tangential = max(tangential_values, default=0.0)
    axial_allowed = subspace in {"radial_axial", "radial_axial_rotation"}
    valid = bool(
        maximum_tangential <= tolerance
        and (axial_allowed or maximum_axial <= tolerance)
    )
    return {
        "directional_contract_required": True,
        "directional_contract_valid": valid,
        "maximum_radial_translation_observed": max(
            (abs(value) for value in radial_values),
            default=0.0,
        ),
        "maximum_axial_translation_observed": maximum_axial,
        "maximum_tangential_translation_leakage": maximum_tangential,
        "axial_translation_allowed": axial_allowed,
    }


def _objective_record_is_finite(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for phase in ("initial", "proposed", "delta"):
        terms = value.get(phase)
        if not isinstance(terms, dict) or not all(
            _finite_number(terms.get(term)) for term in _OBJECTIVE_TERMS
        ):
            return False
    return True


def _strict_json_value(value: Any) -> Any:
    """Replace non-finite diagnostic sentinels with JSON null."""

    if isinstance(value, dict):
        return {
            str(key): _strict_json_value(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_strict_json_value(child) for child in value]
    if isinstance(value, tuple):
        return [_strict_json_value(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_mobility_trajectory(
    *,
    result_json: str | Path,
    output: str | Path,
) -> bool:
    """Write a small, strict-JSON mobility artifact when diagnostics exist."""

    result_path = Path(result_json).resolve()
    output_path = Path(output).resolve()
    diagnostics = _load(result_path).get("motif_mobility_diagnostics")
    if diagnostics is None:
        return False
    if not isinstance(diagnostics, dict):
        raise ValueError("Result mobility diagnostics must be an object")
    trajectory = diagnostics.get("trajectory")
    if not isinstance(trajectory, list):
        raise ValueError("Result mobility trajectory must be a list")
    payload = _strict_json_value(
        {
            "artifact": "rfd3_mosaic.mobility_trajectory",
            "schema_version": 1,
            "source_result_json": str(result_path),
            "proposal_source": diagnostics.get("proposal_source"),
            "mobile_orbit_count": diagnostics.get("mobile_orbit_count"),
            "constraint_runtime": diagnostics.get("constraint_runtime"),
            "scaffold_guidance_config": diagnostics.get(
                "scaffold_guidance_config"
            ),
            "final_orbits": diagnostics.get("orbits", []),
            "trajectory": trajectory,
        }
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return True


def audit_component_mobility(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    input_path = Path(compiled_input).resolve()
    result_path = Path(result_json).resolve()
    example = _single_example(input_path)
    declared = [
        orbit
        for orbit in (example.get("extra") or {}).get(
            "motif_constraint_orbits", []
        )
        if orbit.get("mobility_mode") == "orbit_rigid"
    ]
    if not declared:
        raise ValueError("Compiled input declares no bounded mobile component")
    diagnostics = _load(result_path).get("motif_mobility_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Result metadata lacks motif mobility diagnostics")
    observed = diagnostics.get("orbits")
    if not isinstance(observed, list):
        raise ValueError("Motif mobility diagnostics lack orbit records")
    trajectory = diagnostics.get("trajectory", [])
    if not isinstance(trajectory, list):
        raise ValueError("Motif mobility diagnostics trajectory must be a list")
    constraint_runtime = diagnostics.get("constraint_runtime")
    runtime_counts = (
        constraint_runtime.get("phase_counts")
        if isinstance(constraint_runtime, dict)
        else None
    )
    expected_refreshes = int(
        diagnostics.get("conditioning_refresh_count", 0)
    )
    expected_proposals = int(diagnostics.get("update_calls", 0))
    constraint_runtime_valid = bool(
        isinstance(constraint_runtime, dict)
        and isinstance(runtime_counts, dict)
        and constraint_runtime.get("state") == "finalized"
        and int(runtime_counts.get("initialize", -1)) == 1
        and int(runtime_counts.get("finalize", -1)) == 1
        and int(runtime_counts.get("model_prediction", 0)) > 0
        and int(runtime_counts.get("state_update", -1))
        == int(runtime_counts.get("model_prediction", -2))
        and int(runtime_counts.get("proposal", -1))
        == expected_proposals
        and int(runtime_counts.get("proposal_applied", -1))
        == max(expected_refreshes - 1, 0)
        and int(
            constraint_runtime.get("conditioning_refresh_count", -1)
        )
        == expected_refreshes
        and str(constraint_runtime.get("proposal_source"))
        == str(diagnostics.get("proposal_source"))
        and int(constraint_runtime.get("proposal_interval", -1))
        == int(diagnostics.get("update_interval", -2))
    )

    declared_ids = [
        str(
            declaration.get("constraint_orbit_id")
            or f"orbit_{index}"
        )
        for index, declaration in enumerate(declared)
    ]
    declared_component_ids = [
        str(
            declaration.get("coupling_group_id")
            or declared_ids[index]
        )
        for index, declaration in enumerate(declared)
    ]
    identifier_contract_valid = bool(
        len(set(declared_ids)) == len(declared_ids)
        and len(set(declared_component_ids)) == len(declared_component_ids)
        and all(declared_ids)
        and all(declared_component_ids)
    )
    observed_by_id = {
        str(record["constraint_orbit_id"]): record
        for record in observed
        if isinstance(record, dict) and record.get("constraint_orbit_id")
    }
    symmetry_axis = diagnostics.get("symmetry_axis")

    component_reports: list[dict[str, Any]] = []
    group_action_contracts: list[bool] = []
    for index, declaration in enumerate(declared):
        record = observed_by_id.get(declared_ids[index])
        if record is None and len(declared) == 1 and index < len(observed):
            record = observed[index]
        if not isinstance(record, dict):
            group_action_contracts.append(False)
            component_reports.append(
                {
                    "component_id": declaration.get("coupling_group_id"),
                    "passed": False,
                    "failure": "missing runtime orbit diagnostics",
                }
            )
            continue
        translations = [
            float(value) for value in record.get("translation_norms", [])
        ]
        rotations = [
            float(value) for value in record.get("rotation_degrees", [])
        ]
        maximum_translation = float(
            declaration.get("max_translation") or 0.0
        )
        maximum_rotation = float(
            declaration.get("max_rotation_deg") or 0.0
        )
        declared_transform_ids = [
            int(value)
            for value in declaration.get("group_transform_ids", [])
        ]
        observed_transform_ids = [
            int(value)
            for value in record.get("group_transform_ids", [])
        ]
        observed_action_count = int(
            record.get("group_action_count", len(observed_transform_ids))
        )
        group_action_contract = bool(
            not declared_transform_ids
            or (
                len(set(declared_transform_ids))
                == len(declared_transform_ids)
                and observed_action_count == len(declared_transform_ids)
                and observed_transform_ids == declared_transform_ids
            )
        )
        group_action_contracts.append(group_action_contract)
        finite = all(
            math.isfinite(value) for value in translations + rotations
        )
        translation_observed = max(translations, default=float("inf"))
        rotation_observed = max(rotations, default=float("inf"))
        subspace = declaration.get("mobility_subspace")
        directional = _directional_motion_metrics(
            record=record,
            subspace=str(subspace) if subspace is not None else None,
            symmetry_axis=symmetry_axis,
            tolerance=tolerance,
        )
        passed = bool(
            translations
            and rotations
            and finite
            and group_action_contract
            and translation_observed <= maximum_translation + tolerance
            and rotation_observed <= maximum_rotation + tolerance
            and directional["directional_contract_valid"]
        )
        component_reports.append(
            {
                "component_id": declaration.get("coupling_group_id"),
                "constraint_orbit_id": declaration.get(
                    "constraint_orbit_id"
                ),
                "mobility_subspace": subspace,
                "declared_group_action_count": len(
                    declared_transform_ids
                ),
                "runtime_group_action_count": observed_action_count,
                "group_action_contract_valid": group_action_contract,
                "passed": passed,
                "maximum_translation_allowed": maximum_translation,
                "maximum_translation_observed": translation_observed,
                "maximum_rotation_deg_allowed": maximum_rotation,
                "maximum_rotation_deg_observed": rotation_observed,
                **directional,
            }
        )

    scaffold_steps = [
        step
        for step in trajectory
        if isinstance(step, dict)
        and step.get("proposal_source") == "scaffold_boundary"
    ]
    active_scaffold_steps = [
        step
        for step in scaffold_steps
        if float(step.get("window_weight", 0.0)) > 0.0
    ]

    def valid_joint_step(step: dict[str, Any]) -> bool:
        proposals = step.get("orbit_proposals")
        if (
            step.get("atomic_joint_acceptance") is not True
            or not isinstance(proposals, list)
            or len(proposals) != len(declared)
        ):
            return False
        proposal_ids = [
            str(proposal.get("constraint_orbit_id") or "")
            for proposal in proposals
            if isinstance(proposal, dict)
        ]
        if (
            len(proposal_ids) != len(declared)
            or set(proposal_ids) != set(declared_ids)
        ):
            return False
        applied = bool(step.get("applied"))
        accepted = bool(step.get("accepted"))
        if applied and not accepted:
            return False
        if not all(
            _finite_number(step.get(key))
            for key in ("progress", "window_weight", "joint_energy_delta")
        ):
            return False
        for energy_key in ("initial_energy", "proposed_energy"):
            energy = step.get(energy_key)
            if not isinstance(energy, dict) or not all(
                _finite_number(energy.get(term))
                for term in ("total", "junction", "clash")
            ):
                return False
        for proposal in proposals:
            if not isinstance(proposal, dict):
                return False
            local_accepted = bool(proposal.get("accepted"))
            expected_commit = applied and local_accepted
            if bool(proposal.get("committed")) != expected_commit:
                return False
            if bool(proposal.get("active")) and not (
                _objective_record_is_finite(proposal.get("objective"))
                and all(
                    _finite_number(value)
                    for value in proposal.get(
                        "proposed_delta_translation", []
                    )
                )
                and len(proposal.get("proposed_delta_translation", [])) == 3
                and _finite_number(
                    proposal.get("proposed_delta_rotation_degrees")
                )
            ):
                return False
        return True

    valid_joint_steps = [
        step for step in active_scaffold_steps if valid_joint_step(step)
    ]
    atomic_joint_runtime = bool(
        len(declared) <= 1
        or (
            identifier_contract_valid
            and active_scaffold_steps
            and len(valid_joint_steps) == len(active_scaffold_steps)
            and any(bool(step.get("applied")) for step in active_scaffold_steps)
        )
    )
    runtime_group_action_count = int(
        diagnostics.get("runtime_group_action_count", 0)
    )
    declared_action_counts = {
        len(declaration.get("group_transform_ids", []))
        for declaration in declared
        if declaration.get("group_transform_ids")
    }
    complete_group_action_orbits = bool(
        all(group_action_contracts)
        and (
            not declared_action_counts
            or (
                len(declared_action_counts) == 1
                and runtime_group_action_count
                == next(iter(declared_action_counts))
            )
        )
    )
    runtime_active = bool(
        diagnostics.get("apply_updates")
        and int(diagnostics.get("update_calls", 0)) > 0
        and int(diagnostics.get("active_window_calls", 0)) > 0
        and int(diagnostics.get("conditioning_refresh_count", 0)) > 0
        and int(diagnostics.get("mobile_orbit_count", -1)) == len(declared)
        and len(observed) == len(declared)
        and atomic_joint_runtime
        and complete_group_action_orbits
        and constraint_runtime_valid
    )
    passed = runtime_active and all(
        component["passed"] for component in component_reports
    )
    return {
        "audit": "rfd3_mosaic.bounded_component_mobility",
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "compiled_input": str(input_path),
            "result_json": str(result_path),
        },
        "thresholds": {"numeric_tolerance": tolerance},
        "summary": {
            "declared_mobile_components": len(declared),
            "runtime_mobile_components": len(observed),
            "runtime_active": runtime_active,
            "update_calls": int(diagnostics.get("update_calls", 0)),
            "active_window_calls": int(
                diagnostics.get("active_window_calls", 0)
            ),
            "conditioning_refresh_count": int(
                diagnostics.get("conditioning_refresh_count", 0)
            ),
            "atomic_joint_runtime": atomic_joint_runtime,
            "symmetry_id": diagnostics.get("symmetry_id"),
            "runtime_group_action_count": runtime_group_action_count,
            "complete_group_action_orbits": complete_group_action_orbits,
            "scaffold_trajectory_steps": len(scaffold_steps),
            "active_scaffold_trajectory_steps": len(
                active_scaffold_steps
            ),
            "valid_joint_trajectory_steps": len(valid_joint_steps),
            "identifier_contract_valid": identifier_contract_valid,
            "constraint_runtime_valid": constraint_runtime_valid,
            "components": component_reports,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bounded rigid motion of fixed components."
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_component_mobility(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
        tolerance=arguments.tolerance,
    )
    arguments.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        "Component mobility audit: "
        + ("PASSED" if report["passed"] else "FAILED")
    )
    print(
        "mobile components: "
        f"{summary['runtime_mobile_components']}/"
        f"{summary['declared_mobile_components']}"
    )
    print(f"active update windows: {summary['active_window_calls']}")
    print(f"report: {arguments.output.resolve()}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["audit_component_mobility", "write_mobility_trajectory"]
