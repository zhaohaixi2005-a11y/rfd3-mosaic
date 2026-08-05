"""Audit execution and bounds of compiler-declared component mobility."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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

    component_reports: list[dict[str, Any]] = []
    for index, declaration in enumerate(declared):
        record = observed[index] if index < len(observed) else None
        if not isinstance(record, dict):
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
        maximum_translation = float(declaration["max_translation"])
        maximum_rotation = float(declaration["max_rotation_deg"])
        finite = all(
            math.isfinite(value) for value in translations + rotations
        )
        translation_observed = max(translations, default=float("inf"))
        rotation_observed = max(rotations, default=float("inf"))
        passed = bool(
            translations
            and rotations
            and finite
            and translation_observed <= maximum_translation + tolerance
            and rotation_observed <= maximum_rotation + tolerance
        )
        component_reports.append(
            {
                "component_id": declaration.get("coupling_group_id"),
                "constraint_orbit_id": declaration.get(
                    "constraint_orbit_id"
                ),
                "passed": passed,
                "maximum_translation_allowed": maximum_translation,
                "maximum_translation_observed": translation_observed,
                "maximum_rotation_deg_allowed": maximum_rotation,
                "maximum_rotation_deg_observed": rotation_observed,
            }
        )

    runtime_active = bool(
        diagnostics.get("apply_updates")
        and int(diagnostics.get("update_calls", 0)) > 0
        and int(diagnostics.get("active_window_calls", 0)) > 0
        and int(diagnostics.get("conditioning_refresh_count", 0)) > 0
        and int(diagnostics.get("mobile_orbit_count", -1)) == len(declared)
        and len(observed) == len(declared)
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


__all__ = ["audit_component_mobility"]
