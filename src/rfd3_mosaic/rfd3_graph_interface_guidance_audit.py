"""Audit execution of unified output-stage graph-interface guidance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _single_example(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if len(payload) != 1:
        raise ValueError("Compiled input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled example must be an object")
    return example


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def audit_graph_interface_guidance(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
) -> dict[str, Any]:
    """Prove that every declared design edge entered the sampler field."""

    input_path = Path(compiled_input).resolve()
    result_path = Path(result_json).resolve()
    example = _single_example(input_path)
    declared = [
        relation
        for relation in (example.get("extra") or {}).get(
            "assembly_interface_relations", []
        )
        if bool(relation.get("required", True))
        and relation.get("satisfaction_stage") == "output"
        and (relation.get("target_geometry") or {}).get("mode")
        == "geometric_constraints"
    ]
    if not declared:
        raise ValueError(
            "Compiled input declares no required output-stage contact edge"
        )
    expected_ids = [str(edge["edge_instance_id"]) for edge in declared]
    expected_source_ids = [
        str(
            edge.get("source_interface_id")
            or str(edge["edge_instance_id"]).split("@", 1)[0]
        )
        for edge in declared
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Compiled output-stage interface IDs are not unique")

    diagnostics = _load(result_path).get(
        "graph_interface_guidance_diagnostics"
    )
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics_schema_version = int(diagnostics.get("schema_version", 1))
    observed_ids = [str(value) for value in diagnostics.get("edge_ids", [])]
    observed_source_ids = [
        str(value)
        for value in diagnostics.get("source_interface_ids", [])
    ]
    steps = diagnostics.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    applied = [step for step in steps if bool(step.get("applied"))]
    finite_applied_steps = []
    packing_evidence_steps = []
    for step in applied:
        base_evidence = (
            all(
                _finite(step.get(key))
                for key in (
                    "window_weight",
                    "energy",
                    "attraction",
                    "clash",
                    "distance",
                    "maximum_token_step",
                )
            )
            and len(step.get("minimum_distances", []))
            == len(expected_ids)
            and all(
                _finite(value)
                for value in step.get("minimum_distances", [])
            )
        )
        packing_evidence = (
            diagnostics_schema_version < 2
            or (
                _finite(step.get("coverage"))
                and _finite(step.get("continuity"))
                and _finite(step.get("mean_token_step"))
                and len(step.get("covered_left_residues", []))
                == len(expected_ids)
                and len(step.get("covered_right_residues", []))
                == len(expected_ids)
                and len(step.get("target_residues_per_side", []))
                == len(expected_ids)
                and (
                    diagnostics_schema_version < 3
                    or len(
                        step.get(
                            "target_contiguous_residues_per_side",
                            [],
                        )
                    )
                    == len(expected_ids)
                )
                and len(step.get("contiguous_left_residues", []))
                == len(expected_ids)
                and len(step.get("contiguous_right_residues", []))
                == len(expected_ids)
                and len(step.get("per_edge_total", []))
                == len(expected_ids)
            )
        )
        finite_applied_steps.append(base_evidence and packing_evidence)
        packing_evidence_steps.append(packing_evidence)
    runtime_active = diagnostics.get("runtime_active") is True
    identifier_contract = (
        len(observed_ids) == len(set(observed_ids))
        and set(observed_ids) == set(expected_ids)
        and int(diagnostics.get("edge_count", -1)) == len(expected_ids)
        and (
            diagnostics_schema_version < 2
            or observed_source_ids == expected_source_ids
        )
    )
    applied_count = int(diagnostics.get("applied_steps", -1))
    execution_contract = bool(
        steps
        and applied
        and applied_count == len(applied)
        and all(finite_applied_steps)
    )
    passed = runtime_active and identifier_contract and execution_contract
    return {
        "audit": "rfd3_mosaic.graph_interface_guidance",
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "compiled_input": str(input_path),
            "result_json": str(result_path),
        },
        "summary": {
            "diagnostics_schema_version": diagnostics_schema_version,
            "runtime_active": runtime_active,
            "declared_edge_count": len(expected_ids),
            "runtime_edge_count": len(observed_ids),
            "identifier_contract_valid": identifier_contract,
            "trajectory_steps": len(steps),
            "applied_steps": len(applied),
            "finite_applied_steps": sum(finite_applied_steps),
            "packing_evidence_steps": sum(packing_evidence_steps),
            "execution_contract_valid": execution_contract,
        },
        "declared_edge_ids": expected_ids,
        "runtime_edge_ids": observed_ids,
        "declared_source_interface_ids": expected_source_ids,
        "runtime_source_interface_ids": observed_source_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit graph-interface guidance runtime execution."
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_graph_interface_guidance(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Graph interface guidance audit: "
        + ("PASSED" if report["passed"] else "FAILED")
    )
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
