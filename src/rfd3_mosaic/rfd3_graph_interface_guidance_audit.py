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
    unique_expected_source_ids = list(dict.fromkeys(expected_source_ids))
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
    locked_steps = [step for step in steps if step.get("patch_locked") is True]
    locked_patch_assignments = [
        step.get("patch_assignments") for step in locked_steps
    ]
    patch_identity_contract = bool(
        diagnostics_schema_version < 7
        or (
            locked_patch_assignments
            and isinstance(locked_patch_assignments[0], dict)
            and bool(locked_patch_assignments[0])
            and all(
                isinstance(assignments, dict)
                and assignments == locked_patch_assignments[0]
                for assignments in locked_patch_assignments
            )
        )
    )
    applied = [step for step in steps if bool(step.get("applied"))]
    adaptive_phase_contract = bool(
        diagnostics_schema_version < 8
        or (
            applied
            and all(
                step.get("adaptive_phase")
                in {"capture", "expand", "polish"}
                and _finite(step.get("scheduled_target_ca_distance"))
                and _finite(step.get("time_scheduled_target_ca_distance"))
                for step in applied
            )
        )
    )
    capacity_preflight = diagnostics.get("capacity_preflight", [])
    capacity_preflight_contract = bool(
        diagnostics_schema_version < 8
        or (
            isinstance(capacity_preflight, list)
            and len(capacity_preflight) == len(expected_ids)
            and {
                str(record.get("edge_id"))
                for record in capacity_preflight
                if isinstance(record, dict)
            }
            == set(expected_ids)
            and all(
                int(record.get("available_residues_left", 0))
                >= int(record.get("requested_residues_per_side", 1))
                and int(record.get("available_residues_right", 0))
                >= int(record.get("requested_residues_per_side", 1))
                and int(
                    record.get("available_contiguous_residues_left", 0)
                )
                >= int(
                    record.get(
                        "requested_contiguous_residues_per_side", 1
                    )
                )
                and int(
                    record.get("available_contiguous_residues_right", 0)
                )
                >= int(
                    record.get(
                        "requested_contiguous_residues_per_side", 1
                    )
                )
                for record in capacity_preflight
            )
        )
    )
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
                and (
                    diagnostics_schema_version < 4
                    or (
                        all(
                            _finite(step.get(key))
                            for key in (
                                "orientation",
                                "shape",
                                "backbone",
                                "interface_balance",
                            )
                        )
                        and all(
                            len(step.get(key, [])) == len(expected_ids)
                            and all(_finite(value) for value in step[key])
                            for key in (
                                "per_edge_orientation",
                                "per_edge_shape",
                                "per_edge_backbone",
                            )
                        )
                        and len(step.get("per_source_total", []))
                        == len(unique_expected_source_ids)
                        and all(
                            _finite(value)
                            for value in step.get("per_source_total", [])
                        )
                    )
                )
            )
        )
        finite_applied_steps.append(base_evidence and packing_evidence)
        packing_evidence_steps.append(packing_evidence)
    runtime_active = diagnostics.get("runtime_active") is True
    final_proxy = diagnostics.get("final_proxy")
    final_proxy_contract = bool(
        diagnostics_schema_version < 5
        or (
            isinstance(
                diagnostics.get("final_proxy_targets_satisfied"), bool
            )
            and isinstance(final_proxy, dict)
            and all(
                isinstance(final_proxy.get(key), list)
                and len(final_proxy[key]) == len(expected_ids)
                for key in (
                    "covered_left_residues",
                    "covered_right_residues",
                    "target_residues_per_side",
                    "contiguous_left_residues",
                    "contiguous_right_residues",
                    "target_contiguous_residues_per_side",
                )
            )
            and isinstance(diagnostics.get("final_polish_steps"), int)
            and int(diagnostics["final_polish_steps"]) >= 0
            and (
                diagnostics_schema_version < 6
                or (
                    all(
                        _finite(final_proxy.get(key))
                        for key in (
                            "energy",
                            "attraction",
                            "coverage",
                            "continuity",
                            "orientation",
                            "shape",
                            "backbone",
                            "interface_balance",
                            "clash",
                            "distance",
                        )
                    )
                    and all(
                        isinstance(final_proxy.get(key), list)
                        and len(final_proxy[key]) == len(expected_ids)
                        and all(_finite(value) for value in final_proxy[key])
                        for key in (
                            "minimum_distances",
                            "mean_selected_distances",
                            "per_edge_orientation",
                            "per_edge_shape",
                            "per_edge_backbone",
                            "per_edge_total",
                        )
                    )
                    and isinstance(
                        final_proxy.get("per_source_total"), list
                    )
                    and len(final_proxy["per_source_total"])
                    == len(unique_expected_source_ids)
                    and all(
                        _finite(value)
                        for value in final_proxy["per_source_total"]
                    )
                )
            )
        )
    )
    final_proxy_targets_satisfied = bool(
        diagnostics_schema_version < 5
        or diagnostics.get("final_proxy_targets_satisfied") is True
    )
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
        and final_proxy_contract
    )
    # Schema v5 made the final coverage/continuity proxy an explicit
    # generated-interface objective.  Merely proving that the controller ran
    # is not enough: a trajectory whose own final objective is unsatisfied
    # must not be reported as a successful guidance result.  Legacy
    # diagnostics predate this result contract and retain their execution-only
    # interpretation.
    passed = bool(
        runtime_active
        and identifier_contract
        and execution_contract
        and patch_identity_contract
        and adaptive_phase_contract
        and capacity_preflight_contract
        and final_proxy_targets_satisfied
    )
    final_step = applied[-1] if applied else {}
    final_metric_source = (
        final_proxy
        if diagnostics_schema_version >= 6 and isinstance(final_proxy, dict)
        else final_step
    )
    final_packing_metrics = {
        key: final_metric_source.get(key)
        for key in (
            "energy",
            "attraction",
            "coverage",
            "continuity",
            "orientation",
            "shape",
            "backbone",
            "junction",
            "interface_balance",
            "patch_exclusivity",
            "clash",
            "global_safety_clash",
            "minimum_global_safety_distance",
            "maximum_token_step",
            "mean_token_step",
        )
        if _finite(final_metric_source.get(key))
    }
    final_minimum_distances = [
        value
        for value in final_metric_source.get("minimum_distances", [])
        if _finite(value)
    ]
    final_source_totals = [
        value
        for value in final_metric_source.get("per_source_total", [])
        if _finite(value)
    ]
    if final_minimum_distances:
        final_packing_metrics["minimum_edge_distance"] = min(
            final_minimum_distances
        )
    if final_source_totals:
        final_packing_metrics["maximum_source_objective"] = max(
            final_source_totals
        )
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
            "patch_identity_contract_valid": patch_identity_contract,
            "adaptive_phase_contract_valid": adaptive_phase_contract,
            "capacity_preflight_contract_valid": (
                capacity_preflight_contract
            ),
            "adaptive_phase_counts": {
                phase: sum(
                    step.get("adaptive_phase") == phase for step in applied
                )
                for phase in ("capture", "expand", "polish")
            },
            "locked_patch_steps": len(locked_steps),
            "trajectory_steps": len(steps),
            "applied_steps": len(applied),
            "finite_applied_steps": sum(finite_applied_steps),
            "packing_evidence_steps": sum(packing_evidence_steps),
            "final_proxy_contract_valid": final_proxy_contract,
            "final_proxy_targets_satisfied": (
                diagnostics.get("final_proxy_targets_satisfied")
            ),
            "final_result_contract_valid": (
                final_proxy_targets_satisfied
            ),
            "final_polish_steps": diagnostics.get("final_polish_steps", 0),
            "final_metrics_source": (
                "post_finalize_state"
                if diagnostics_schema_version >= 6
                else "last_pre_update_guidance_step"
            ),
            "execution_contract_valid": execution_contract,
            "final_packing_metrics": final_packing_metrics,
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
