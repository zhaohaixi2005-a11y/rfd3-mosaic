"""Audit execution and final metrics of intra/inter scaffold guidance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def audit_scaffold_core_guidance(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
) -> dict[str, Any]:
    input_path = Path(compiled_input).resolve()
    result_path = Path(result_json).resolve()
    compiled = _object(input_path)
    if len(compiled) != 1:
        raise ValueError("Compiled input must contain exactly one example")
    example = next(iter(compiled.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled example must be an object")
    extra = example.get("extra") or {}
    plan = extra.get("scaffold_core_guidance")
    if not isinstance(plan, dict):
        # Compatibility with inputs frozen before intra/inter guidance was
        # separated from automatic generated-interface packing.
        plan = extra.get("automatic_symmetric_scaffold_packing")
    if not isinstance(plan, dict):
        raise ValueError("Compiled input has no scaffold core guidance plan")
    expected_intra = float(plan.get("intra_chain_weight", 0.0))
    expected_inter = float(plan.get("inter_chain_weight", 1.0))
    expected_excess_penalty = float(plan.get("inter_chain_excess_penalty", 0.0))
    quality_contract = plan.get("quality_contract")
    if quality_contract is None:
        quality_contract = {"required": False}
    if not isinstance(quality_contract, dict):
        raise ValueError("scaffold core quality_contract must be an object")

    diagnostics = _object(result_path).get("scaffold_core_guidance_diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    config = diagnostics.get("config")
    steps = diagnostics.get("steps")
    final_metrics = diagnostics.get("final_metrics")
    if not isinstance(config, dict):
        config = {}
    if not isinstance(steps, list):
        steps = []
    if not isinstance(final_metrics, dict):
        final_metrics = {}

    config_contract = bool(
        _finite(config.get("intra_chain_weight"))
        and _finite(config.get("inter_chain_weight"))
        and math.isclose(
            float(config["intra_chain_weight"]), expected_intra, abs_tol=1e-8
        )
        and math.isclose(
            float(config["inter_chain_weight"]), expected_inter, abs_tol=1e-8
        )
        and _finite(config.get("inter_chain_excess_penalty", 0.0))
        and math.isclose(
            float(config.get("inter_chain_excess_penalty", 0.0)),
            expected_excess_penalty,
            abs_tol=1e-8,
        )
    )
    metric_names = (
        "total",
        "long_range_contacts",
        "normalized_rg",
        "tertiary_support",
        "inter_chain_excess",
        "clash",
        "continuity",
        "mean_normalized_rg",
        "mean_tertiary_support_fraction",
        "generated_inter_chain_contact_pairs",
        "generated_inter_chain_contact_coverage",
        "minimum_generated_inter_chain_distance",
    )
    final_metric_contract = all(
        _finite(final_metrics.get(name)) for name in metric_names
    )
    applied = [step for step in steps if bool(step.get("applied"))]
    applied_count = int(diagnostics.get("applied_steps", -1))
    step_contract = bool(
        steps
        and applied
        and applied_count == len(applied)
        and all(
            isinstance(step.get("initial"), dict)
            and isinstance(step.get("final"), dict)
            and _finite(step["initial"].get("total"))
            and _finite(step["final"].get("total"))
            and float(step["final"]["total"]) <= float(step["initial"]["total"]) + 1e-7
            for step in applied
        )
    )
    # Geometry safety remains mandatory.  Scientific-quality thresholds are
    # a separate, explicitly enabled contract so older designs do not acquire
    # a new calibrated gate merely by being replayed with newer software.
    safety_contract = bool(
        final_metric_contract
        and float(final_metrics["clash"]) <= 1e-4
        and float(final_metrics["continuity"]) <= 0.05
    )
    quality_required = quality_contract.get("required") is True
    quality_thresholds_valid = bool(
        _finite(quality_contract.get("maximum_mean_normalized_rg"))
        and _finite(quality_contract.get("minimum_mean_tertiary_support_fraction"))
        and _finite(quality_contract.get("maximum_long_range_contact_deficit"))
    )
    scientific_quality_satisfied = bool(
        final_metric_contract
        and quality_thresholds_valid
        and float(final_metrics["mean_normalized_rg"])
        <= float(quality_contract["maximum_mean_normalized_rg"])
        and float(final_metrics["mean_tertiary_support_fraction"])
        >= float(quality_contract["minimum_mean_tertiary_support_fraction"])
        and float(final_metrics["long_range_contacts"])
        <= float(quality_contract["maximum_long_range_contact_deficit"])
    )
    quality_gate_satisfied = bool(not quality_required or scientific_quality_satisfied)
    passed = bool(
        diagnostics.get("runtime_active") is True
        and int(diagnostics.get("chain_count", 0)) >= 1
        and config_contract
        and step_contract
        and final_metric_contract
        and safety_contract
        and quality_gate_satisfied
    )
    return {
        "audit": "rfd3_mosaic.scaffold_core_guidance",
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "compiled_input": str(input_path),
            "result_json": str(result_path),
        },
        "summary": {
            "runtime_active": diagnostics.get("runtime_active") is True,
            "chain_count": diagnostics.get("chain_count", 0),
            "trajectory_steps": len(steps),
            "applied_steps": len(applied),
            "config_contract_valid": config_contract,
            "step_contract_valid": step_contract,
            "final_metric_contract_valid": final_metric_contract,
            "safety_contract_valid": safety_contract,
            "quality_required": quality_required,
            "quality_thresholds_valid": quality_thresholds_valid,
            "scientific_quality_satisfied": (scientific_quality_satisfied),
            "quality_gate_satisfied": quality_gate_satisfied,
            "quality_contract": quality_contract,
            "final_metrics": final_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit intra/inter scaffold-core guidance."
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_scaffold_core_guidance(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Scaffold core guidance audit: " + ("PASSED" if report["passed"] else "FAILED")
    )
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
