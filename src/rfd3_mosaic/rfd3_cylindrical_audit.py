"""Audit execution of compiler-declared cylindrical hard constraints."""

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


def audit_cylindrical_coordinates(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    input_path = Path(compiled_input).resolve()
    result_path = Path(result_json).resolve()
    example = _single_example(input_path)
    declarations = (example.get("extra") or {}).get(
        "cylindrical_constraints"
    ) or []
    if not declarations:
        raise ValueError(
            "Cylindrical audit requires compiler-declared constraints"
        )
    result = _load(result_path)
    diagnostics = result.get("constraint_runtime_diagnostics") or {}
    phase_counts = diagnostics.get("phase_counts") or {}
    final_error = diagnostics.get("final_cylindrical_maximum_error")
    finite_error = (
        isinstance(final_error, (int, float))
        and not isinstance(final_error, bool)
        and math.isfinite(float(final_error))
    )
    runtime_active = bool(
        diagnostics.get("cylindrical_projector_active")
        and diagnostics.get("state") == "finalized"
        and int(phase_counts.get("initialize", 0)) == 1
        and int(phase_counts.get("finalize", 0)) == 1
    )
    identifier_contract_valid = (
        len({str(item["constraint_id"]) for item in declarations})
        == len(declarations)
        and all(item.get("atom_keys") for item in declarations)
        and all(item.get("keep") for item in declarations)
    )
    passed = bool(
        runtime_active
        and identifier_contract_valid
        and finite_error
        and float(final_error) <= tolerance
    )
    return {
        "audit": "rfd3_mosaic.cylindrical_coordinates",
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "compiled_input": str(input_path),
            "result_json": str(result_path),
        },
        "thresholds": {"maximum_coordinate_error": tolerance},
        "summary": {
            "declared_constraint_count": len(declarations),
            "constraint_ids": [
                str(item["constraint_id"]) for item in declarations
            ],
            "runtime_active": runtime_active,
            "identifier_contract_valid": identifier_contract_valid,
            "final_cylindrical_maximum_error": final_error,
            "phase_counts": phase_counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cylindrical hard-coordinate execution."
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_cylindrical_coordinates(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
        tolerance=arguments.tolerance,
    )
    arguments.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Cylindrical coordinate audit: "
        + ("PASSED" if report["passed"] else "FAILED")
    )
    print(f"report: {arguments.output.resolve()}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["audit_cylindrical_coordinates"]
