"""Compatibility CLI for the historical central-motif audit name."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rfd3_mosaic.rfd3_constraint_orbit_audit import audit_constraint_orbit


def audit_central_motif(
    *,
    probe_input: str | Path,
    result_json: str | Path,
    result_structure: str | Path | None = None,
    max_joint_rmsd: float = 0.5,
    max_joint_coordinate_error: float = 0.001,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    """Preserve the original Python API while using the generic engine."""

    # Kept only so historical callers do not break.  Complete fixed geometry
    # is intentionally invariant to one common laboratory-frame SE(3)
    # transform, so absolute coordinate offsets are never a pass/fail gate.
    del max_joint_coordinate_error
    report = audit_constraint_orbit(
        compiled_input=probe_input,
        result_json=result_json,
        result_structure=result_structure,
        max_joint_rmsd=max_joint_rmsd,
        min_atom_completeness=min_atom_completeness,
    )
    report["audit"] = "rfd3_mosaic.central_fixed_motif_orbit"
    inputs = report["inputs"]
    inputs["probe_input"] = inputs.pop("compiled_input")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit preservation of one symmetric central motif orbit."
    )
    parser.add_argument("--probe-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-joint-rmsd", type=float, default=0.5)
    parser.add_argument(
        "--max-joint-coordinate-error",
        type=float,
        default=0.001,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-atom-completeness", type=float, default=0.99)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_central_motif(
        probe_input=arguments.probe_input,
        result_json=arguments.result_json,
        result_structure=arguments.result_structure,
        max_joint_rmsd=arguments.max_joint_rmsd,
        max_joint_coordinate_error=arguments.max_joint_coordinate_error,
        min_atom_completeness=arguments.min_atom_completeness,
    )
    arguments.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"Central motif audit: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"joint orbit RMSD: {summary['joint_orbit_rmsd']:.6f} A")
    print(
        "orbit distance-matrix RMSD: "
        f"{summary['orbit_distance_matrix_rmsd']:.6f} A"
    )
    print(f"report: {arguments.output.resolve()}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["audit_central_motif"]
