"""Command-line scaffold geometry audit for RFD3 output structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfd3_mosaic.rfd3_seed_audit import _derive_structure_path
from rfd3_mosaic.structure import read_structure_atoms
from rfd3_mosaic.validation.scaffold_validity import audit_scaffold_geometry


def _load_declared_symmetry_transforms(
    input_path: Path,
) -> tuple[tuple[object, ...], int]:
    """Load the adapter's prevalidated runtime-ordered transform registry."""

    import numpy as np

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(
            "RFD3 input must contain exactly one example for scaffold audit"
        )
    specification = next(iter(payload.values()))
    extra = specification.get("extra") or {}
    order = list(extra.get("registry_transform_order") or ())
    matrices = extra.get("registry_transform_matrices") or {}
    if not order or set(order) != set(matrices):
        raise ValueError(
            "RFD3 input lacks a complete runtime-ordered transform registry"
        )
    transforms = tuple(
        np.asarray(matrices[transform_id], dtype=float)
        for transform_id in order
    )
    multiplicity = int(extra.get("symmetry_multiplicity", len(order)))
    if len(transforms) != multiplicity:
        raise ValueError(
            "Declared transform count does not match symmetry multiplicity"
        )
    return transforms, multiplicity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--rfd3-input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-chain-ca-rg", type=float, default=25.0)
    parser.add_argument("--expected-symmetry-multiplicity", type=int)
    parser.add_argument(
        "--max-chain-distance-matrix-rmsd",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-chain-distance-matrix-error",
        type=float,
        default=0.03,
    )
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()

    expected_transforms = None
    if arguments.rfd3_input is not None:
        expected_transforms, declared_multiplicity = (
            _load_declared_symmetry_transforms(
                arguments.rfd3_input.resolve()
            )
        )
        if (
            arguments.expected_symmetry_multiplicity is not None
            and arguments.expected_symmetry_multiplicity
            != declared_multiplicity
        ):
            raise ValueError(
                "--expected-symmetry-multiplicity disagrees with "
                "--rfd3-input"
            )
        arguments.expected_symmetry_multiplicity = declared_multiplicity
    elif arguments.expected_symmetry_multiplicity is not None:
        raise ValueError(
            "--rfd3-input is required for transform-aware symmetry audit"
        )

    result_json = arguments.result_json.resolve()
    structure = (
        arguments.result_structure.resolve()
        if arguments.result_structure
        else _derive_structure_path(result_json)
    )
    report = audit_scaffold_geometry(
        read_structure_atoms(structure),
        max_chain_ca_rg=arguments.max_chain_ca_rg,
        expected_symmetry_multiplicity=(
            arguments.expected_symmetry_multiplicity
        ),
        expected_symmetry_transforms=expected_transforms,
        max_chain_distance_matrix_rmsd=(
            arguments.max_chain_distance_matrix_rmsd
        ),
        max_chain_distance_matrix_error=(
            arguments.max_chain_distance_matrix_error
        ),
    )
    report["inputs"] = {
        "result_json": str(result_json),
        "result_structure": str(structure),
        "rfd3_input": (
            str(arguments.rfd3_input.resolve())
            if arguments.rfd3_input is not None
            else None
        ),
    }
    output = arguments.output.resolve()
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"Scaffold audit: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"chains:              {summary['chain_count']}")
    print(f"chain breaks:        {summary['chain_break_count']}")
    print(
        "maximum chain CA Rg: "
        f"{summary['maximum_chain_ca_radius_of_gyration']:.3f} A"
    )
    print(f"CA clashes:          {summary['ca_clash_count']}")
    if arguments.expected_symmetry_multiplicity is not None:
        print(
            "symmetry DM RMSD:   "
            f"{summary['maximum_copy_internal_distance_matrix_rmsd']:.6f} A"
        )
        print(
            "symmetry DM max:    "
            f"{summary['maximum_copy_internal_distance_matrix_error']:.6f} A"
        )
        print(
            "symmetry xyz RMSD:  "
            f"{summary['maximum_symmetry_coordinate_rmsd']:.6f} A"
        )
        print(
            "symmetry xyz max:   "
            f"{summary['maximum_symmetry_coordinate_error']:.6f} A"
        )
    print(f"report: {output}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
