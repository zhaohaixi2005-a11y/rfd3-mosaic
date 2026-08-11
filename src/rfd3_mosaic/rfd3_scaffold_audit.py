"""Command-line scaffold geometry audit for RFD3 output structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rfd3_mosaic.rfd3_seed_audit import _derive_structure_path
from rfd3_mosaic.structure import read_structure_atoms
from rfd3_mosaic.validation.scaffold_validity import audit_scaffold_geometry


def _effective_chain_rg_limit(
    *,
    explicit_limit: float | None,
    fixed_geometry_floor: float,
    automatic_margin: float = 2.0,
) -> float:
    """Resolve the compactness gate without invalidating fixed geometry.

    An explicit user limit remains authoritative.  The automatic policy only
    raises the historical 25 A default far enough to accommodate geometry
    that the hard projector makes immutable.  Every input is checked here so
    a malformed audit configuration cannot silently disable compactness.
    """

    values = {
        "fixed_geometry_floor": fixed_geometry_floor,
        "automatic_margin": automatic_margin,
    }
    if explicit_limit is not None:
        values["explicit_limit"] = explicit_limit
    for label, value in values.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        if value < 0.0:
            raise ValueError(f"{label} must be non-negative")
    if explicit_limit is not None:
        return float(explicit_limit)
    return max(25.0, float(fixed_geometry_floor + automatic_margin))


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


def _fixed_geometry_chain_rg_floor(input_path: Path) -> float:
    """Return the largest CA Rg already forced by the compiled fixed target.

    A universal 25 A chain-Rg gate is invalid when an immutable multi-fragment
    seed already spans a larger distance.  Build the exact RFD3 runtime input
    and measure only fixed CA atoms, grouped by physical output chain.  This
    is a lower bound on the generated chain size, not a quality relaxation.
    """

    import numpy as np
    from rfd3.inference.input_parsing import (
        DesignInputSpecification,
        ensure_input_is_abspath,
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(
            "RFD3 input must contain exactly one example for compactness "
            "calibration"
        )
    raw_spec = next(iter(payload.values()))
    raw_spec = ensure_input_is_abspath(dict(raw_spec), input_path)
    atom_array = DesignInputSpecification.safe_init(**raw_spec).build()
    fixed = np.asarray(
        atom_array.is_motif_atom_with_fixed_coord,
        dtype=bool,
    )
    ca = np.asarray(atom_array.atom_name) == "CA"
    chain_ids = np.asarray(atom_array.chain_id)
    radii = []
    for chain_id in np.unique(chain_ids[fixed & ca]):
        coordinates = np.asarray(
            atom_array.coord[fixed & ca & (chain_ids == chain_id)],
            dtype=float,
        )
        if not len(coordinates):
            continue
        center = coordinates.mean(axis=0)
        radii.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(np.square(coordinates - center), axis=-1)
                    )
                )
            )
        )
    return max(radii, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--rfd3-input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-chain-ca-rg", type=float)
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
    fixed_geometry_chain_rg_floor = 0.0
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
        fixed_geometry_chain_rg_floor = _fixed_geometry_chain_rg_floor(
            arguments.rfd3_input.resolve()
        )
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
    effective_max_chain_ca_rg = _effective_chain_rg_limit(
        explicit_limit=arguments.max_chain_ca_rg,
        fixed_geometry_floor=fixed_geometry_chain_rg_floor,
    )
    report = audit_scaffold_geometry(
        read_structure_atoms(structure),
        max_chain_ca_rg=effective_max_chain_ca_rg,
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
    report["compactness_calibration"] = {
        "mode": (
            "explicit"
            if arguments.max_chain_ca_rg is not None
            else "fixed_geometry_lower_bound"
        ),
        "fixed_geometry_chain_ca_rg_floor": (
            fixed_geometry_chain_rg_floor
        ),
        "effective_max_chain_ca_rg": effective_max_chain_ca_rg,
        "automatic_margin": (
            None if arguments.max_chain_ca_rg is not None else 2.0
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
    print(
        "allowed chain CA Rg: "
        f"{effective_max_chain_ca_rg:.3f} A "
        f"(fixed-target floor {fixed_geometry_chain_rg_floor:.3f} A)"
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
        if summary.get("assembly_morphology_available"):
            pore = summary.get("assembly_central_pore_diameter")
            outer = summary.get("assembly_outer_radial_diameter")
            spherical_outer = summary.get(
                "assembly_spherical_outer_diameter"
            )
            if pore is not None and outer is not None:
                print(f"central CA pore:    {pore:.3f} A diameter")
                print(f"outer CA radial:   {outer:.3f} A diameter")
            elif spherical_outer is not None:
                spherical_inner = summary.get(
                    "assembly_spherical_inner_diameter"
                )
                if spherical_inner is not None:
                    print(
                        "central CA sphere: "
                        f"{spherical_inner:.3f} A inner diameter"
                    )
                print(
                    "outer CA sphere:   "
                    f"{spherical_outer:.3f} A diameter "
                    "(no unique principal axis)"
                )
    print(f"report: {output}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
