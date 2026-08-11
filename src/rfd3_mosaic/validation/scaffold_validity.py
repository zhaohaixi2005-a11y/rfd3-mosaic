"""Model-independent geometry checks for generated protein scaffolds."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from rfd3_mosaic.structure import AtomRecord
from rfd3_mosaic.validation.assembly_morphology import (
    audit_assembly_morphology,
)


def audit_scaffold_geometry(
    atoms: tuple[AtomRecord, ...],
    *,
    min_cn_distance: float = 1.0,
    max_cn_distance: float = 2.0,
    max_ca_step: float = 4.5,
    max_chain_ca_rg: float = 25.0,
    ca_clash_distance: float = 3.0,
    expected_symmetry_multiplicity: int | None = None,
    expected_symmetry_transforms: tuple[np.ndarray, ...] | None = None,
    max_chain_distance_matrix_rmsd: float = 0.01,
    max_chain_distance_matrix_error: float = 0.03,
    max_symmetry_coordinate_rmsd: float = 0.01,
    max_symmetry_coordinate_error: float = 0.03,
) -> dict[str, Any]:
    """Audit continuity, compactness, clashes, and optional orbit closure."""

    residues: dict[
        tuple[str, int, str], dict[str, np.ndarray]
    ] = defaultdict(dict)
    for atom in atoms:
        if atom.record_type != "ATOM":
            continue
        name = atom.atom_name.upper()
        if name in {"N", "CA", "C"}:
            residues[atom.residue_id][name] = np.asarray(
                atom.coordinate, dtype=float
            )

    by_chain: dict[str, list[tuple[tuple[str, int, str], dict[str, np.ndarray]]]] = (
        defaultdict(list)
    )
    for residue_id, backbone in residues.items():
        by_chain[residue_id[0]].append((residue_id, backbone))
    for chain_residues in by_chain.values():
        chain_residues.sort(key=lambda item: (item[0][1], item[0][2]))

    chains: list[dict[str, Any]] = []
    ca_coordinates_by_chain: dict[str, np.ndarray] = {}
    ca_records: list[tuple[str, int, np.ndarray]] = []
    for chain_id, chain_residues in sorted(by_chain.items()):
        cn_distances: list[float] = []
        ca_steps: list[float] = []
        breaks: list[dict[str, Any]] = []
        ca_coordinates: list[np.ndarray] = []
        for index, (residue_id, backbone) in enumerate(chain_residues):
            if "CA" in backbone:
                ca_coordinates.append(backbone["CA"])
                ca_records.append((chain_id, index, backbone["CA"]))
            if index == 0:
                continue
            previous_id, previous = chain_residues[index - 1]
            residue_numbers_are_contiguous = (
                residue_id[1] == previous_id[1] + 1
                and not previous_id[2]
                and not residue_id[2]
            )
            cn = (
                float(np.linalg.norm(previous["C"] - backbone["N"]))
                if "C" in previous and "N" in backbone
                else None
            )
            ca = (
                float(np.linalg.norm(previous["CA"] - backbone["CA"]))
                if "CA" in previous and "CA" in backbone
                else None
            )
            if cn is not None:
                cn_distances.append(cn)
            if ca is not None:
                ca_steps.append(ca)
            failed = (
                not residue_numbers_are_contiguous
                or cn is None
                or ca is None
                or not min_cn_distance <= cn <= max_cn_distance
                or ca > max_ca_step
            )
            if failed:
                breaks.append(
                    {
                        "previous_residue": previous_id[1],
                        "next_residue": residue_id[1],
                        "residue_numbers_are_contiguous": (
                            residue_numbers_are_contiguous
                        ),
                        "cn_distance": cn,
                        "ca_distance": ca,
                    }
                )
        ca_array = np.asarray(ca_coordinates, dtype=float)
        ca_coordinates_by_chain[chain_id] = ca_array
        ca_rg = (
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (ca_array - ca_array.mean(axis=0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if len(ca_array)
            else float("inf")
        )
        chains.append(
            {
                "chain_id": chain_id,
                "residue_count": len(chain_residues),
                "ca_count": len(ca_coordinates),
                "ca_radius_of_gyration": ca_rg,
                "maximum_cn_distance": max(cn_distances, default=None),
                "maximum_ca_step": max(ca_steps, default=None),
                "chain_breaks": breaks,
                "passed_continuity": not breaks,
                "passed_compactness": ca_rg <= max_chain_ca_rg,
            }
        )

    ca_clashes: list[dict[str, Any]] = []
    for left_index, (left_chain, left_seq, left_coord) in enumerate(ca_records):
        for right_chain, right_seq, right_coord in ca_records[left_index + 1 :]:
            if left_chain == right_chain and abs(left_seq - right_seq) <= 2:
                continue
            distance = float(np.linalg.norm(left_coord - right_coord))
            if distance < ca_clash_distance:
                ca_clashes.append(
                    {
                        "left_chain": left_chain,
                        "left_index": left_seq,
                        "right_chain": right_chain,
                        "right_index": right_seq,
                        "distance": distance,
                    }
                )

    passed_continuity = bool(chains) and all(
        chain["passed_continuity"] for chain in chains
    )
    passed_compactness = bool(chains) and all(
        chain["passed_compactness"] for chain in chains
    )
    passed_clashes = not ca_clashes
    copy_internal_comparisons: list[dict[str, Any]] = []
    transform_comparisons: list[dict[str, Any]] = []
    symmetry_failures: list[str] = []
    validated_symmetry_transforms: tuple[np.ndarray, ...] | None = None
    if expected_symmetry_multiplicity is not None:
        if expected_symmetry_multiplicity < 2:
            raise ValueError(
                "expected_symmetry_multiplicity must be at least two"
            )
        if expected_symmetry_transforms is None:
            symmetry_failures.append(
                "expected symmetry transforms were not provided"
            )
        elif (
            len(expected_symmetry_transforms)
            != expected_symmetry_multiplicity
        ):
            symmetry_failures.append(
                "symmetry transform count does not match expected "
                f"multiplicity {expected_symmetry_multiplicity}"
            )
        else:
            normalized_transforms: list[np.ndarray] = []
            homogeneous_row = np.asarray([0.0, 0.0, 0.0, 1.0])
            for transform_index, raw_transform in enumerate(
                expected_symmetry_transforms
            ):
                transform = np.asarray(raw_transform, dtype=float)
                if (
                    transform.shape != (4, 4)
                    or not np.isfinite(transform).all()
                ):
                    symmetry_failures.append(
                        "expected symmetry transform "
                        f"{transform_index} must be a finite 4x4 matrix"
                    )
                    continue
                rotation = transform[:3, :3]
                orthogonality_error = float(
                    np.max(
                        np.abs(
                            rotation.T @ rotation - np.eye(3)
                        )
                    )
                )
                determinant = float(np.linalg.det(rotation))
                if not np.allclose(
                    transform[3],
                    homogeneous_row,
                    atol=1e-6,
                ):
                    symmetry_failures.append(
                        "expected symmetry transform "
                        f"{transform_index} has an invalid homogeneous row"
                    )
                    continue
                if (
                    orthogonality_error > 1e-5
                    or abs(determinant - 1.0) > 1e-5
                ):
                    symmetry_failures.append(
                        "expected symmetry transform "
                        f"{transform_index} is not a proper rigid rotation"
                    )
                    continue
                normalized_transforms.append(transform)
            if (
                len(normalized_transforms)
                == expected_symmetry_multiplicity
            ):
                validated_symmetry_transforms = tuple(
                    normalized_transforms
                )
        ordered_chain_ids = sorted(ca_coordinates_by_chain)
        if (
            len(ordered_chain_ids) == 0
            or len(ordered_chain_ids)
            % expected_symmetry_multiplicity
            != 0
        ):
            symmetry_failures.append(
                "chain count is not divisible by expected symmetry "
                f"multiplicity {expected_symmetry_multiplicity}"
            )
        else:
            asu_chain_count = (
                len(ordered_chain_ids) // expected_symmetry_multiplicity
            )
            for asu_chain_index in range(asu_chain_count):
                orbit_chain_ids = [
                    ordered_chain_ids[
                        copy_index * asu_chain_count + asu_chain_index
                    ]
                    for copy_index in range(
                        expected_symmetry_multiplicity
                    )
                ]
                reference_id = orbit_chain_ids[0]
                reference_coordinates = ca_coordinates_by_chain[
                    reference_id
                ]
                if not len(reference_coordinates):
                    symmetry_failures.append(
                        f"reference chain {reference_id} has no CA atoms"
                    )
                    continue
                reference_distances = np.linalg.norm(
                    reference_coordinates[:, None, :]
                    - reference_coordinates[None, :, :],
                    axis=-1,
                )
                for copy_index, observed_id in enumerate(
                    orbit_chain_ids[1:],
                    start=1,
                ):
                    observed_coordinates = ca_coordinates_by_chain[
                        observed_id
                    ]
                    if (
                        observed_coordinates.shape
                        != reference_coordinates.shape
                    ):
                        symmetry_failures.append(
                            f"chains {reference_id} and {observed_id} "
                            "have different CA counts"
                        )
                        continue
                    observed_distances = np.linalg.norm(
                        observed_coordinates[:, None, :]
                        - observed_coordinates[None, :, :],
                        axis=-1,
                    )
                    difference = (
                        observed_distances - reference_distances
                    )
                    rmsd = float(
                        np.sqrt(np.mean(np.square(difference)))
                    )
                    maximum = float(np.max(np.abs(difference)))
                    passed = (
                        rmsd <= max_chain_distance_matrix_rmsd
                        and maximum <= max_chain_distance_matrix_error
                    )
                    copy_internal_comparisons.append(
                        {
                            "reference_chain": reference_id,
                            "observed_chain": observed_id,
                            "ca_count": len(reference_coordinates),
                            "distance_matrix_rmsd": rmsd,
                            "maximum_distance_matrix_error": maximum,
                            "passed": passed,
                        }
                    )
                    if validated_symmetry_transforms is None:
                        continue
                    reference_transform = np.asarray(
                        validated_symmetry_transforms[0],
                        dtype=float,
                    )
                    observed_transform = np.asarray(
                        validated_symmetry_transforms[copy_index],
                        dtype=float,
                    )
                    relative_transform = (
                        observed_transform
                        @ np.linalg.inv(reference_transform)
                    )
                    expected_coordinates = (
                        reference_coordinates
                        @ relative_transform[:3, :3].T
                        + relative_transform[:3, 3]
                    )
                    coordinate_errors = np.linalg.norm(
                        observed_coordinates - expected_coordinates,
                        axis=-1,
                    )
                    coordinate_rmsd = float(
                        np.sqrt(np.mean(np.square(coordinate_errors)))
                    )
                    coordinate_maximum = float(
                        np.max(coordinate_errors)
                    )
                    transform_passed = (
                        coordinate_rmsd <= max_symmetry_coordinate_rmsd
                        and coordinate_maximum
                        <= max_symmetry_coordinate_error
                    )
                    transform_comparisons.append(
                        {
                            "reference_chain": reference_id,
                            "observed_chain": observed_id,
                            "runtime_transform_id": copy_index,
                            "ca_count": len(reference_coordinates),
                            "coordinate_rmsd": coordinate_rmsd,
                            "maximum_coordinate_error": coordinate_maximum,
                            "passed": transform_passed,
                        }
                    )
                    if not transform_passed:
                        symmetry_failures.append(
                            f"chains {reference_id}/{observed_id} do not "
                            "satisfy the declared symmetry transform"
                        )
    passed_symmetry = (
        True
        if expected_symmetry_multiplicity is None
        else not symmetry_failures
        and bool(transform_comparisons)
    )
    morphology: dict[str, Any] = {
        "enabled": False,
        "passed": True,
        "measurement_only": True,
        "diagnostic_only": True,
        "summary": None,
        "axes": [],
        "failures": [],
    }
    if validated_symmetry_transforms is not None:
        morphology = audit_assembly_morphology(
            atoms,
            symmetry_transforms=validated_symmetry_transforms,
        )
        morphology["enabled"] = True
    morphology_summary = morphology.get("summary") or {}
    return {
        "audit": "rfd3_mosaic.scaffold_geometry",
        # Morphology is an additive diagnostics block; keep the existing
        # scaffold-report schema compatible for old run consumers.
        "schema_version": 2,
        "passed": (
            passed_continuity
            and passed_compactness
            and passed_clashes
            and passed_symmetry
        ),
        "summary": {
            "chain_count": len(chains),
            "chain_break_count": sum(
                len(chain["chain_breaks"]) for chain in chains
            ),
            "maximum_chain_ca_radius_of_gyration": max(
                (
                    chain["ca_radius_of_gyration"]
                    for chain in chains
                ),
                default=float("inf"),
            ),
            "ca_clash_count": len(ca_clashes),
            "passed_continuity": passed_continuity,
            "passed_compactness": passed_compactness,
            "passed_clashes": passed_clashes,
            "passed_symmetry": passed_symmetry,
            "maximum_copy_internal_distance_matrix_rmsd": max(
                (
                    comparison["distance_matrix_rmsd"]
                    for comparison in copy_internal_comparisons
                ),
                default=0.0,
            ),
            "maximum_copy_internal_distance_matrix_error": max(
                (
                    comparison["maximum_distance_matrix_error"]
                    for comparison in copy_internal_comparisons
                ),
                default=0.0,
            ),
            "maximum_symmetry_coordinate_rmsd": max(
                (
                    comparison["coordinate_rmsd"]
                    for comparison in transform_comparisons
                ),
                default=0.0,
            ),
            "maximum_symmetry_coordinate_error": max(
                (
                    comparison["maximum_coordinate_error"]
                    for comparison in transform_comparisons
                ),
                default=0.0,
            ),
            "assembly_morphology_available": bool(
                morphology.get("enabled")
                and morphology.get("summary") is not None
            ),
            "assembly_morphology_measurement_only": bool(
                morphology.get("measurement_only", True)
            ),
            "assembly_central_pore_diameter": morphology_summary.get(
                "central_pore_diameter"
            ),
            "assembly_central_pore_diameter_p05": morphology_summary.get(
                "central_pore_diameter_p05"
            ),
            "assembly_outer_radial_diameter": morphology_summary.get(
                "outer_radial_diameter"
            ),
            "assembly_outer_radial_diameter_p95": morphology_summary.get(
                "outer_radial_diameter_p95"
            ),
            "assembly_spherical_inner_diameter": morphology_summary.get(
                "spherical_inner_diameter"
            ),
            "assembly_spherical_outer_diameter": morphology_summary.get(
                "spherical_outer_diameter"
            ),
            "assembly_principal_axis_unique": morphology_summary.get(
                "principal_axis_unique"
            ),
        },
        "thresholds": {
            "min_cn_distance": min_cn_distance,
            "max_cn_distance": max_cn_distance,
            "max_ca_step": max_ca_step,
            "max_chain_ca_radius_of_gyration": max_chain_ca_rg,
            "ca_clash_distance": ca_clash_distance,
            "expected_symmetry_multiplicity": (
                expected_symmetry_multiplicity
            ),
            "max_chain_distance_matrix_rmsd": (
                max_chain_distance_matrix_rmsd
            ),
            "max_chain_distance_matrix_error": (
                max_chain_distance_matrix_error
            ),
            "max_symmetry_coordinate_rmsd": (
                max_symmetry_coordinate_rmsd
            ),
            "max_symmetry_coordinate_error": (
                max_symmetry_coordinate_error
            ),
        },
        "chains": chains,
        "ca_clashes": ca_clashes,
        "symmetry": {
            "enabled": expected_symmetry_multiplicity is not None,
            "passed": passed_symmetry,
            "method": "declared_transform_coordinate_residual",
            "transform_comparisons": transform_comparisons,
            "copy_internal_congruence": {
                "hard_gate": False,
                "comparisons": copy_internal_comparisons,
            },
            "failures": symmetry_failures,
        },
        "morphology": morphology,
    }
