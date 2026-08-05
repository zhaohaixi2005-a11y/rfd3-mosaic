"""Batch geometry and packing screen for extracted RFD3 CIF structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rfd3_mosaic.structure import AtomRecord, read_structure_atoms
from rfd3_mosaic.validation.scaffold_validity import (
    audit_scaffold_geometry,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _job_id_from_path(path: Path) -> str | None:
    prefix = path.name.split("__", 1)[0]
    return prefix if prefix.isdigit() else None


def _ca_coordinates_by_chain(
    atoms: tuple[AtomRecord, ...],
) -> dict[str, np.ndarray]:
    records: dict[str, list[tuple[tuple[int, str], np.ndarray]]] = {}
    for atom in atoms:
        if atom.record_type != "ATOM" or atom.atom_name.upper() != "CA":
            continue
        records.setdefault(atom.chain_id, []).append(
            (
                (atom.residue_number, atom.insertion_code),
                np.asarray(atom.coordinate, dtype=float),
            )
        )
    coordinates: dict[str, np.ndarray] = {}
    for chain_id, chain_records in sorted(records.items()):
        chain_records.sort(key=lambda item: item[0])
        coordinates[chain_id] = np.asarray(
            [coordinate for _, coordinate in chain_records],
            dtype=float,
        )
    return coordinates


def cyclic_ring_descriptors(
    ca_by_chain: dict[str, np.ndarray],
    expected_order: int,
) -> dict[str, Any]:
    """Fit the chain COMs to a ring plane and report scale-free diagnostics."""

    if len(ca_by_chain) != expected_order:
        return {
            "available": False,
            "reason": (
                f"expected {expected_order} chains, found "
                f"{len(ca_by_chain)}"
            ),
        }
    if any(len(coordinates) == 0 for coordinates in ca_by_chain.values()):
        return {
            "available": False,
            "reason": "one or more chains have no CA coordinates",
        }

    chain_ids = sorted(ca_by_chain)
    chain_centers = np.asarray(
        [ca_by_chain[chain_id].mean(axis=0) for chain_id in chain_ids],
        dtype=float,
    )
    center = chain_centers.mean(axis=0)
    centered = chain_centers - center
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    plane_x = right_vectors[0]
    plane_y = right_vectors[1]
    axis = right_vectors[-1]
    projected_x = centered @ plane_x
    projected_y = centered @ plane_y
    axial = centered @ axis
    radii = np.sqrt(np.square(projected_x) + np.square(projected_y))
    mean_radius = float(np.mean(radii))

    angles = np.mod(np.arctan2(projected_y, projected_x), 2.0 * math.pi)
    angle_order = np.argsort(angles)
    sorted_angles = angles[angle_order]
    gaps = np.diff(np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * math.pi)))
    expected_gap = 2.0 * math.pi / expected_order
    gap_errors_degrees = np.degrees(gaps - expected_gap)

    all_ca = np.concatenate(
        [ca_by_chain[chain_id] for chain_id in chain_ids],
        axis=0,
    )
    relative_ca = all_ca - center
    ca_axial = relative_ca @ axis
    ca_radial_vectors = relative_ca - ca_axial[:, None] * axis
    ca_radial = np.linalg.norm(ca_radial_vectors, axis=1)
    maximum_radial = float(np.max(ca_radial))

    return {
        "available": True,
        "center": center.tolist(),
        "axis": axis.tolist(),
        "angular_chain_order": [
            chain_ids[int(index)] for index in angle_order
        ],
        "mean_chain_com_radius": mean_radius,
        "chain_com_radial_cv": (
            float(np.std(radii) / mean_radius)
            if mean_radius > 1e-8
            else None
        ),
        "chain_com_axial_rms": float(
            np.sqrt(np.mean(np.square(axial)))
        ),
        "angular_gap_rms_error_degrees": float(
            np.sqrt(np.mean(np.square(gap_errors_degrees)))
        ),
        "angular_gap_max_error_degrees": float(
            np.max(np.abs(gap_errors_degrees))
        ),
        "minimum_ca_axis_clearance": float(np.min(ca_radial)),
        "maximum_ca_axis_extent": maximum_radial,
        "ca_radial_thickness_fraction": (
            float((maximum_radial - np.min(ca_radial)) / maximum_radial)
            if maximum_radial > 1e-8
            else None
        ),
        "ca_axial_to_radial_aspect_ratio": (
            float(np.ptp(ca_axial) / (2.0 * maximum_radial))
            if maximum_radial > 1e-8
            else None
        ),
    }


def interchain_packing_descriptors(
    ca_by_chain: dict[str, np.ndarray],
    angular_chain_order: list[str],
    *,
    contact_distance: float = 8.0,
) -> dict[str, Any]:
    """Measure coarse neighbouring and non-neighbouring CA packing."""

    if len(angular_chain_order) < 2:
        return {
            "available": False,
            "reason": "at least two angularly ordered chains are required",
        }

    neighbor_keys = {
        frozenset(
            (
                angular_chain_order[index],
                angular_chain_order[(index + 1) % len(angular_chain_order)],
            )
        )
        for index in range(len(angular_chain_order))
    }
    neighbor_contacts: list[int] = []
    neighbor_minimum_distances: list[float] = []
    nonneighbor_contacts = 0
    global_minimum = float("inf")

    chain_ids = sorted(ca_by_chain)
    for left_index, left_chain in enumerate(chain_ids):
        left = ca_by_chain[left_chain]
        for right_chain in chain_ids[left_index + 1 :]:
            right = ca_by_chain[right_chain]
            distances = np.linalg.norm(
                left[:, None, :] - right[None, :, :],
                axis=-1,
            )
            contact_count = int(np.sum(distances < contact_distance))
            minimum = float(np.min(distances))
            global_minimum = min(global_minimum, minimum)
            if frozenset((left_chain, right_chain)) in neighbor_keys:
                neighbor_contacts.append(contact_count)
                neighbor_minimum_distances.append(minimum)
            else:
                nonneighbor_contacts += contact_count

    mean_contacts = float(np.mean(neighbor_contacts))
    mean_chain_length = float(
        np.mean([len(ca_by_chain[chain]) for chain in chain_ids])
    )
    return {
        "available": True,
        "contact_distance": contact_distance,
        "neighbor_pair_count": len(neighbor_contacts),
        "minimum_neighbor_ca_contacts": min(neighbor_contacts),
        "mean_neighbor_ca_contacts": mean_contacts,
        "maximum_neighbor_ca_contacts": max(neighbor_contacts),
        "neighbor_contact_cv": (
            float(np.std(neighbor_contacts) / mean_contacts)
            if mean_contacts > 0.0
            else None
        ),
        "mean_neighbor_contacts_per_chain_residue": (
            mean_contacts / mean_chain_length
            if mean_chain_length > 0.0
            else None
        ),
        "minimum_neighbor_ca_distance": min(
            neighbor_minimum_distances
        ),
        "minimum_interchain_ca_distance": global_minimum,
        "nonneighbor_ca_contacts": nonneighbor_contacts,
    }


def _declared_transforms(
    input_path: Path,
    expected_order: int,
) -> tuple[np.ndarray, ...] | None:
    payload = _load_json(input_path)
    if payload is None or len(payload) != 1:
        return None
    specification = next(iter(payload.values()))
    if not isinstance(specification, dict):
        return None
    extra = specification.get("extra")
    if not isinstance(extra, dict):
        return None
    order = extra.get("registry_transform_order")
    matrices = extra.get("registry_transform_matrices")
    if (
        not isinstance(order, list)
        or not isinstance(matrices, dict)
        or len(order) != expected_order
    ):
        return None
    try:
        transforms = tuple(
            np.asarray(matrices[transform_id], dtype=float)
            for transform_id in order
        )
    except (KeyError, TypeError, ValueError):
        return None
    if any(transform.shape != (4, 4) for transform in transforms):
        return None
    return transforms


def _adapter_metadata(input_path: Path) -> dict[str, Any]:
    payload = _load_json(input_path)
    if payload is None or len(payload) != 1:
        return {}
    specification = next(iter(payload.values()))
    if not isinstance(specification, dict):
        return {}
    extra = specification.get("extra")
    if not isinstance(extra, dict):
        return {}
    return {
        "pose_seed": extra.get("pose_seed"),
        "materialized_linker_length": extra.get(
            "materialized_linker_length"
        ),
        "pose_candidate_manifest": extra.get(
            "pose_candidate_manifest"
        ),
    }


def _screen_one(
    structure_path: Path,
    *,
    expected_order: int,
    run_root: Path,
) -> dict[str, Any]:
    job_id = _job_id_from_path(structure_path)
    job_directory = run_root / job_id if job_id is not None else None
    input_path = (
        job_directory / "adapter/rfd3_input.json"
        if job_directory is not None
        else Path("")
    )
    transforms = (
        _declared_transforms(input_path, expected_order)
        if job_directory is not None
        else None
    )
    atoms = read_structure_atoms(structure_path)
    scaffold = audit_scaffold_geometry(
        atoms,
        expected_symmetry_multiplicity=(
            expected_order if transforms is not None else None
        ),
        expected_symmetry_transforms=transforms,
    )
    ca_by_chain = _ca_coordinates_by_chain(atoms)
    ring = cyclic_ring_descriptors(ca_by_chain, expected_order)
    packing = (
        interchain_packing_descriptors(
            ca_by_chain,
            ring["angular_chain_order"],
        )
        if ring.get("available")
        else {
            "available": False,
            "reason": "ring descriptors are unavailable",
        }
    )
    seed_report = (
        _load_json(job_directory / "seed_integrity_audit.json")
        if job_directory is not None
        else None
    )
    existing_scaffold_report = (
        _load_json(job_directory / "scaffold_validity_audit.json")
        if job_directory is not None
        else None
    )
    metadata = (
        _adapter_metadata(input_path)
        if job_directory is not None
        else {}
    )
    summary = scaffold["summary"]
    seed_passed = (
        seed_report.get("passed")
        if seed_report is not None
        else None
    )
    declared_symmetry_available = transforms is not None
    strict_passed = bool(
        declared_symmetry_available
        and seed_passed is True
        and scaffold["passed"]
    )
    diffusion_seed = None
    if seed_report is not None:
        inputs = seed_report.get("inputs")
        if isinstance(inputs, dict):
            diffusion_seed = inputs.get("rfd3_seed")

    return {
        "structure": str(structure_path),
        "job_id": job_id,
        "expected_order": expected_order,
        **metadata,
        "diffusion_seed": diffusion_seed,
        "seed_audit_available": seed_report is not None,
        "seed_passed": seed_passed,
        "existing_scaffold_audit_available": (
            existing_scaffold_report is not None
        ),
        "existing_scaffold_passed": (
            existing_scaffold_report.get("passed")
            if existing_scaffold_report is not None
            else None
        ),
        "declared_symmetry_available": declared_symmetry_available,
        "recomputed_scaffold_passed": scaffold["passed"],
        "strict_passed": strict_passed,
        "scaffold_summary": summary,
        "ring": ring,
        "packing": packing,
    }


def _sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    summary = record["scaffold_summary"]
    packing = record["packing"]
    contacts = (
        packing.get("minimum_neighbor_ca_contacts", -1)
        if packing.get("available")
        else -1
    )
    return (
        not record["strict_passed"],
        record["seed_passed"] is not True,
        not summary["passed_continuity"],
        not summary["passed_symmetry"],
        summary["chain_break_count"],
        summary["ca_clash_count"],
        -contacts,
        record["job_id"] or "",
    )


def _flat_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["scaffold_summary"]
    ring = record["ring"]
    packing = record["packing"]
    return {
        "rank": record.get("rank"),
        "job_id": record["job_id"],
        "order": record["expected_order"],
        "pose_seed": record.get("pose_seed"),
        "diffusion_seed": record.get("diffusion_seed"),
        "linker_length": record.get("materialized_linker_length"),
        "seed_passed": record["seed_passed"],
        "existing_scaffold_passed": record[
            "existing_scaffold_passed"
        ],
        "geometry_passed": record["recomputed_scaffold_passed"],
        "declared_symmetry_available": record[
            "declared_symmetry_available"
        ],
        "strict_passed": record["strict_passed"],
        "chain_count": summary["chain_count"],
        "chain_breaks": summary["chain_break_count"],
        "ca_clashes": summary["ca_clash_count"],
        "max_chain_ca_rg": summary[
            "maximum_chain_ca_radius_of_gyration"
        ],
        "symmetry_max_error": (
            summary["maximum_symmetry_coordinate_error"]
            if record["declared_symmetry_available"]
            else None
        ),
        "min_interchain_ca_distance": packing.get(
            "minimum_interchain_ca_distance"
        ),
        "min_neighbor_ca_contacts": packing.get(
            "minimum_neighbor_ca_contacts"
        ),
        "mean_neighbor_ca_contacts": packing.get(
            "mean_neighbor_ca_contacts"
        ),
        "neighbor_contacts_per_residue": packing.get(
            "mean_neighbor_contacts_per_chain_residue"
        ),
        "nonneighbor_ca_contacts": packing.get(
            "nonneighbor_ca_contacts"
        ),
        "ring_radius": ring.get("mean_chain_com_radius"),
        "ring_radial_cv": ring.get("chain_com_radial_cv"),
        "ring_axial_rms": ring.get("chain_com_axial_rms"),
        "angular_gap_rms_deg": ring.get(
            "angular_gap_rms_error_degrees"
        ),
        "minimum_ca_axis_clearance": ring.get(
            "minimum_ca_axis_clearance"
        ),
        "structure": record["structure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--symmetry-order", required=True, type=int)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    arguments = parser.parse_args()

    input_directory = arguments.input_dir.resolve()
    if not input_directory.is_dir():
        raise ValueError(
            f"Input directory does not exist: {input_directory}"
        )
    if arguments.symmetry_order < 2:
        raise ValueError("--symmetry-order must be at least two")
    run_root = (
        arguments.run_root.resolve()
        if arguments.run_root is not None
        else input_directory.parent
    )
    structures = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file()
        and path.name.lower().endswith(
            (".cif", ".cif.gz", ".pdb", ".pdb.gz")
        )
    )
    if not structures:
        raise ValueError(
            f"No PDB/mmCIF structures found in {input_directory}"
        )

    records = [
        _screen_one(
            path,
            expected_order=arguments.symmetry_order,
            run_root=run_root,
        )
        for path in structures
    ]
    records.sort(key=_sort_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    prefix = (
        arguments.output_prefix.resolve()
        if arguments.output_prefix is not None
        else input_directory
        / f"c{arguments.symmetry_order}_batch_screen"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "audit": "rfd3_mosaic.extracted_structure_batch_screen",
                "schema_version": 1,
                "input_directory": str(input_directory),
                "run_root": str(run_root),
                "expected_order": arguments.symmetry_order,
                "structure_count": len(records),
                "strict_pass_count": sum(
                    record["strict_passed"] for record in records
                ),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    flat_records = [_flat_record(record) for record in records]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat_records[0]),
        )
        writer.writeheader()
        writer.writerows(flat_records)

    print(f"screened structures: {len(records)}")
    print(
        "strict seed+scaffold passes: "
        f"{sum(record['strict_passed'] for record in records)}"
    )
    print("top candidates:")
    for record in records[: min(10, len(records))]:
        summary = record["scaffold_summary"]
        packing = record["packing"]
        print(
            f"rank={record['rank']:3d} "
            f"job={record['job_id'] or 'unknown'} "
            f"strict={record['strict_passed']} "
            f"seed={record['seed_passed']} "
            f"breaks={summary['chain_break_count']} "
            f"clashes={summary['ca_clash_count']} "
            f"min_neighbor_contacts="
            f"{packing.get('minimum_neighbor_ca_contacts', 'NA')}"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()
