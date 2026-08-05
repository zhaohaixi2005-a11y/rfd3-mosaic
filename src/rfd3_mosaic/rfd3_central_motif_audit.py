"""Audit a symmetric fixed central-motif orbit after RFD3 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rfd3_mosaic.structure import read_structure_atoms
from rfd3_mosaic.rfd3_central_motif_probe import _parse_selector


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _single_example(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if len(payload) != 1:
        raise ValueError("Probe input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Probe example must be an object")
    return example


def _component(value: str) -> tuple[str, int]:
    cursor = 0
    while cursor < len(value) and not value[cursor].isdigit():
        cursor += 1
    if cursor == 0 or cursor == len(value):
        raise ValueError(f"Invalid residue component {value!r}")
    return value[:cursor], int(value[cursor:])


def _is_heavy(atom) -> bool:
    name = atom.atom_name.lstrip("0123456789").upper()
    return not (atom.element.upper().startswith("H") or name.startswith("H"))


def _kabsch_align(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    moving_zero = moving - moving_center
    reference_zero = reference - reference_center
    left, _, right_t = np.linalg.svd(moving_zero.T @ reference_zero)
    correction = np.eye(3)
    correction[-1, -1] = (
        -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
    )
    rotation = left @ correction @ right_t
    return moving_zero @ rotation + reference_center


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=-1))))


def _derive_result_structure(result_json: Path) -> Path:
    stem = result_json.with_suffix("")
    for suffix in (".cif.gz", ".cif", ".pdb"):
        candidate = Path(f"{stem}{suffix}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No result CIF/PDB exists beside result metadata")


def audit_central_motif(
    *,
    probe_input: str | Path,
    result_json: str | Path,
    result_structure: str | Path | None = None,
    max_joint_rmsd: float = 0.5,
    max_joint_coordinate_error: float = 0.001,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    input_path = Path(probe_input).resolve()
    result_json_path = Path(result_json).resolve()
    example = _single_example(input_path)
    result = _load_json(result_json_path)
    extra = example.get("extra") or {}
    declared_selectors = extra.get("probe_fixed_selectors")
    if declared_selectors is None:
        selector = extra.get("probe_fixed_selector")
        if selector is not None:
            selectors = [str(selector)]
        else:
            selectors = list((example.get("select_fixed_atoms") or {}).keys())
    elif isinstance(declared_selectors, list):
        selectors = [str(selector) for selector in declared_selectors]
    else:
        raise ValueError("probe_fixed_selectors must be a list")
    if not selectors:
        raise ValueError("Probe input declares no fixed motif selectors")
    parsed_selectors = [_parse_selector(selector) for selector in selectors]
    source_structure = Path(str(example["input"]))
    if not source_structure.is_absolute():
        source_structure = input_path.parent / source_structure

    source_lookup = {}
    for atom in read_structure_atoms(
        source_structure,
        mmcif_identifier_namespace="label",
    ):
        if atom.record_type != "ATOM" or not _is_heavy(atom):
            continue
        if not any(
            atom.chain_id == chain and start <= atom.residue_number <= end
            for chain, start, end in parsed_selectors
        ):
            continue
        key = (
            atom.chain_id,
            atom.residue_number,
            atom.atom_name.upper(),
        )
        if key in source_lookup:
            raise ValueError(f"Duplicate source motif atom {key}")
        source_lookup[key] = np.asarray(atom.coordinate, dtype=float)
    if not source_lookup:
        raise ValueError(f"Selectors {selectors!r} matched no source atoms")

    index_map = result.get("diffused_index_map", {})
    residue_map: dict[tuple[str, int], int] = {}
    master_output_chain: str | None = None
    for source_chain, start, end in parsed_selectors:
        for source_residue in range(start, end + 1):
            destination = index_map.get(f"{source_chain}{source_residue}")
            if destination is None:
                continue
            output_chain, output_residue = _component(str(destination))
            if master_output_chain is None:
                master_output_chain = output_chain
            elif output_chain != master_output_chain:
                raise ValueError("Fixed motif maps to multiple master chains")
            residue_map[(source_chain, source_residue)] = output_residue
    if master_output_chain is None:
        raise ValueError("Result metadata contains no central-motif mapping")

    output_atoms = read_structure_atoms(
        Path(result_structure).resolve()
        if result_structure is not None
        else _derive_result_structure(result_json_path),
        mmcif_identifier_namespace="label",
    )
    output_lookup = {
        (atom.chain_id, atom.residue_number, atom.atom_name.upper()): np.asarray(
            atom.coordinate, dtype=float
        )
        for atom in output_atoms
        if atom.record_type == "ATOM" and _is_heavy(atom)
    }
    candidate_chains = sorted(
        {
            atom.chain_id
            for atom in output_atoms
            if atom.record_type == "ATOM"
            and atom.residue_number in set(residue_map.values())
        }
    )

    matrices = example.get("extra", {}).get("registry_transform_matrices")
    order = example.get("extra", {}).get("registry_transform_order")
    multiplicity = int(example["extra"]["symmetry_multiplicity"])
    if not isinstance(matrices, dict) or not isinstance(order, list):
        raise ValueError("Probe input lacks the validated transform registry")
    if len(order) != multiplicity or len(candidate_chains) != multiplicity:
        raise ValueError(
            "Output chain count or transform count does not match symmetry "
            f"multiplicity {multiplicity}: chains={candidate_chains}, order={order}"
        )

    atom_keys = sorted(source_lookup)
    master = np.asarray([source_lookup[key] for key in atom_keys])
    expected_copies = []
    observed_copies = []
    matched_per_copy = []
    for chain_id, transform_name in zip(candidate_chains, order):
        matrix = np.asarray(matrices[str(transform_name)], dtype=float)
        expected = master @ matrix[:3, :3].T + matrix[:3, 3]
        observed = []
        keep = []
        for atom_index, (
            source_chain,
            source_residue,
            atom_name,
        ) in enumerate(atom_keys):
            output_residue = residue_map.get((source_chain, source_residue))
            coordinate = output_lookup.get(
                (chain_id, output_residue, atom_name)
            )
            if coordinate is None:
                continue
            keep.append(atom_index)
            observed.append(coordinate)
        expected_copies.append(expected[np.asarray(keep, dtype=int)])
        observed_copies.append(np.asarray(observed, dtype=float))
        matched_per_copy.append(len(observed))

    matched = sum(matched_per_copy)
    expected_count = len(atom_keys) * multiplicity
    completeness = matched / expected_count if expected_count else 0.0
    if not matched or len(set(matched_per_copy)) != 1:
        joint_rmsd = float("inf")
        joint_maximum = float("inf")
        distance_matrix_rmsd = float("inf")
        joint_coordinate_rmsd = float("inf")
        joint_coordinate_maximum_error = float("inf")
        per_copy_rmsd = [float("inf")] * multiplicity
    else:
        expected_joint = np.concatenate(expected_copies, axis=0)
        observed_joint = np.concatenate(observed_copies, axis=0)
        aligned_joint = _kabsch_align(observed_joint, expected_joint)
        deviations = np.linalg.norm(aligned_joint - expected_joint, axis=-1)
        joint_rmsd = _rmsd(aligned_joint, expected_joint)
        joint_maximum = float(np.max(deviations))
        raw_deviations = np.linalg.norm(
            observed_joint - expected_joint,
            axis=-1,
        )
        joint_coordinate_rmsd = _rmsd(observed_joint, expected_joint)
        joint_coordinate_maximum_error = float(np.max(raw_deviations))
        expected_distances = np.linalg.norm(
            expected_joint[:, None, :] - expected_joint[None, :, :], axis=-1
        )
        observed_distances = np.linalg.norm(
            observed_joint[:, None, :] - observed_joint[None, :, :], axis=-1
        )
        distance_matrix_rmsd = float(
            np.sqrt(np.mean((observed_distances - expected_distances) ** 2))
        )
        per_copy_rmsd = [
            _rmsd(_kabsch_align(observed, expected), expected)
            for observed, expected in zip(observed_copies, expected_copies)
        ]

    passed = bool(
        completeness >= min_atom_completeness
        and joint_rmsd <= max_joint_rmsd
        and joint_coordinate_maximum_error <= max_joint_coordinate_error
    )
    return {
        "audit": "rfd3_mosaic.central_fixed_motif_orbit",
        "schema_version": 1,
        "passed": passed,
        "inputs": {
            "probe_input": str(input_path),
            "source_structure": str(source_structure.resolve()),
            "result_json": str(result_json_path),
            "result_structure": str(
                Path(result_structure).resolve()
                if result_structure is not None
                else _derive_result_structure(result_json_path)
            ),
            "fixed_selector": selectors[0] if len(selectors) == 1 else None,
            "fixed_selectors": selectors,
        },
        "thresholds": {
            "max_joint_orbit_rmsd": max_joint_rmsd,
            "max_joint_coordinate_error": max_joint_coordinate_error,
            "min_atom_completeness": min_atom_completeness,
        },
        "summary": {
            "symmetry_multiplicity": multiplicity,
            "output_chains": candidate_chains,
            "expected_heavy_atoms": expected_count,
            "matched_heavy_atoms": matched,
            "atom_completeness": completeness,
            "joint_orbit_rmsd": joint_rmsd,
            "joint_orbit_maximum_error": joint_maximum,
            "joint_coordinate_rmsd": joint_coordinate_rmsd,
            "joint_coordinate_maximum_error": (
                joint_coordinate_maximum_error
            ),
            "orbit_distance_matrix_rmsd": distance_matrix_rmsd,
            "maximum_per_copy_internal_rmsd": max(per_copy_rmsd),
            "per_copy_internal_rmsd": per_copy_rmsd,
        },
    }


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
        "joint coordinate max error: "
        f"{summary['joint_coordinate_maximum_error']:.6f} A"
    )
    print(
        "orbit distance-matrix RMSD: "
        f"{summary['orbit_distance_matrix_rmsd']:.6f} A"
    )
    print(f"report: {arguments.output.resolve()}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
