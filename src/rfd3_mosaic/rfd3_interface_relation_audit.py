"""Audit public assembly-graph interface relations after RFD3 inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from rfd3_mosaic.structure import read_structure_atoms


_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)-([0-9]+)$")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _single_example(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if len(payload) != 1:
        raise ValueError("Compiled input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled example must be an object")
    return example


def _component(value: str) -> tuple[str, int]:
    cursor = 0
    while cursor < len(value) and not value[cursor].isdigit():
        cursor += 1
    if cursor == 0 or cursor == len(value):
        raise ValueError(f"Invalid residue component {value!r}")
    return value[:cursor], int(value[cursor:])


def _parse_selector(value: str) -> tuple[str, int, int]:
    match = _SELECTOR.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Interface relation selector {value!r} is not contiguous"
        )
    chain, start_text, end_text = match.groups()
    start, end = int(start_text), int(end_text)
    if end < start:
        raise ValueError(f"Interface relation selector is reversed: {value}")
    return chain, start, end


def _chain_sort_key(chain: str) -> tuple[int, ...]:
    """Sort compiler chain IDs A..Z, AA..AZ, BA... correctly."""

    if chain and chain.isalpha() and chain.isupper():
        value = 0
        for character in chain:
            value = value * 26 + ord(character) - ord("A") + 1
        return (0, value)
    return (1, *chain.encode("utf-8"))


def _is_heavy(atom) -> bool:
    name = atom.atom_name.lstrip("0123456789").upper()
    return not (atom.element.upper().startswith("H") or name.startswith("H"))


def _derive_result_structure(result_json: Path) -> Path:
    stem = result_json.with_suffix("")
    for suffix in (".cif.gz", ".cif", ".pdb"):
        candidate = Path(f"{stem}{suffix}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No result CIF/PDB exists beside result metadata")


def _fit_transform(
    reference: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if reference.shape != observed.shape or len(reference) < 3:
        raise ValueError("Rigid-pose fitting requires at least three atom pairs")
    reference_center = reference.mean(axis=0)
    observed_center = observed.mean(axis=0)
    reference_zero = reference - reference_center
    observed_zero = observed - observed_center
    left, _, right_t = np.linalg.svd(reference_zero.T @ observed_zero)
    correction = np.eye(3)
    correction[-1, -1] = (
        -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
    )
    rotation = left @ correction @ right_t
    translation = observed_center - reference_center @ rotation
    fitted = reference @ rotation + translation
    rmsd = float(
        np.sqrt(np.mean(np.sum((fitted - observed) ** 2, axis=-1)))
    )
    return rotation, translation, rmsd


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(
        np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    )
    return float(np.rad2deg(np.arccos(cosine)))


def _source_atom_keys(
    selectors: list[str],
    source_lookup: dict[tuple[str, int, str], np.ndarray],
) -> list[tuple[str, int, str]]:
    selected: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for selector in selectors:
        chain, start, end = _parse_selector(selector)
        keys = sorted(
            key
            for key in source_lookup
            if key[0] == chain and start <= key[1] <= end
        )
        if not keys:
            raise ValueError(
                f"Interface relation selector {selector!r} matched no atoms"
            )
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(
                "Interface relation selectors overlap on source atoms: "
                f"{sorted(overlap)[:5]}"
            )
        seen.update(keys)
        selected.extend(keys)
    return selected


def _observed_coordinates(
    atom_keys: list[tuple[str, int, str]],
    *,
    copy_index: int,
    index_map: dict[str, Any],
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    ordered_output_chains: list[str],
    asu_chain_count: int,
) -> tuple[np.ndarray, list[int]]:
    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    coordinates: list[np.ndarray] = []
    keep: list[int] = []
    for atom_index, (source_chain, source_residue, atom_name) in enumerate(
        atom_keys
    ):
        destination = index_map.get(f"{source_chain}{source_residue}")
        if destination is None:
            continue
        master_chain, output_residue = _component(str(destination))
        if master_chain not in chain_positions:
            continue
        asu_chain_index = chain_positions[master_chain] % asu_chain_count
        output_chain_index = copy_index * asu_chain_count + asu_chain_index
        if output_chain_index >= len(ordered_output_chains):
            continue
        output_chain = ordered_output_chains[output_chain_index]
        coordinate = output_lookup.get(
            (output_chain, output_residue, atom_name)
        )
        if coordinate is None:
            continue
        keep.append(atom_index)
        coordinates.append(coordinate)
    return np.asarray(coordinates, dtype=float), keep


def _output_chains_for_source_keys(
    atom_keys: list[tuple[str, int, str]],
    *,
    copy_index: int,
    index_map: dict[str, Any],
    ordered_output_chains: list[str],
    asu_chain_count: int,
) -> set[str]:
    """Resolve which concrete output chains own one compiled port."""

    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    output_chains: set[str] = set()
    for source_chain, source_residue, _ in atom_keys:
        destination = index_map.get(f"{source_chain}{source_residue}")
        if destination is None:
            continue
        master_chain, _ = _component(str(destination))
        if master_chain not in chain_positions:
            continue
        asu_chain_index = chain_positions[master_chain] % asu_chain_count
        output_index = copy_index * asu_chain_count + asu_chain_index
        if output_index < len(ordered_output_chains):
            output_chains.add(ordered_output_chains[output_index])
    return output_chains


def _mapped_fixed_output_residues(
    *,
    index_map: dict[str, Any],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    multiplicity: int,
) -> set[tuple[str, int]]:
    """Materialize all input-mapped motif residues in every output copy."""

    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    master_destinations = {
        _component(str(destination)) for destination in index_map.values()
    }
    mapped: set[tuple[str, int]] = set()
    for master_chain, residue in master_destinations:
        if master_chain not in chain_positions:
            continue
        asu_chain_index = chain_positions[master_chain] % asu_chain_count
        for copy_index in range(multiplicity):
            output_index = copy_index * asu_chain_count + asu_chain_index
            if output_index < len(ordered_output_chains):
                mapped.add((ordered_output_chains[output_index], residue))
    return mapped


def _longest_contiguous_residue_run(
    residues: set[tuple[str, int]],
) -> int:
    maximum = 0
    by_chain: dict[str, list[int]] = {}
    for chain, residue in residues:
        by_chain.setdefault(chain, []).append(residue)
    for numbers in by_chain.values():
        current = 0
        previous: int | None = None
        for residue in sorted(set(numbers)):
            is_adjacent = (
                previous is not None and residue == previous + 1
            )
            current = current + 1 if is_adjacent else 1
            maximum = max(maximum, current)
            previous = residue
    return maximum


def _automatic_interface_targets(
    left_available: int,
    right_available: int,
) -> tuple[int, int]:
    """Mirror the sampler's scale-aware automatic interface contract."""

    available = min(left_available, right_available)
    if available < 1:
        return 0, 0
    coverage = min(available, min(12, max(3, math.ceil(math.sqrt(available)))))
    continuity = min(coverage, max(2, math.ceil(0.6 * coverage)))
    return coverage, continuity


def audit_interface_relations(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
    result_structure: str | Path | None = None,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    """Evaluate every compiler-declared graph edge in the generated result."""

    input_path = Path(compiled_input).resolve()
    result_path = Path(result_json).resolve()
    structure_path = (
        Path(result_structure).resolve()
        if result_structure is not None
        else _derive_result_structure(result_path)
    )
    example = _single_example(input_path)
    result = _load_json(result_path)
    extra = example.get("extra") or {}
    plan = extra.get("assembly_interface_relations")
    if not isinstance(plan, list) or not plan:
        raise ValueError(
            "Compiled input declares no assembly interface relation plan"
        )
    order = extra.get("registry_transform_order")
    matrices = extra.get("registry_transform_matrices")
    multiplicity = int(extra.get("symmetry_multiplicity", 0))
    if (
        not isinstance(order, list)
        or not isinstance(matrices, dict)
        or len(order) != multiplicity
    ):
        raise ValueError("Compiled input lacks a complete symmetry registry")

    source_structure = Path(str(example["input"]))
    if not source_structure.is_absolute():
        source_structure = input_path.parent / source_structure
    source_lookup = {
        (atom.chain_id, atom.residue_number, atom.atom_name.upper()): np.asarray(
            atom.coordinate, dtype=float
        )
        for atom in read_structure_atoms(
            source_structure,
            mmcif_identifier_namespace="label",
        )
        if atom.record_type == "ATOM" and _is_heavy(atom)
    }
    output_atoms = read_structure_atoms(
        structure_path,
        mmcif_identifier_namespace="label",
    )
    output_lookup = {
        (atom.chain_id, atom.residue_number, atom.atom_name.upper()): np.asarray(
            atom.coordinate, dtype=float
        )
        for atom in output_atoms
        if atom.record_type == "ATOM" and _is_heavy(atom)
    }
    ordered_output_chains = sorted(
        {
            atom.chain_id
            for atom in output_atoms
            if atom.record_type == "ATOM"
        },
        key=_chain_sort_key,
    )
    if (
        multiplicity < 1
        or not ordered_output_chains
        or len(ordered_output_chains) % multiplicity != 0
    ):
        raise ValueError(
            "Output chain count is not divisible by symmetry multiplicity: "
            f"chains={ordered_output_chains}, multiplicity={multiplicity}"
        )
    asu_chain_count = len(ordered_output_chains) // multiplicity
    index_map = result.get("diffused_index_map") or {}
    if not isinstance(index_map, dict) or not index_map:
        raise ValueError("Result metadata contains no diffused_index_map")
    mapped_fixed_residues = _mapped_fixed_output_residues(
        index_map=index_map,
        ordered_output_chains=ordered_output_chains,
        asu_chain_count=asu_chain_count,
        multiplicity=multiplicity,
    )

    edge_reports: list[dict[str, Any]] = []
    for edge in plan:
        if not isinstance(edge, dict):
            raise ValueError("Interface relation plan entries must be objects")
        left_keys = _source_atom_keys(
            list(edge["left_source_components"]), source_lookup
        )
        right_keys = _source_atom_keys(
            list(edge["right_source_components"]), source_lookup
        )
        source_copy = int(edge["source_copy_index"])
        target_copy = int(edge["target_copy_index"])
        if not (0 <= source_copy < multiplicity) or not (
            0 <= target_copy < multiplicity
        ):
            raise ValueError(
                f"Interface edge {edge.get('edge_instance_id')!r} has an "
                "out-of-range copy index"
            )

        left_observed, left_keep = _observed_coordinates(
            left_keys,
            copy_index=source_copy,
            index_map=index_map,
            output_lookup=output_lookup,
            ordered_output_chains=ordered_output_chains,
            asu_chain_count=asu_chain_count,
        )
        right_observed, right_keep = _observed_coordinates(
            right_keys,
            copy_index=target_copy,
            index_map=index_map,
            output_lookup=output_lookup,
            ordered_output_chains=ordered_output_chains,
            asu_chain_count=asu_chain_count,
        )
        left_matrix = np.asarray(matrices[str(order[source_copy])], dtype=float)
        right_matrix = np.asarray(matrices[str(order[target_copy])], dtype=float)
        left_master = np.asarray([source_lookup[key] for key in left_keys])
        right_master = np.asarray([source_lookup[key] for key in right_keys])
        left_reference_all = (
            left_master @ left_matrix[:3, :3].T + left_matrix[:3, 3]
        )
        right_reference_all = (
            right_master @ right_matrix[:3, :3].T + right_matrix[:3, 3]
        )
        left_reference = left_reference_all[np.asarray(left_keep, dtype=int)]
        right_reference = right_reference_all[np.asarray(right_keep, dtype=int)]
        expected_atoms = len(left_keys) + len(right_keys)
        matched_atoms = len(left_keep) + len(right_keep)
        completeness = matched_atoms / expected_atoms if expected_atoms else 0.0

        geometry = edge["target_geometry"]
        satisfaction_stage = str(
            edge.get("satisfaction_stage", "input")
        )
        evaluation_scope = "declared_port_atoms"
        left_evaluation_atoms = None
        right_evaluation_atoms = None
        if (
            geometry["mode"] == "geometric_constraints"
            and satisfaction_stage == "output"
        ):
            left_chains = _output_chains_for_source_keys(
                left_keys,
                copy_index=source_copy,
                index_map=index_map,
                ordered_output_chains=ordered_output_chains,
                asu_chain_count=asu_chain_count,
            )
            right_chains = _output_chains_for_source_keys(
                right_keys,
                copy_index=target_copy,
                index_map=index_map,
                ordered_output_chains=ordered_output_chains,
                asu_chain_count=asu_chain_count,
            )
            left_evaluation_atoms = [
                atom
                for atom in output_atoms
                if atom.record_type == "ATOM"
                and _is_heavy(atom)
                and atom.chain_id in left_chains
                and (atom.chain_id, atom.residue_number)
                not in mapped_fixed_residues
            ]
            right_evaluation_atoms = [
                atom
                for atom in output_atoms
                if atom.record_type == "ATOM"
                and _is_heavy(atom)
                and atom.chain_id in right_chains
                and (atom.chain_id, atom.residue_number)
                not in mapped_fixed_residues
            ]
            left_observed = np.asarray(
                [atom.coordinate for atom in left_evaluation_atoms],
                dtype=float,
            )
            right_observed = np.asarray(
                [atom.coordinate for atom in right_evaluation_atoms],
                dtype=float,
            )
            evaluation_scope = "generated_chain_atoms"

        if len(left_observed) and len(right_observed):
            distances = np.linalg.norm(
                left_observed[:, None, :] - right_observed[None, :, :],
                axis=-1,
            )
            centroid_distance = float(
                np.linalg.norm(
                    left_observed.mean(axis=0)
                    - right_observed.mean(axis=0)
                )
            )
            minimum_distance = float(distances.min())
            hard_clashes = int((distances < 2.0).sum())
        else:
            distances = None
            centroid_distance = None
            minimum_distance = None
            hard_clashes = 0

        report: dict[str, Any] = {
            "edge_instance_id": str(edge["edge_instance_id"]),
            "source_interface_id": str(edge["source_interface_id"]),
            "required": bool(edge["required"]),
            "satisfaction_stage": satisfaction_stage,
            "evaluation_scope": evaluation_scope,
            "evaluated_left_atoms": len(left_observed),
            "evaluated_right_atoms": len(right_observed),
            "source_copy_index": source_copy,
            "target_copy_index": target_copy,
            "target_mode": str(geometry["mode"]),
            "reference_basis": str(
                edge.get("reference_basis", "compiled_input")
            ),
            "expected_heavy_atoms": expected_atoms,
            "matched_heavy_atoms": matched_atoms,
            "atom_completeness": completeness,
            "centroid_distance": centroid_distance,
            "minimum_atom_distance": minimum_distance,
            "hard_clashes_below_2_0A": hard_clashes,
        }

        if geometry["mode"] == "reference_transform":
            enough_atoms = len(left_observed) >= 3 and len(right_observed) >= 3
            if enough_atoms:
                left_rotation, left_translation, left_rmsd = _fit_transform(
                    left_reference, left_observed
                )
                right_rotation, _, right_rmsd = _fit_transform(
                    right_reference, right_observed
                )
                predicted_right_center = (
                    right_reference.mean(axis=0) @ left_rotation
                    + left_translation
                )
                translation_error = float(
                    np.linalg.norm(
                        predicted_right_center - right_observed.mean(axis=0)
                    )
                )
                rotation_error = _rotation_error_deg(
                    left_rotation, right_rotation
                )
            else:
                left_rmsd = None
                right_rmsd = None
                translation_error = None
                rotation_error = None
            translation_tolerance = float(geometry["translation_tolerance"])
            rotation_tolerance = float(geometry["rotation_tolerance_deg"])
            contact_cutoff = float(geometry.get("contact_cutoff", 4.5))
            minimum_contacts = int(
                geometry.get("minimum_heavy_atom_contacts", 0)
            )
            contact_count = (
                int((distances < contact_cutoff).sum())
                if distances is not None
                else 0
            )
            contacts_satisfied = contact_count >= minimum_contacts
            satisfied = bool(
                completeness >= min_atom_completeness
                and translation_error is not None
                and translation_error <= translation_tolerance
                and rotation_error is not None
                and rotation_error <= rotation_tolerance
                and contacts_satisfied
                and hard_clashes == 0
            )
            report.update(
                {
                    "left_internal_fit_rmsd": left_rmsd,
                    "right_internal_fit_rmsd": right_rmsd,
                    "translation_error": translation_error,
                    "translation_tolerance": translation_tolerance,
                    "rotation_error_deg": rotation_error,
                    "rotation_tolerance_deg": rotation_tolerance,
                    "declared_contact_cutoff": contact_cutoff,
                    "declared_contact_count": contact_count,
                    "minimum_heavy_atom_contacts": minimum_contacts,
                    "contacts_satisfied": contacts_satisfied,
                    "satisfied": satisfied,
                }
            )
        elif geometry["mode"] == "geometric_constraints":
            checks: list[bool] = []
            distance = geometry.get("distance")
            if distance is not None:
                distance_error = (
                    abs(centroid_distance - float(distance["target"]))
                    if centroid_distance is not None
                    else None
                )
                distance_satisfied = bool(
                    distance_error is not None
                    and distance_error <= float(distance["tolerance"])
                )
                report.update(
                    {
                        "distance_type": str(distance["type"]),
                        "distance_target": float(distance["target"]),
                        "distance_tolerance": float(distance["tolerance"]),
                        "distance_error": distance_error,
                        "distance_satisfied": distance_satisfied,
                    }
                )
                checks.append(distance_satisfied)
            contacts = geometry.get("contacts")
            if contacts is not None:
                cutoff = float(contacts["cutoff"])
                contact_count = (
                    int((distances < cutoff).sum())
                    if distances is not None
                    else 0
                )
                contacts_satisfied = contact_count >= int(
                    contacts["min_heavy_atom_contacts"]
                )
                report.update(
                    {
                        "declared_contact_cutoff": cutoff,
                        "declared_contact_count": contact_count,
                        "minimum_heavy_atom_contacts": int(
                            contacts["min_heavy_atom_contacts"]
                        ),
                        "contacts_satisfied": contacts_satisfied,
                    }
                )
                checks.append(contacts_satisfied)
            coverage = geometry.get("coverage")
            if coverage is not None:
                cutoff = float(
                    (contacts or {}).get("cutoff", 8.0)
                )
                if (
                    distances is not None
                    and left_evaluation_atoms is not None
                    and right_evaluation_atoms is not None
                ):
                    left_indices, right_indices = np.nonzero(
                        distances < cutoff
                    )
                    left_contact_residues = {
                        (
                            left_evaluation_atoms[index].chain_id,
                            left_evaluation_atoms[index].residue_number,
                        )
                        for index in left_indices.tolist()
                    }
                    right_contact_residues = {
                        (
                            right_evaluation_atoms[index].chain_id,
                            right_evaluation_atoms[index].residue_number,
                        )
                        for index in right_indices.tolist()
                    }
                    left_available = len(
                        {
                            (atom.chain_id, atom.residue_number)
                            for atom in left_evaluation_atoms
                        }
                    )
                    right_available = len(
                        {
                            (atom.chain_id, atom.residue_number)
                            for atom in right_evaluation_atoms
                        }
                    )
                else:
                    left_contact_residues = set()
                    right_contact_residues = set()
                    left_available = 0
                    right_available = 0
                automatic_coverage, automatic_continuity = (
                    _automatic_interface_targets(
                        left_available,
                        right_available,
                    )
                )
                minimum_coverage = int(
                    coverage.get("minimum_contact_residues_per_side")
                    or automatic_coverage
                )
                minimum_contiguous = int(
                    coverage.get(
                        "minimum_contiguous_contact_residues_per_side"
                    )
                    or automatic_continuity
                )
                left_coverage = len(left_contact_residues)
                right_coverage = len(right_contact_residues)
                left_contiguous = _longest_contiguous_residue_run(
                    left_contact_residues
                )
                right_contiguous = _longest_contiguous_residue_run(
                    right_contact_residues
                )
                coverage_satisfied = bool(
                    left_coverage >= minimum_coverage
                    and right_coverage >= minimum_coverage
                )
                continuity_satisfied = bool(
                    left_contiguous >= minimum_contiguous
                    and right_contiguous >= minimum_contiguous
                )
                report.update(
                    {
                        "interface_quality_mode": str(
                            coverage.get("mode", "auto")
                        ),
                        "minimum_contact_residues_per_side": minimum_coverage,
                        "contact_residue_count_left": left_coverage,
                        "contact_residue_count_right": right_coverage,
                        "contact_residue_coverage_satisfied": (
                            coverage_satisfied
                        ),
                        "minimum_contiguous_contact_residues_per_side": (
                            minimum_contiguous
                        ),
                        "maximum_contiguous_contact_residues_left": (
                            left_contiguous
                        ),
                        "maximum_contiguous_contact_residues_right": (
                            right_contiguous
                        ),
                        "contact_continuity_satisfied": continuity_satisfied,
                    }
                )
                checks.extend((coverage_satisfied, continuity_satisfied))
            report["satisfied"] = bool(
                completeness >= min_atom_completeness
                and checks
                and all(checks)
                and hard_clashes == 0
            )
        else:
            raise ValueError(
                f"Unsupported target geometry mode {geometry['mode']!r}"
            )
        edge_reports.append(report)

    required_reports = [report for report in edge_reports if report["required"]]
    failed_required = [
        report["edge_instance_id"]
        for report in required_reports
        if not report["satisfied"]
    ]
    interface_ids = sorted(
        {report["source_interface_id"] for report in edge_reports}
    )
    return {
        "audit": "rfd3_mosaic.assembly_interface_relations",
        "schema_version": 1,
        "passed": not failed_required,
        "inputs": {
            "compiled_input": str(input_path),
            "source_structure": str(source_structure.resolve()),
            "result_json": str(result_path),
            "result_structure": str(structure_path),
        },
        "thresholds": {
            "min_atom_completeness": min_atom_completeness,
            "hard_clash_cutoff": 2.0,
        },
        "summary": {
            "symmetry_multiplicity": multiplicity,
            "asu_chain_count": asu_chain_count,
            "interface_count": len(interface_ids),
            "edge_instance_count": len(edge_reports),
            "required_edge_instance_count": len(required_reports),
            "satisfied_required_edge_instance_count": (
                len(required_reports) - len(failed_required)
            ),
            "failed_required_edge_instances": failed_required,
        },
        "interfaces": edge_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit compiler-declared assembly-graph interface relations "
            "after RFD3 inference."
        )
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--min-atom-completeness", type=float, default=0.99
    )
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_interface_relations(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
        result_structure=arguments.result_structure,
        min_atom_completeness=arguments.min_atom_completeness,
    )
    output = arguments.output.resolve()
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        "Assembly interface relation audit: "
        + ("PASSED" if report["passed"] else "FAILED")
    )
    print(
        "required edges: "
        f"{summary['satisfied_required_edge_instance_count']}/"
        f"{summary['required_edge_instance_count']}"
    )
    print(f"report: {output}")
    if not report["passed"] and not arguments.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["audit_interface_relations"]
