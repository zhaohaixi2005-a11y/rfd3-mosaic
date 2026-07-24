"""Geometry audit for two-fragment interface seeds in symmetric RFD3 output."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

from rfd3_mosaic.structure import AtomRecord


@dataclass(frozen=True)
class FragmentPlacement:
    """Where one source fragment appears in the generated ASU chain."""

    fragment_id: str
    asu_chain_id: str
    output_residues_by_source: dict[int, int]

    @property
    def output_residue_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self.output_residues_by_source.values()))


def _parse_component(component: str) -> tuple[str, int]:
    cursor = 0
    while cursor < len(component) and not (
        component[cursor].isdigit()
        or component[cursor] in {"+", "-"}
    ):
        cursor += 1
    if cursor == 0 or cursor == len(component):
        raise ValueError(f"Invalid residue component {component!r}")
    return component[:cursor], int(component[cursor:])


def infer_fragment_placements(
    mapping: dict[str, Any],
    diffused_index_map: dict[str, str],
) -> dict[str, FragmentPlacement]:
    """Map source fragment residue numbers to generated ASU residue numbers."""

    residue_records: dict[
        tuple[str, str, int], tuple[int, str, int]
    ] = {}
    for record in mapping.get("atom_mappings", []):
        source = record["source"]
        compiled = record["compiled"]
        fragment_id = source["fragment_id"]
        compiled_chain = str(compiled["chain_id"])
        compiled_residue = int(compiled["label_seq_id"])
        key = f"{compiled_chain}{compiled_residue}"
        destination = diffused_index_map.get(key)
        if destination is None:
            continue
        output_chain, output_residue = _parse_component(destination)
        residue_records[
            (fragment_id, compiled_chain, compiled_residue)
        ] = (
            int(source["residue_number"]),
            output_chain,
            output_residue,
        )

    grouped: dict[str, list[tuple[int, str, int]]] = {}
    for (fragment_id, _, _), item in residue_records.items():
        grouped.setdefault(fragment_id, []).append(item)
    if len(grouped) != 2:
        raise ValueError(
            "Seed integrity audit currently requires exactly two indexed "
            f"source fragments, found {sorted(grouped)}"
        )

    placements: dict[str, FragmentPlacement] = {}
    for fragment_id, items in grouped.items():
        output_chains = {item[1] for item in items}
        if len(output_chains) != 1:
            raise ValueError(
                f"Fragment {fragment_id!r} maps to multiple ASU chains"
            )
        source_to_output = {item[0]: item[2] for item in items}
        if len(source_to_output) != len(items):
            raise ValueError(
                f"Fragment {fragment_id!r} has ambiguous residue mapping"
            )
        placements[fragment_id] = FragmentPlacement(
            fragment_id=fragment_id,
            asu_chain_id=next(iter(output_chains)),
            output_residues_by_source=source_to_output,
        )
    return placements


def _is_heavy(atom: AtomRecord) -> bool:
    element = atom.element.upper()
    name = atom.atom_name.lstrip("0123456789").upper()
    return not (element.startswith("H") or name.startswith("H"))


def _atom_lookup(
    atoms: tuple[AtomRecord, ...],
) -> dict[tuple[str, int, str], AtomRecord]:
    lookup: dict[tuple[str, int, str], AtomRecord] = {}
    for atom in atoms:
        if atom.record_type != "ATOM" or not _is_heavy(atom):
            continue
        key = (atom.chain_id, atom.residue_number, atom.atom_name.upper())
        lookup.setdefault(key, atom)
    return lookup


def _reference_lookup(
    atoms: tuple[AtomRecord, ...],
) -> dict[tuple[int, str], AtomRecord]:
    return {
        (atom.residue_number, atom.atom_name.upper()): atom
        for atom in atoms
        if atom.record_type == "ATOM" and _is_heavy(atom)
    }


def _kabsch_rmsd(
    moving: np.ndarray,
    reference: np.ndarray,
) -> float:
    if moving.shape != reference.shape or moving.ndim != 2:
        raise ValueError("Kabsch coordinate arrays must have equal Nx3 shape")
    if moving.shape[0] == 0:
        return float("inf")
    moving_centered = moving - moving.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    covariance = moving_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = (
        -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
    )
    rotation = left @ correction @ right_t
    aligned = moving_centered @ rotation
    return float(
        np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1)))
    )


def _paired_coordinates(
    *,
    fragment_id: str,
    placement: FragmentPlacement,
    output_chain: str,
    reference_atoms: tuple[AtomRecord, ...],
    output_lookup: dict[tuple[str, int, str], AtomRecord],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], int]:
    reference_lookup = _reference_lookup(reference_atoms)
    moving: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    atom_names: list[str] = []
    for (source_residue, atom_name), reference_atom in sorted(
        reference_lookup.items()
    ):
        output_residue = placement.output_residues_by_source.get(
            source_residue
        )
        if output_residue is None:
            continue
        output_atom = output_lookup.get(
            (output_chain, output_residue, atom_name)
        )
        if output_atom is None:
            continue
        moving.append(np.asarray(output_atom.coordinate, dtype=float))
        reference.append(np.asarray(reference_atom.coordinate, dtype=float))
        atom_names.append(atom_name)
    return moving, reference, atom_names, len(reference_lookup)


def _evaluate_pair(
    *,
    left_id: str,
    right_id: str,
    left_chain: str,
    right_chain: str,
    placements: dict[str, FragmentPlacement],
    references: dict[str, tuple[AtomRecord, ...]],
    output_lookup: dict[tuple[str, int, str], AtomRecord],
    contact_cutoff: float,
) -> dict[str, Any]:
    coordinates: dict[
        str,
        tuple[list[np.ndarray], list[np.ndarray], list[str], int],
    ] = {}
    for fragment_id, chain_id in (
        (left_id, left_chain),
        (right_id, right_chain),
    ):
        coordinates[fragment_id] = _paired_coordinates(
            fragment_id=fragment_id,
            placement=placements[fragment_id],
            output_chain=chain_id,
            reference_atoms=references[fragment_id],
            output_lookup=output_lookup,
        )

    output_coordinates = np.asarray(
        coordinates[left_id][0] + coordinates[right_id][0],
        dtype=float,
    )
    reference_coordinates = np.asarray(
        coordinates[left_id][1] + coordinates[right_id][1],
        dtype=float,
    )
    atom_names = coordinates[left_id][2] + coordinates[right_id][2]
    expected_atoms = coordinates[left_id][3] + coordinates[right_id][3]
    matched_atoms = len(output_coordinates)
    completeness = matched_atoms / expected_atoms if expected_atoms else 0.0

    ca_mask = np.asarray([name == "CA" for name in atom_names], dtype=bool)
    all_atom_rmsd = _kabsch_rmsd(
        output_coordinates, reference_coordinates
    )
    ca_rmsd = _kabsch_rmsd(
        output_coordinates[ca_mask],
        reference_coordinates[ca_mask],
    )

    left_count = len(coordinates[left_id][0])
    ref_left = reference_coordinates[:left_count]
    ref_right = reference_coordinates[left_count:]
    out_left = output_coordinates[:left_count]
    out_right = output_coordinates[left_count:]
    ref_distances = np.linalg.norm(
        ref_left[:, None, :] - ref_right[None, :, :], axis=-1
    )
    out_distances = np.linalg.norm(
        out_left[:, None, :] - out_right[None, :, :], axis=-1
    )
    contact_mask = ref_distances <= contact_cutoff
    reference_contacts = int(contact_mask.sum())
    retained_contacts = int(
        np.logical_and(contact_mask, out_distances <= contact_cutoff).sum()
    )
    contact_retention = (
        retained_contacts / reference_contacts
        if reference_contacts
        else 0.0
    )
    contact_distance_rmse = (
        float(
            np.sqrt(
                np.mean(
                    (
                        out_distances[contact_mask]
                        - ref_distances[contact_mask]
                    )
                    ** 2
                )
            )
        )
        if reference_contacts
        else float("inf")
    )
    return {
        "left_chain": left_chain,
        "right_chain": right_chain,
        "cross_chain": left_chain != right_chain,
        "matched_heavy_atoms": matched_atoms,
        "expected_heavy_atoms": expected_atoms,
        "atom_completeness": completeness,
        "all_atom_rmsd": all_atom_rmsd,
        "ca_rmsd": ca_rmsd,
        "reference_contacts": reference_contacts,
        "retained_contacts": retained_contacts,
        "contact_retention": contact_retention,
        "contact_distance_rmse": contact_distance_rmse,
    }


def audit_two_fragment_seed(
    *,
    output_atoms: tuple[AtomRecord, ...],
    references: dict[str, tuple[AtomRecord, ...]],
    placements: dict[str, FragmentPlacement],
    left_fragment_id: str,
    right_fragment_id: str,
    contact_cutoff: float = 4.5,
    max_ca_rmsd: float = 0.5,
    max_all_atom_rmsd: float = 0.75,
    min_contact_retention: float = 0.9,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    """Find the best cross-chain cyclic pairing and audit every seed copy."""

    if set(references) != set(placements):
        raise ValueError("Reference fragments and output placements differ")
    if left_fragment_id == right_fragment_id:
        raise ValueError("Interface seed fragments must be distinct")

    output_lookup = _atom_lookup(output_atoms)
    required_residues = set(
        placements[left_fragment_id].output_residue_numbers
        + placements[right_fragment_id].output_residue_numbers
    )
    chain_residues: dict[str, set[int]] = {}
    for atom in output_atoms:
        if atom.record_type == "ATOM":
            chain_residues.setdefault(atom.chain_id, set()).add(
                atom.residue_number
            )
    chains = sorted(
        chain_id
        for chain_id, residues in chain_residues.items()
        if required_residues.issubset(residues)
    )
    if len(chains) < 2:
        raise ValueError(
            "Generated structure does not contain at least two complete "
            "candidate protomer chains"
        )

    pair_matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for left_chain in chains:
        for right_chain in chains:
            if left_chain == right_chain:
                continue
            pair_matrix[(left_chain, right_chain)] = _evaluate_pair(
                left_id=left_fragment_id,
                right_id=right_fragment_id,
                left_chain=left_chain,
                right_chain=right_chain,
                placements=placements,
                references=references,
                output_lookup=output_lookup,
                contact_cutoff=contact_cutoff,
            )

    best_pairs: list[dict[str, Any]] | None = None
    best_score = float("inf")
    for right_order in permutations(chains):
        if any(left == right for left, right in zip(chains, right_order)):
            continue
        candidate = [
            pair_matrix[(left, right)]
            for left, right in zip(chains, right_order)
        ]
        score = float(np.mean([item["ca_rmsd"] for item in candidate]))
        if score < best_score:
            best_score = score
            best_pairs = candidate
    if best_pairs is None:
        raise ValueError("No one-to-one cross-chain seed pairing is possible")

    for pair in best_pairs:
        failures: list[str] = []
        if pair["atom_completeness"] < min_atom_completeness:
            failures.append("atom_completeness")
        if pair["ca_rmsd"] > max_ca_rmsd:
            failures.append("ca_rmsd")
        if pair["all_atom_rmsd"] > max_all_atom_rmsd:
            failures.append("all_atom_rmsd")
        if pair["contact_retention"] < min_contact_retention:
            failures.append("contact_retention")
        pair["passed"] = not failures
        pair["failed_checks"] = failures

    passed = all(pair["passed"] for pair in best_pairs)
    return {
        "schema_version": 1,
        "audit": "rfd3_mosaic.cross_chain_interface_seed_integrity",
        "passed": passed,
        "fragment_roles": {
            "left": left_fragment_id,
            "right": right_fragment_id,
        },
        "candidate_protomer_chains": chains,
        "pairing_method": "minimum_mean_ca_rmsd_one_to_one_cross_chain",
        "thresholds": {
            "contact_cutoff": contact_cutoff,
            "max_ca_rmsd": max_ca_rmsd,
            "max_all_atom_rmsd": max_all_atom_rmsd,
            "min_contact_retention": min_contact_retention,
            "min_atom_completeness": min_atom_completeness,
        },
        "summary": {
            "seed_pairs": len(best_pairs),
            "passed_pairs": sum(pair["passed"] for pair in best_pairs),
            "maximum_ca_rmsd": max(pair["ca_rmsd"] for pair in best_pairs),
            "maximum_all_atom_rmsd": max(
                pair["all_atom_rmsd"] for pair in best_pairs
            ),
            "minimum_contact_retention": min(
                pair["contact_retention"] for pair in best_pairs
            ),
            "minimum_atom_completeness": min(
                pair["atom_completeness"] for pair in best_pairs
            ),
        },
        "seed_pairs": best_pairs,
    }
