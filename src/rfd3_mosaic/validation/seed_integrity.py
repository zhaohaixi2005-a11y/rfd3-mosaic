"""Geometry audits for interface seeds in symmetric RFD3 output."""

from __future__ import annotations

from dataclasses import dataclass
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
    if not grouped:
        raise ValueError(
            "Seed integrity audit found no indexed source fragments in the "
            "adapter and result mappings"
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


def _minimum_cost_cross_chain_assignment(
    chains: list[str],
    pair_matrix: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Solve the directed chain pairing in O(n^3) with no self-pairs.

    This is the rectangular-potential form of the Hungarian assignment
    algorithm specialized to a square matrix.  It replaces the legacy
    factorial permutation search, which becomes unusable for multiple ASU
    chains or higher symmetry orders.
    """

    chain_count = len(chains)
    costs = np.full((chain_count, chain_count), np.inf, dtype=float)
    for left_index, left_chain in enumerate(chains):
        for right_index, right_chain in enumerate(chains):
            if left_chain == right_chain:
                continue
            costs[left_index, right_index] = pair_matrix[
                (left_chain, right_chain)
            ]["ca_rmsd"]

    row_potential = np.zeros(chain_count + 1, dtype=float)
    column_potential = np.zeros(chain_count + 1, dtype=float)
    matched_row = np.zeros(chain_count + 1, dtype=int)
    predecessor = np.zeros(chain_count + 1, dtype=int)

    for row in range(1, chain_count + 1):
        matched_row[0] = row
        minimum = np.full(chain_count + 1, np.inf, dtype=float)
        used = np.zeros(chain_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, chain_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1, candidate_column - 1]
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if not np.isfinite(delta):
                raise ValueError(
                    "No finite one-to-one cross-chain seed pairing is "
                    "possible"
                )
            for candidate_column in range(chain_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assigned_right_by_left = [-1] * chain_count
    for right_column in range(1, chain_count + 1):
        left_row = matched_row[right_column]
        if left_row:
            assigned_right_by_left[left_row - 1] = right_column - 1
    if any(index < 0 for index in assigned_right_by_left):
        raise ValueError("Cross-chain assignment did not cover every chain")
    return [
        pair_matrix[(chains[left_index], chains[right_index])]
        for left_index, right_index in enumerate(assigned_right_by_left)
    ]


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

    best_pairs = _minimum_cost_cross_chain_assignment(chains, pair_matrix)

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
        "pairing_method": (
            "minimum_total_ca_rmsd_one_to_one_cross_chain_hungarian"
        ),
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


def audit_interface_seed_pairs(
    *,
    output_atoms: tuple[AtomRecord, ...],
    references: dict[str, tuple[AtomRecord, ...]],
    placements: dict[str, FragmentPlacement],
    fragment_pairs: tuple[tuple[str, str], ...],
    contact_cutoff: float = 4.5,
    max_ca_rmsd: float = 0.5,
    max_all_atom_rmsd: float = 0.75,
    min_contact_retention: float = 0.9,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    """Audit one or more independently defined two-fragment interfaces.

    The legacy single-interface report is returned unchanged.  Multiple
    interfaces are evaluated independently and combined under a fail-closed
    top-level report so existing audit gates can continue to inspect
    ``passed`` without understanding the richer schema.
    """

    if not fragment_pairs:
        raise ValueError("At least one interface fragment pair is required")
    if len(set(fragment_pairs)) != len(fragment_pairs):
        raise ValueError("Interface fragment pairs must be unique")

    required_fragments = {
        fragment_id for pair in fragment_pairs for fragment_id in pair
    }
    missing_references = required_fragments - set(references)
    missing_placements = required_fragments - set(placements)
    if missing_references or missing_placements:
        raise ValueError(
            "Interface fragment pairs reference unavailable fragments; "
            f"missing references={sorted(missing_references)}, "
            f"missing placements={sorted(missing_placements)}"
        )

    reports = []
    for left_fragment_id, right_fragment_id in fragment_pairs:
        pair_fragments = {left_fragment_id, right_fragment_id}
        report = audit_two_fragment_seed(
            output_atoms=output_atoms,
            references={
                key: references[key] for key in pair_fragments
            },
            placements={key: placements[key] for key in pair_fragments},
            left_fragment_id=left_fragment_id,
            right_fragment_id=right_fragment_id,
            contact_cutoff=contact_cutoff,
            max_ca_rmsd=max_ca_rmsd,
            max_all_atom_rmsd=max_all_atom_rmsd,
            min_contact_retention=min_contact_retention,
            min_atom_completeness=min_atom_completeness,
        )
        reports.append(report)

    if len(reports) == 1:
        return reports[0]

    seed_pairs = [
        {
            **pair,
            "interface_index": interface_index,
            "left_fragment": report["fragment_roles"]["left"],
            "right_fragment": report["fragment_roles"]["right"],
        }
        for interface_index, report in enumerate(reports)
        for pair in report["seed_pairs"]
    ]
    candidate_chains = sorted(
        {
            chain
            for report in reports
            for chain in report["candidate_protomer_chains"]
        }
    )
    return {
        "schema_version": 1,
        "audit": "rfd3_mosaic.multi_interface_seed_integrity",
        "passed": all(report["passed"] for report in reports),
        "fragment_roles": [
            report["fragment_roles"] for report in reports
        ],
        "candidate_protomer_chains": candidate_chains,
        "pairing_method": (
            "per_interface_minimum_mean_ca_rmsd_one_to_one_cross_chain"
        ),
        "thresholds": reports[0]["thresholds"],
        "summary": {
            "interface_seeds": len(reports),
            "passed_interface_seeds": sum(
                report["passed"] for report in reports
            ),
            "seed_pairs": len(seed_pairs),
            "passed_pairs": sum(pair["passed"] for pair in seed_pairs),
            "maximum_ca_rmsd": max(
                pair["ca_rmsd"] for pair in seed_pairs
            ),
            "maximum_all_atom_rmsd": max(
                pair["all_atom_rmsd"] for pair in seed_pairs
            ),
            "minimum_contact_retention": min(
                pair["contact_retention"] for pair in seed_pairs
            ),
            "minimum_atom_completeness": min(
                pair["atom_completeness"] for pair in seed_pairs
            ),
        },
        "interface_seed_audits": reports,
        "seed_pairs": seed_pairs,
    }
