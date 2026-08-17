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
_PROVENANCE_EQUIVALENCE_TOLERANCE = 0.02

_AtomKey = tuple[str, int, str]
_RuntimeBinding = tuple[_AtomKey, int]


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


def _runtime_action_index(value: Any, registry_order: list[str]) -> int:
    """Resolve a declared runtime action encoded as an index or registry id."""

    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean symmetry action {value!r}")
    if isinstance(value, int):
        index = value
    elif isinstance(value, str) and value in registry_order:
        index = registry_order.index(value)
    else:
        try:
            index = int(str(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown runtime symmetry action {value!r}"
            ) from error
    if not 0 <= index < len(registry_order):
        raise ValueError(
            f"Runtime symmetry action {index} is outside the declared "
            f"registry of size {len(registry_order)}"
        )
    return index


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


def _source_atom_key_blocks(
    selectors: list[str],
    source_lookup: dict[tuple[str, int, str], np.ndarray],
) -> list[list[_AtomKey]]:
    """Select source atoms while preserving explicit selector boundaries."""

    blocks: list[list[_AtomKey]] = []
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
        blocks.append(keys)
    return blocks


def _edge_side_fragment_ids(
    extra: dict[str, Any],
    *,
    edge_instance_id: str,
    role: str,
) -> set[str] | None:
    """Resolve authoritative source-fragment identities when available."""

    topology = extra.get("interleaved_interface_seed_topology")
    if not isinstance(topology, dict):
        return None
    sides = topology.get("interface_sides")
    if not isinstance(sides, list):
        return None
    fragment_ids: set[str] = set()
    found = False
    for side in sides:
        if not isinstance(side, dict):
            continue
        if (
            str(side.get("edge_instance_id", "")) != edge_instance_id
            or str(side.get("role", "")) != role
        ):
            continue
        found = True
        instances = side.get("fragment_instance_ids")
        if not isinstance(instances, list):
            continue
        fragment_ids.update(
            str(instance).split("@", 1)[0] for instance in instances
        )
    return fragment_ids if found and fragment_ids else None


def _ordered_residue_keys(
    atom_keys: list[_AtomKey],
) -> list[list[_AtomKey]]:
    """Group already ordered atom keys by residue, preserving atom order."""

    residues: list[list[_AtomKey]] = []
    current: list[_AtomKey] = []
    previous: tuple[str, int] | None = None
    for key in atom_keys:
        residue = (key[0], key[1])
        if current and residue != previous:
            residues.append(current)
            current = []
        current.append(key)
        previous = residue
    if current:
        residues.append(current)
    return residues


def _member_residue_runs(
    member: dict[str, Any],
    source_lookup: dict[_AtomKey, np.ndarray],
) -> list[list[list[_AtomKey]]]:
    """Return candidate contiguous residue runs for one runtime member."""

    components = member.get("src_components")
    if not isinstance(components, list) or not components:
        return []
    residues: list[list[_AtomKey]] = []
    for value in components:
        residue_id = _component(str(value))
        keys = sorted(
            key
            for key in source_lookup
            if (key[0], key[1]) == residue_id
        )
        if keys:
            residues.append(keys)

    runs: list[list[list[_AtomKey]]] = []
    current: list[list[_AtomKey]] = []
    previous: tuple[str, int] | None = None
    for keys in residues:
        residue = (keys[0][0], keys[0][1])
        if (
            current
            and previous is not None
            and (
                residue[0] != previous[0]
                or residue[1] != previous[1] + 1
            )
        ):
            runs.append(current)
            current = []
        current.append(keys)
        previous = residue
    if current:
        runs.append(current)
    return runs


def _transformed_source_coordinates(
    atom_keys: list[_AtomKey],
    *,
    action_index: int,
    source_lookup: dict[_AtomKey, np.ndarray],
    registry_order: list[str],
    registry_matrices: dict[str, Any],
) -> np.ndarray:
    matrix = np.asarray(
        registry_matrices[registry_order[action_index]], dtype=float
    )
    coordinates = np.asarray(
        [source_lookup[key] for key in atom_keys], dtype=float
    )
    return coordinates @ matrix[:3, :3].T + matrix[:3, 3]


def _resolve_runtime_bindings(
    atom_keys: list[_AtomKey],
    *,
    atom_key_blocks: list[list[_AtomKey]],
    desired_action_index: int,
    source_lookup: dict[_AtomKey, np.ndarray],
    index_map: dict[str, Any],
    motif_constraint_groups: Any,
    allowed_source_fragment_ids: set[str] | None,
    registry_order: list[str],
    registry_matrices: dict[str, Any],
    preexpanded: bool = False,
) -> tuple[list[_RuntimeBinding | None], int]:
    """Bind reference-port atoms to their compiled runtime provenance.

    A compiler-selected asymmetric unit may cross the cyclic seam.  In that
    case an original port such as ``B1-3@copy0`` can be represented at runtime
    by a symmetry-equivalent source fragment such as ``F1-3@copy1``.  The
    authoritative runtime members are ``motif_constraint_groups``; absolute
    transformed source coordinates establish the equivalent correspondence.
    """

    bindings: list[_RuntimeBinding | None] = [None] * len(atom_keys)
    used: set[_RuntimeBinding] = set()
    remapped = 0
    groups = (
        motif_constraint_groups
        if isinstance(motif_constraint_groups, list)
        else []
    )

    atom_indices = {key: index for index, key in enumerate(atom_keys)}
    for block_keys in atom_key_blocks:
        block_indices = [atom_indices[key] for key in block_keys]
        if (preexpanded or not groups) and all(
            f"{chain}{residue}" in index_map
            for chain, residue, _ in block_keys
        ):
            for index, key in zip(block_indices, block_keys, strict=True):
                binding = (key, desired_action_index)
                bindings[index] = binding
                used.add(binding)
            continue

        original_residues = _ordered_residue_keys(block_keys)
        original_signature = [
            tuple(key[2] for key in residue_keys)
            for residue_keys in original_residues
        ]
        expected = _transformed_source_coordinates(
            block_keys,
            action_index=desired_action_index,
            source_lookup=source_lookup,
            registry_order=registry_order,
            registry_matrices=registry_matrices,
        )
        candidates: list[tuple[float, list[_RuntimeBinding]]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            members = group.get("members")
            if not isinstance(members, list):
                continue
            for member in members:
                if not isinstance(member, dict):
                    continue
                if allowed_source_fragment_ids is not None and str(
                    member.get("source_fragment_id", "")
                ) not in allowed_source_fragment_ids:
                    continue
                for residue_run in _member_residue_runs(
                    member, source_lookup
                ):
                    window_size = len(original_residues)
                    for start in range(
                        0, len(residue_run) - window_size + 1
                    ):
                        residue_window = residue_run[
                            start : start + window_size
                        ]
                        signature = [
                            tuple(key[2] for key in residue_keys)
                            for residue_keys in residue_window
                        ]
                        if signature != original_signature:
                            continue
                        action_index = _runtime_action_index(
                            member.get("sym_transform_id"), registry_order
                        )
                        candidate_keys = [
                            key
                            for residue_keys in residue_window
                            for key in residue_keys
                        ]
                        candidate_bindings = [
                            (key, action_index) for key in candidate_keys
                        ]
                        if any(
                            binding in used
                            for binding in candidate_bindings
                        ):
                            continue
                        if not all(
                            f"{chain}{residue}" in index_map
                            for chain, residue, _ in candidate_keys
                        ):
                            continue
                        candidate = _transformed_source_coordinates(
                            candidate_keys,
                            action_index=action_index,
                            source_lookup=source_lookup,
                            registry_order=registry_order,
                            registry_matrices=registry_matrices,
                        )
                        rmsd = float(
                            np.sqrt(
                                np.mean(
                                    np.sum((expected - candidate) ** 2, axis=-1)
                                )
                            )
                        )
                        candidates.append((rmsd, candidate_bindings))

        if candidates:
            score, selected = min(candidates, key=lambda item: item[0])
            if score <= _PROVENANCE_EQUIVALENCE_TOLERANCE:
                for index, binding in zip(block_indices, selected, strict=True):
                    bindings[index] = binding
                    used.add(binding)
                    if binding[0] != atom_keys[index] or (
                        binding[1] != desired_action_index
                    ):
                        remapped += 1
                continue

        # Preserve the historical incomplete-atom report when no equivalent
        # runtime provenance exists; do not manufacture a correspondence.
        for index, key in zip(block_indices, block_keys, strict=True):
            if f"{key[0]}{key[1]}" in index_map:
                binding = (key, desired_action_index)
                bindings[index] = binding
                used.add(binding)
    return bindings, remapped


def _observed_coordinates(
    bindings: list[_RuntimeBinding | None],
    *,
    index_map: dict[str, Any],
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    preexpanded: bool = False,
) -> tuple[np.ndarray, list[int]]:
    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    coordinates: list[np.ndarray] = []
    keep: list[int] = []
    for atom_index, binding in enumerate(bindings):
        if binding is None:
            continue
        (source_chain, source_residue, atom_name), action_index = binding
        destination = index_map.get(f"{source_chain}{source_residue}")
        if destination is None:
            continue
        master_chain, output_residue = _component(str(destination))
        if master_chain not in chain_positions:
            continue
        if preexpanded:
            output_chain_index = chain_positions[master_chain]
        else:
            asu_chain_index = chain_positions[master_chain] % asu_chain_count
            output_chain_index = (
                action_index * asu_chain_count + asu_chain_index
            )
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


def _output_chains_for_bindings(
    bindings: list[_RuntimeBinding | None],
    *,
    index_map: dict[str, Any],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    preexpanded: bool = False,
) -> set[str]:
    """Resolve which concrete output chains own one compiled port."""

    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    output_chains: set[str] = set()
    for binding in bindings:
        if binding is None:
            continue
        (source_chain, source_residue, _), action_index = binding
        destination = index_map.get(f"{source_chain}{source_residue}")
        if destination is None:
            continue
        master_chain, _ = _component(str(destination))
        if master_chain not in chain_positions:
            continue
        if preexpanded:
            output_index = chain_positions[master_chain]
        else:
            asu_chain_index = chain_positions[master_chain] % asu_chain_count
            output_index = action_index * asu_chain_count + asu_chain_index
        if output_index < len(ordered_output_chains):
            output_chains.add(ordered_output_chains[output_index])
    return output_chains


def _mapped_fixed_output_residues(
    *,
    index_map: dict[str, Any],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    multiplicity: int,
    preexpanded: bool = False,
) -> set[tuple[str, int]]:
    """Materialize all input-mapped motif residues in every output copy."""

    chain_positions = {
        chain: index for index, chain in enumerate(ordered_output_chains)
    }
    master_destinations = {
        _component(str(destination)) for destination in index_map.values()
    }
    if preexpanded:
        return {
            destination
            for destination in master_destinations
            if destination[0] in chain_positions
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


def _contiguous_residue_island_count(
    residues: set[tuple[str, int]],
) -> int:
    """Count sequence-contiguous contact islands without joining chains."""

    islands = 0
    by_chain: dict[str, list[int]] = {}
    for chain, residue in residues:
        by_chain.setdefault(chain, []).append(residue)
    for numbers in by_chain.values():
        previous: int | None = None
        for residue in sorted(set(numbers)):
            if previous is None or residue != previous + 1:
                islands += 1
            previous = residue
    return islands


def _heavy_atom_packing_metrics(
    left_atoms: list[Any],
    right_atoms: list[Any],
    distances: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, Any]:
    """Measure reciprocal residue packing in the written all-atom result.

    These descriptors deliberately keep the word ``proxy`` where no solvent
    surface calculation is performed.  They are nevertheless stronger than
    a raw atom-pair count: both residue sides must participate, sequence
    islands remain visible, and a single deep point contact cannot masquerade
    as one broad uniformly packed patch.
    """

    contact_rows, contact_columns = np.nonzero(distances < cutoff)
    left_keys = [
        (atom.chain_id, atom.residue_number) for atom in left_atoms
    ]
    right_keys = [
        (atom.chain_id, atom.residue_number) for atom in right_atoms
    ]
    residue_pairs = {
        (left_keys[left_index], right_keys[right_index])
        for left_index, right_index in zip(
            contact_rows.tolist(),
            contact_columns.tolist(),
            strict=True,
        )
    }
    left_contact_residues = {pair[0] for pair in residue_pairs}
    right_contact_residues = {pair[1] for pair in residue_pairs}
    hydrophobic_names = {
        "ALA",
        "VAL",
        "ILE",
        "LEU",
        "MET",
        "PHE",
        "TRP",
        "TYR",
        "PRO",
    }
    residue_names = {
        (atom.chain_id, atom.residue_number): atom.residue_name.upper()
        for atom in (*left_atoms, *right_atoms)
    }
    available_residues = set(residue_names)
    contact_residues = left_contact_residues | right_contact_residues
    hydrophobic_available = {
        key
        for key in available_residues
        if residue_names[key] in hydrophobic_names
    }
    hydrophobic_contacting = hydrophobic_available & contact_residues

    def residue_nearest_depths(
        atom_keys: list[tuple[str, int]],
        nearest: np.ndarray,
        contacting: set[tuple[str, int]],
    ) -> list[float]:
        minima: dict[tuple[str, int], float] = {}
        for key, distance in zip(atom_keys, nearest.tolist(), strict=True):
            if key in contacting:
                minima[key] = min(minima.get(key, float("inf")), distance)
        return [cutoff - minima[key] for key in sorted(minima)]

    depths = residue_nearest_depths(
        left_keys,
        distances.min(axis=1),
        left_contact_residues,
    ) + residue_nearest_depths(
        right_keys,
        distances.min(axis=0),
        right_contact_residues,
    )
    depth_mean = float(np.mean(depths)) if depths else 0.0
    depth_std = float(np.std(depths)) if depths else 0.0
    residue_pair_count = len(residue_pairs)
    reciprocal_density = residue_pair_count / max(
        len(left_contact_residues),
        len(right_contact_residues),
        1,
    )
    burial_proxy = float(
        np.clip((cutoff - distances) / max(cutoff - 2.0, 1.0), 0.0, 1.0)
        .sum()
    )

    # A loose residue-pair grid exposes holes inside the already selected
    # contact patch.  This is a porosity descriptor, not a SASA calculation.
    loose_pairs: set[
        tuple[tuple[str, int], tuple[str, int]]
    ] = set()
    loose_rows, loose_columns = np.nonzero(distances < cutoff + 1.5)
    for left_index, right_index in zip(
        loose_rows.tolist(),
        loose_columns.tolist(),
        strict=True,
    ):
        left_key = left_keys[left_index]
        right_key = right_keys[right_index]
        if (
            left_key in left_contact_residues
            and right_key in right_contact_residues
        ):
            loose_pairs.add((left_key, right_key))
    possible_patch_pairs = max(
        len(left_contact_residues) * len(right_contact_residues),
        1,
    )
    return {
        "reciprocal_contact_residue_pair_count": residue_pair_count,
        "reciprocal_contact_density": float(reciprocal_density),
        "heavy_atom_burial_proxy": burial_proxy,
        "contact_depth_mean": depth_mean,
        "contact_depth_standard_deviation": depth_std,
        "contact_island_count_left": _contiguous_residue_island_count(
            left_contact_residues
        ),
        "contact_island_count_right": _contiguous_residue_island_count(
            right_contact_residues
        ),
        "local_contact_void_fraction_proxy": float(
            1.0 - len(loose_pairs) / possible_patch_pairs
        ),
        "hydrophobic_contact_residue_fraction_proxy": float(
            len(hydrophobic_contacting) / max(len(contact_residues), 1)
        ),
        "unpaired_hydrophobic_residue_fraction_proxy": float(
            len(hydrophobic_available - contact_residues)
            / max(len(available_residues), 1)
        ),
    }


def _automatic_interface_targets(
    left_available: int,
    right_available: int,
    *,
    left_contiguous_capacity: int | None = None,
    right_contiguous_capacity: int | None = None,
) -> tuple[int, int]:
    """Derive feasible scale-aware automatic interface targets.

    Coverage may be accumulated across several generated regions on one
    output chain.  Continuity cannot: a fixed motif or an ungenerated residue
    gap separates two contact patches permanently.  Automatic mode therefore
    caps its contiguous-patch target by the longest generated residue run on
    both sides.  Explicit user targets are handled by the caller and are never
    relaxed here.
    """

    available = min(left_available, right_available)
    if available < 1:
        return 0, 0
    coverage = min(available, min(12, max(3, math.ceil(math.sqrt(available)))))
    continuity = min(coverage, max(2, math.ceil(0.6 * coverage)))
    capacities = tuple(
        capacity
        for capacity in (
            left_contiguous_capacity,
            right_contiguous_capacity,
        )
        if capacity is not None
    )
    if capacities:
        continuity = min(continuity, *capacities)
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
    preexpanded_layout = extra.get("preexpanded_chain_layout")
    if not ordered_output_chains or multiplicity < 1:
        raise ValueError("Interface audit requires output chains and symmetry")
    if preexpanded_layout is not None:
        if len(preexpanded_layout) != len(ordered_output_chains):
            raise ValueError(
                "Preexpanded chain layout does not match output chain "
                f"count: {len(preexpanded_layout)} != "
                f"{len(ordered_output_chains)}"
            )
        asu_chain_count = sum(
            bool(record.get("is_asu", False))
            for record in preexpanded_layout
        )
    elif len(ordered_output_chains) % multiplicity != 0:
        raise ValueError(
            "Output chain count is not divisible by symmetry multiplicity: "
            f"chains={ordered_output_chains}, multiplicity={multiplicity}"
        )
    else:
        asu_chain_count = len(ordered_output_chains) // multiplicity
    index_map = result.get("diffused_index_map") or {}
    if not isinstance(index_map, dict) or not index_map:
        raise ValueError("Result metadata contains no diffused_index_map")
    mapped_fixed_residues = _mapped_fixed_output_residues(
        index_map=index_map,
        ordered_output_chains=ordered_output_chains,
        asu_chain_count=asu_chain_count,
        multiplicity=multiplicity,
        preexpanded=preexpanded_layout is not None,
    )
    motif_constraint_groups = extra.get("motif_constraint_groups")

    edge_reports: list[dict[str, Any]] = []
    for edge in plan:
        if not isinstance(edge, dict):
            raise ValueError("Interface relation plan entries must be objects")
        left_blocks = _source_atom_key_blocks(
            list(edge["left_source_components"]), source_lookup
        )
        right_blocks = _source_atom_key_blocks(
            list(edge["right_source_components"]), source_lookup
        )
        left_keys = [key for block in left_blocks for key in block]
        right_keys = [key for block in right_blocks for key in block]
        source_copy = int(
            edge.get("left_transform_index")
            if edge.get("left_transform_index") is not None
            else edge["source_copy_index"]
        )
        target_copy = int(
            edge.get("right_transform_index")
            if edge.get("right_transform_index") is not None
            else edge["target_copy_index"]
        )
        if not (0 <= source_copy < multiplicity) or not (
            0 <= target_copy < multiplicity
        ):
            raise ValueError(
                f"Interface edge {edge.get('edge_instance_id')!r} has an "
                "out-of-range copy index"
            )

        left_bindings, left_remapped = _resolve_runtime_bindings(
            left_keys,
            atom_key_blocks=left_blocks,
            desired_action_index=source_copy,
            source_lookup=source_lookup,
            index_map=index_map,
            motif_constraint_groups=motif_constraint_groups,
            allowed_source_fragment_ids=_edge_side_fragment_ids(
                extra,
                edge_instance_id=str(edge["edge_instance_id"]),
                role="left",
            ),
            registry_order=order,
            registry_matrices=matrices,
            preexpanded=preexpanded_layout is not None,
        )
        right_bindings, right_remapped = _resolve_runtime_bindings(
            right_keys,
            atom_key_blocks=right_blocks,
            desired_action_index=target_copy,
            source_lookup=source_lookup,
            index_map=index_map,
            motif_constraint_groups=motif_constraint_groups,
            allowed_source_fragment_ids=_edge_side_fragment_ids(
                extra,
                edge_instance_id=str(edge["edge_instance_id"]),
                role="right",
            ),
            registry_order=order,
            registry_matrices=matrices,
            preexpanded=preexpanded_layout is not None,
        )
        left_observed, left_keep = _observed_coordinates(
            left_bindings,
            index_map=index_map,
            output_lookup=output_lookup,
            ordered_output_chains=ordered_output_chains,
            asu_chain_count=asu_chain_count,
            preexpanded=preexpanded_layout is not None,
        )
        right_observed, right_keep = _observed_coordinates(
            right_bindings,
            index_map=index_map,
            output_lookup=output_lookup,
            ordered_output_chains=ordered_output_chains,
            asu_chain_count=asu_chain_count,
            preexpanded=preexpanded_layout is not None,
        )
        left_matrix = np.asarray(matrices[str(order[source_copy])], dtype=float)
        right_matrix = np.asarray(matrices[str(order[target_copy])], dtype=float)
        left_master = np.asarray([source_lookup[key] for key in left_keys])
        right_master = np.asarray([source_lookup[key] for key in right_keys])
        if preexpanded_layout is not None:
            # The compiler materialized each quotient copy directly in the
            # input structure; applying its group transform again would
            # rotate the reference twice.
            left_reference_all = left_master
            right_reference_all = right_master
        else:
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
            left_chains = _output_chains_for_bindings(
                left_bindings,
                index_map=index_map,
                ordered_output_chains=ordered_output_chains,
                asu_chain_count=asu_chain_count,
                preexpanded=preexpanded_layout is not None,
            )
            right_chains = _output_chains_for_bindings(
                right_bindings,
                index_map=index_map,
                ordered_output_chains=ordered_output_chains,
                asu_chain_count=asu_chain_count,
                preexpanded=preexpanded_layout is not None,
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
            "hyperedge_id": str(
                edge.get("hyperedge_id") or edge["source_interface_id"]
            ),
            "required": bool(edge["required"]),
            "satisfaction_stage": satisfaction_stage,
            "evaluation_scope": evaluation_scope,
            "evaluated_left_atoms": len(left_observed),
            "evaluated_right_atoms": len(right_observed),
            "source_copy_index": source_copy,
            "target_copy_index": target_copy,
            "action_copy_index": edge.get("action_copy_index"),
            "orbit_id": edge.get("orbit_id"),
            "target_mode": str(geometry["mode"]),
            "reference_basis": str(
                edge.get("reference_basis", "compiled_input")
            ),
            "expected_heavy_atoms": expected_atoms,
            "matched_heavy_atoms": matched_atoms,
            "atom_completeness": completeness,
            "runtime_provenance_remapped": bool(
                left_remapped or right_remapped
            ),
            "left_runtime_remapped_heavy_atoms": left_remapped,
            "right_runtime_remapped_heavy_atoms": right_remapped,
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
                    left_available_residues = {
                        (atom.chain_id, atom.residue_number)
                        for atom in left_evaluation_atoms
                    }
                    right_available_residues = {
                        (atom.chain_id, atom.residue_number)
                        for atom in right_evaluation_atoms
                    }
                else:
                    left_contact_residues = set()
                    right_contact_residues = set()
                    left_available_residues = set()
                    right_available_residues = set()
                left_available = len(left_available_residues)
                right_available = len(right_available_residues)
                left_contiguous_capacity = _longest_contiguous_residue_run(
                    left_available_residues
                )
                right_contiguous_capacity = _longest_contiguous_residue_run(
                    right_available_residues
                )
                automatic_coverage, automatic_continuity = (
                    _automatic_interface_targets(
                        left_available,
                        right_available,
                        left_contiguous_capacity=left_contiguous_capacity,
                        right_contiguous_capacity=right_contiguous_capacity,
                    )
                )
                minimum_coverage = int(
                    coverage.get("minimum_contact_residues_per_side")
                    or automatic_coverage
                )
                declared_minimum_contiguous = coverage.get(
                    "minimum_contiguous_contact_residues_per_side"
                )
                minimum_contiguous = int(
                    automatic_continuity
                    if declared_minimum_contiguous is None
                    else declared_minimum_contiguous
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
                packing_metrics = (
                    _heavy_atom_packing_metrics(
                        left_evaluation_atoms,
                        right_evaluation_atoms,
                        distances,
                        cutoff=cutoff,
                    )
                    if (
                        distances is not None
                        and left_evaluation_atoms is not None
                        and right_evaluation_atoms is not None
                    )
                    else {
                        "reciprocal_contact_residue_pair_count": 0,
                        "reciprocal_contact_density": 0.0,
                        "heavy_atom_burial_proxy": 0.0,
                        "contact_depth_mean": 0.0,
                        "contact_depth_standard_deviation": 0.0,
                        "contact_island_count_left": 0,
                        "contact_island_count_right": 0,
                        "local_contact_void_fraction_proxy": 1.0,
                        "hydrophobic_contact_residue_fraction_proxy": 0.0,
                        "unpaired_hydrophobic_residue_fraction_proxy": 0.0,
                    }
                )
                minimum_residue_pairs = minimum_coverage
                maximum_contact_islands = max(
                    1,
                    minimum_coverage - minimum_contiguous + 1,
                )
                maximum_depth_standard_deviation = max(2.0, 0.45 * cutoff)
                packing_quality_satisfied = bool(
                    packing_metrics[
                        "reciprocal_contact_residue_pair_count"
                    ]
                    >= minimum_residue_pairs
                    and packing_metrics["reciprocal_contact_density"] >= 1.0
                    and packing_metrics[
                        "contact_depth_standard_deviation"
                    ]
                    <= maximum_depth_standard_deviation
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
                        "available_contiguous_residues_left": (
                            left_contiguous_capacity
                        ),
                        "available_contiguous_residues_right": (
                            right_contiguous_capacity
                        ),
                        "maximum_contiguous_contact_residues_left": (
                            left_contiguous
                        ),
                        "maximum_contiguous_contact_residues_right": (
                            right_contiguous
                        ),
                        "contact_continuity_satisfied": continuity_satisfied,
                        **packing_metrics,
                        "minimum_reciprocal_contact_residue_pairs": (
                            minimum_residue_pairs
                        ),
                        "maximum_contact_islands_per_side": (
                            maximum_contact_islands
                        ),
                        "maximum_contact_depth_standard_deviation": (
                            maximum_depth_standard_deviation
                        ),
                        "output_packing_quality_satisfied": (
                            packing_quality_satisfied
                        ),
                    }
                )
                checks.extend(
                    (
                        coverage_satisfied,
                        continuity_satisfied,
                        packing_quality_satisfied,
                    )
                )
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
    hyperedge_ids = sorted(
        {report["hyperedge_id"] for report in edge_reports}
    )
    hyperedge_reports = []
    for hyperedge_id in hyperedge_ids:
        members = [
            report
            for report in edge_reports
            if report["hyperedge_id"] == hyperedge_id
        ]
        physical_members: dict[
            tuple[str | None, int], list[dict[str, Any]]
        ] = {}
        for member in members:
            action_copy = member.get("action_copy_index")
            physical_index = int(
                member["source_copy_index"]
                if action_copy is None
                else action_copy
            )
            physical_key = (member.get("orbit_id"), physical_index)
            physical_members.setdefault(physical_key, []).append(member)
        hyperedge_reports.append({
            "hyperedge_id": hyperedge_id,
            "physical_instance_count": len(physical_members),
            "member_edge_instance_count": len(members),
            "members_per_physical_instance": sorted(
                {len(group) for group in physical_members.values()}
            ),
            "required": any(member["required"] for member in members),
            "satisfied": all(
                member["satisfied"]
                for member in members
                if member["required"]
            ),
            "physical_instances": [
                {
                    "orbit_id": orbit_id,
                    "action_copy_index": physical_index,
                    "member_edge_instance_ids": sorted(
                        member["edge_instance_id"]
                        for member in group
                    ),
                    "satisfied": all(
                        member["satisfied"]
                        for member in group
                        if member["required"]
                    ),
                }
                for (orbit_id, physical_index), group in sorted(
                    physical_members.items(),
                    key=lambda item: (str(item[0][0]), item[0][1]),
                )
            ],
            "source_interface_ids": sorted(
                {member["source_interface_id"] for member in members}
            ),
        })
    output_packing_reports = [
        report
        for report in edge_reports
        if report.get("satisfaction_stage") == "output"
        and "output_packing_quality_satisfied" in report
    ]
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
            "interface_hyperedge_count": len(hyperedge_ids),
            "edge_instance_count": len(edge_reports),
            "required_edge_instance_count": len(required_reports),
            "satisfied_required_edge_instance_count": (
                len(required_reports) - len(failed_required)
            ),
            "failed_required_edge_instances": failed_required,
            "output_packing_edge_count": len(output_packing_reports),
            "output_packing_quality_satisfied": bool(
                output_packing_reports
                and all(
                    report["output_packing_quality_satisfied"]
                    for report in output_packing_reports
                )
            ),
            "minimum_reciprocal_contact_residue_pairs": (
                min(
                    report["reciprocal_contact_residue_pair_count"]
                    for report in output_packing_reports
                )
                if output_packing_reports
                else None
            ),
            "minimum_reciprocal_contact_density": (
                min(
                    report["reciprocal_contact_density"]
                    for report in output_packing_reports
                )
                if output_packing_reports
                else None
            ),
            "maximum_contact_depth_standard_deviation": (
                max(
                    report["contact_depth_standard_deviation"]
                    for report in output_packing_reports
                )
                if output_packing_reports
                else None
            ),
            "maximum_local_contact_void_fraction_proxy": (
                max(
                    report["local_contact_void_fraction_proxy"]
                    for report in output_packing_reports
                )
                if output_packing_reports
                else None
            ),
        },
        "interface_hyperedges": hyperedge_reports,
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
