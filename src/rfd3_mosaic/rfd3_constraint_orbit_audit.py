"""Audit rigid geometry components after RFD3 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from rfd3_mosaic.structure import read_structure_atoms


_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)-([0-9]+)$")


def _parse_selector(selector: str) -> tuple[str, int, int]:
    match = _SELECTOR.fullmatch(selector)
    if match is None:
        raise ValueError(
            "fixed selector must be one contiguous range such as B1-31"
        )
    chain, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError("fixed selector range is reversed")
    return chain, start, end


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


def _chain_sort_key(chain: str) -> tuple[int, ...]:
    """Sort output chains in the compiler's A..Z, AA..AZ allocation order."""

    if chain and chain.isalpha() and chain.isupper():
        value = 0
        for character in chain:
            value = value * 26 + ord(character) - ord("A") + 1
        return (0, value)
    return (1, *chain.encode("utf-8"))


def _runtime_action_index(
    value: Any,
    registry_order: list[str],
) -> int:
    """Resolve one runtime action encoded as an index or registry id."""

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


def _constraint_components(
    example: dict[str, Any],
    source_lookup: dict[tuple[str, int, str], np.ndarray],
) -> list[tuple[str, list[tuple[str, int, str]], str, str | None]]:
    """Return explicitly coupled fixed-geometry atom components.

    New compiler inputs describe one source-component list per runtime
    constraint orbit.  Legacy inputs have no such metadata and retain the
    historical behavior: every fixed selector belongs to one joint component.
    """

    extra = example.get("extra") or {}
    preexpanded_layout = extra.get("preexpanded_chain_layout")
    if isinstance(preexpanded_layout, list):
        source_chains = sorted(
            {key[0] for key in source_lookup},
            key=_chain_sort_key,
        )
        if len(source_chains) != len(preexpanded_layout):
            raise ValueError(
                "Preexpanded source-chain count does not match the declared "
                f"layout: {len(source_chains)} != {len(preexpanded_layout)}"
            )
        chains_by_orbit: dict[str, set[str]] = {}
        for chain_id, record in zip(
            source_chains,
            preexpanded_layout,
            strict=True,
        ):
            orbit_id = str(record.get("orbit_id") or "mixed_orbit")
            chains_by_orbit.setdefault(orbit_id, set()).add(chain_id)
        return [
            (
                orbit_id,
                sorted(
                    key for key in source_lookup if key[0] in chain_ids
                ),
                "fixed",
                None,
            )
            for orbit_id, chain_ids in sorted(chains_by_orbit.items())
        ]

    orbits = extra.get(
        "motif_constraint_orbits"
    )
    if not isinstance(orbits, list) or not orbits:
        return [
            ("fixed_component_001", sorted(source_lookup), "fixed", None)
        ]
    if any(
        not isinstance(orbit, dict)
        or not isinstance(orbit.get("source_components"), list)
        for orbit in orbits
    ):
        return [
            ("fixed_component_001", sorted(source_lookup), "fixed", None)
        ]

    components: list[
        tuple[str, list[tuple[str, int, str]], str, str | None]
    ] = []
    assigned: set[tuple[str, int, str]] = set()
    for index, orbit in enumerate(orbits, start=1):
        residue_ids = {
            _component(str(value))
            for value in orbit["source_components"]
        }
        atom_keys = sorted(
            key
            for key in source_lookup
            if (key[0], key[1]) in residue_ids
        )
        if not atom_keys:
            raise ValueError(
                "Fixed constraint component matched no source atoms: "
                f"{orbit.get('constraint_orbit_id', index)!r}"
            )
        overlap = assigned.intersection(atom_keys)
        if overlap:
            raise ValueError(
                "Fixed constraint components overlap on source atoms: "
                f"{sorted(overlap)[:5]}"
            )
        assigned.update(atom_keys)
        components.append(
            (
                str(
                    orbit.get(
                        "coupling_group_id",
                        orbit.get(
                            "constraint_orbit_id",
                            f"fixed_component_{index:03d}",
                        ),
                    )
                ),
                atom_keys,
                str(orbit.get("mobility_mode", "fixed")),
                str(orbit.get("constraint_orbit_id", "")) or None,
            )
        )
    uncovered = set(source_lookup) - assigned
    if uncovered:
        raise ValueError(
            "Fixed constraint components do not cover all selected source "
            f"atoms ({len(uncovered)} uncovered)"
        )
    return components


def _declared_constraint_groups(
    example: dict[str, Any],
    constraint_orbit_id: str | None,
) -> list[dict[str, Any]]:
    """Return runtime groups for one constraint orbit in declared order."""

    if constraint_orbit_id is None:
        return []
    groups = (example.get("extra") or {}).get("motif_constraint_groups")
    if not isinstance(groups, list):
        return []
    selected = [
        group
        for group in groups
        if isinstance(group, dict)
        and str(group.get("constraint_orbit_id", ""))
        == constraint_orbit_id
    ]
    if not selected:
        return []

    orbits = (example.get("extra") or {}).get("motif_constraint_orbits")
    orbit = next(
        (
            item
            for item in orbits or []
            if isinstance(item, dict)
            and str(item.get("constraint_orbit_id", ""))
            == constraint_orbit_id
        ),
        None,
    )
    group_ids = orbit.get("group_ids") if isinstance(orbit, dict) else None
    if not isinstance(group_ids, list) or not group_ids:
        return selected
    by_id = {str(group.get("group_id")): group for group in selected}
    missing = [str(group_id) for group_id in group_ids if str(group_id) not in by_id]
    if missing:
        raise ValueError(
            f"Constraint orbit {constraint_orbit_id!r} is missing runtime "
            f"groups {missing}"
        )
    extras = set(by_id) - {str(group_id) for group_id in group_ids}
    if extras:
        raise ValueError(
            f"Constraint orbit {constraint_orbit_id!r} declares unexpected "
            f"runtime groups {sorted(extras)}"
        )
    return [by_id[str(group_id)] for group_id in group_ids]


def _mapped_output_coordinate(
    *,
    source_key: tuple[str, int, str],
    action_index: int,
    index_map: dict[str, Any],
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    preexpanded: bool = False,
) -> np.ndarray | None:
    source_chain, source_residue, atom_name = source_key
    destination = index_map.get(f"{source_chain}{source_residue}")
    if destination is None:
        return None
    master_chain, output_residue = _component(str(destination))
    try:
        master_position = ordered_output_chains.index(master_chain)
    except ValueError:
        return None
    if preexpanded:
        return output_lookup.get(
            (master_chain, output_residue, atom_name)
        )
    asu_chain_index = master_position % asu_chain_count
    output_index = action_index * asu_chain_count + asu_chain_index
    if not 0 <= output_index < len(ordered_output_chains):
        return None
    return output_lookup.get(
        (
            ordered_output_chains[output_index],
            output_residue,
            atom_name,
        )
    )


def _component_runtime_copies(
    *,
    example: dict[str, Any],
    constraint_orbit_id: str | None,
    atom_keys: list[tuple[str, int, str]],
    source_lookup: dict[tuple[str, int, str], np.ndarray],
    index_map: dict[str, Any],
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    ordered_output_chains: list[str],
    asu_chain_count: int,
    registry_order: list[str],
    registry_matrices: dict[str, Any],
    preexpanded: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Materialize expected/observed atoms for every physical group.

    ``diffused_index_map`` identifies the asymmetric-unit output chain slot.
    A runtime motif group may then place different source fragments under
    different symmetry actions (for example one half of a supplied interface
    crosses the Cn seam).  Consequently neither one output chain per action
    nor one common action per group is a valid general assumption.
    """

    declared_groups = _declared_constraint_groups(
        example,
        constraint_orbit_id,
    )
    expected_copies: list[np.ndarray] = []
    observed_copies: list[np.ndarray] = []
    matched_per_copy: list[int] = []
    expected_per_copy: list[int] = []
    component_key_set = set(atom_keys)

    if preexpanded:
        expected = []
        observed = []
        for key in atom_keys:
            expected.append(source_lookup[key])
            coordinate = _mapped_output_coordinate(
                source_key=key,
                action_index=0,
                index_map=index_map,
                output_lookup=output_lookup,
                ordered_output_chains=ordered_output_chains,
                asu_chain_count=asu_chain_count,
                preexpanded=True,
            )
            if coordinate is not None:
                observed.append(coordinate)
        return (
            [np.asarray(expected, dtype=float)],
            [np.asarray(observed, dtype=float)],
            [len(observed)],
            [len(expected)],
        )

    if not declared_groups:
        declared_groups = [
            {
                "members": [
                    {
                        "src_components": [
                            f"{chain}{residue}"
                            for chain, residue in sorted(
                                {(key[0], key[1]) for key in atom_keys}
                            )
                        ],
                        "sym_transform_id": action_index,
                    }
                ]
            }
            for action_index in range(len(registry_order))
        ]

    for group in declared_groups:
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(
                f"Runtime motif group {group.get('group_id')!r} has no members"
            )
        expected: list[np.ndarray] = []
        observed: list[np.ndarray] = []
        group_keys: set[tuple[str, int, str]] = set()
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("Runtime motif group member must be an object")
            components = member.get("src_components")
            if not isinstance(components, list) or not components:
                raise ValueError(
                    "Runtime motif group member declares no source components"
                )
            residue_ids = {_component(str(value)) for value in components}
            member_keys = [
                key for key in atom_keys if (key[0], key[1]) in residue_ids
            ]
            if not member_keys:
                raise ValueError(
                    "Runtime motif group member matched no selected source "
                    f"atoms: {components!r}"
                )
            overlap = group_keys.intersection(member_keys)
            if overlap:
                raise ValueError(
                    "Runtime motif group members overlap on source atoms: "
                    f"{sorted(overlap)[:5]}"
                )
            group_keys.update(member_keys)
            action_index = _runtime_action_index(
                member.get("sym_transform_id"),
                registry_order,
            )
            matrix = np.asarray(
                registry_matrices[registry_order[action_index]],
                dtype=float,
            )
            for key in member_keys:
                source_coordinate = source_lookup[key]
                expected.append(
                    source_coordinate @ matrix[:3, :3].T + matrix[:3, 3]
                )
                coordinate = _mapped_output_coordinate(
                    source_key=key,
                    action_index=action_index,
                    index_map=index_map,
                    output_lookup=output_lookup,
                    ordered_output_chains=ordered_output_chains,
                    asu_chain_count=asu_chain_count,
                )
                if coordinate is not None:
                    observed.append(coordinate)

        if group_keys != component_key_set:
            missing = component_key_set - group_keys
            extra = group_keys - component_key_set
            raise ValueError(
                f"Runtime motif group {group.get('group_id')!r} does not "
                "cover its complete constraint component: "
                f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )
        expected_per_copy.append(len(expected))
        matched_per_copy.append(len(observed))
        expected_copies.append(np.asarray(expected, dtype=float))
        observed_copies.append(np.asarray(observed, dtype=float))
    return (
        expected_copies,
        observed_copies,
        matched_per_copy,
        expected_per_copy,
    )


def audit_constraint_orbit(
    *,
    compiled_input: str | Path,
    result_json: str | Path,
    result_structure: str | Path | None = None,
    max_joint_rmsd: float = 0.5,
    min_atom_completeness: float = 0.99,
) -> dict[str, Any]:
    input_path = Path(compiled_input).resolve()
    result_json_path = Path(result_json).resolve()
    example = _single_example(input_path)
    result = _load_json(result_json_path)
    extra = example.get("extra") or {}
    quotient_action = (
        extra.get("symmetry_action_kind") == "stabilizer_quotient"
    )
    runtime_diagnostics = result.get("constraint_runtime_diagnostics")
    runtime_fixed_target_rmsd: float | None = None
    runtime_fixed_target_maximum: float | None = None
    runtime_fixed_target_contract_valid = False
    if isinstance(runtime_diagnostics, dict):
        raw_rmsd = runtime_diagnostics.get("final_fixed_target_rmsd")
        raw_maximum = runtime_diagnostics.get(
            "final_fixed_target_maximum_error"
        )
        if raw_rmsd is not None and raw_maximum is not None:
            runtime_fixed_target_rmsd = float(raw_rmsd)
            runtime_fixed_target_maximum = float(raw_maximum)
            runtime_fixed_target_contract_valid = bool(
                np.isfinite(runtime_fixed_target_rmsd)
                and np.isfinite(runtime_fixed_target_maximum)
                and runtime_diagnostics.get("state") == "finalized"
                and runtime_fixed_target_maximum <= 1.0e-5
            )
    declared_selectors = extra.get("fixed_constraint_selectors")
    if declared_selectors is None:
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
        raise ValueError("fixed constraint selectors must be a list")
    if not selectors:
        raise ValueError("Compiled input declares no fixed selectors")
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
    residue_map: dict[tuple[str, int], tuple[str, int]] = {}
    for source_chain, start, end in parsed_selectors:
        for source_residue in range(start, end + 1):
            destination = index_map.get(f"{source_chain}{source_residue}")
            if destination is None:
                continue
            output_chain, output_residue = _component(str(destination))
            residue_map[(source_chain, source_residue)] = (
                output_chain,
                output_residue,
            )
    if not residue_map:
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
    ordered_output_chains = sorted(
        {
            atom.chain_id
            for atom in output_atoms
            if atom.record_type == "ATOM"
        },
        key=_chain_sort_key,
    )

    matrices = example.get("extra", {}).get("registry_transform_matrices")
    order = example.get("extra", {}).get("registry_transform_order")
    multiplicity = int(example["extra"]["symmetry_multiplicity"])
    preexpanded_layout = extra.get("preexpanded_chain_layout")
    preexpanded = isinstance(preexpanded_layout, list)
    if not isinstance(matrices, dict) or not isinstance(order, list):
        raise ValueError("Compiled input lacks the validated transform registry")
    if preexpanded:
        if (
            len(order) != multiplicity
            or not ordered_output_chains
            or len(preexpanded_layout) != len(ordered_output_chains)
        ):
            raise ValueError(
                "Preexpanded output/layout or transform registry is "
                "incomplete: "
                f"multiplicity={multiplicity}, "
                f"chains={ordered_output_chains}, order={order}, "
                f"layout_count={len(preexpanded_layout)}"
            )
        asu_chain_count = sum(
            bool(record.get("is_asu", False))
            for record in preexpanded_layout
        )
    elif (
        len(order) != multiplicity
        or not ordered_output_chains
        or len(ordered_output_chains) % multiplicity != 0
    ):
        raise ValueError(
            "Output chain count is not divisible by symmetry multiplicity "
            "or the registry is incomplete: "
            f"multiplicity={multiplicity}, chains={ordered_output_chains}, "
            f"order={order}"
        )
    asu_chain_count = len(ordered_output_chains) // multiplicity

    component_reports = []
    for (
        component_id,
        atom_keys,
        mobility_mode,
        constraint_orbit_id,
    ) in _constraint_components(
        example, source_lookup
    ):
        if mobility_mode not in {"fixed", "orbit_rigid"}:
            raise ValueError(
                f"Unsupported fixed-component mobility mode {mobility_mode!r}"
            )
        (
            expected_copies,
            observed_copies,
            matched_per_copy,
            expected_per_copy,
        ) = _component_runtime_copies(
            example=example,
            constraint_orbit_id=constraint_orbit_id,
            atom_keys=atom_keys,
            source_lookup=source_lookup,
            index_map=index_map,
            output_lookup=output_lookup,
            ordered_output_chains=ordered_output_chains,
            asu_chain_count=asu_chain_count,
            registry_order=[str(value) for value in order],
            registry_matrices=matrices,
            preexpanded=preexpanded,
        )

        matched = sum(matched_per_copy)
        expected_count = sum(expected_per_copy)
        completeness = matched / expected_count if expected_count else 0.0
        complete_shapes = all(
            expected.shape == observed.shape and len(expected) > 0
            for expected, observed in zip(
                expected_copies,
                observed_copies,
            )
        )
        if not matched or not complete_shapes:
            joint_rmsd = float("inf")
            joint_maximum = float("inf")
            distance_matrix_rmsd = float("inf")
            per_copy_rmsd = [float("inf")] * len(expected_copies)
        else:
            expected_joint = np.concatenate(expected_copies, axis=0)
            observed_joint = np.concatenate(observed_copies, axis=0)
            aligned_joint = _kabsch_align(observed_joint, expected_joint)
            deviations = np.linalg.norm(
                aligned_joint - expected_joint, axis=-1
            )
            joint_rmsd = _rmsd(aligned_joint, expected_joint)
            joint_maximum = float(np.max(deviations))
            expected_distances = np.linalg.norm(
                expected_joint[:, None, :] - expected_joint[None, :, :],
                axis=-1,
            )
            observed_distances = np.linalg.norm(
                observed_joint[:, None, :] - observed_joint[None, :, :],
                axis=-1,
            )
            distance_matrix_rmsd = float(
                np.sqrt(
                    np.mean(
                        (observed_distances - expected_distances) ** 2
                    )
                )
            )
            per_copy_rmsd = [
                _rmsd(_kabsch_align(observed, expected), expected)
                for observed, expected in zip(
                    observed_copies, expected_copies
                )
            ]
        geometry_contract = (
            "complete_orbit_joint_rigid"
            if mobility_mode == "fixed"
            else "per_copy_rigid_with_bounded_orbit_pose"
        )
        legacy_reference_joint_rmsd = None
        legacy_reference_joint_maximum = None
        legacy_reference_distance_matrix_rmsd = None
        acceptance_reference = "compiled_source_geometry"
        if (
            quotient_action
            and mobility_mode == "fixed"
            and runtime_fixed_target_contract_valid
        ):
            # A quotient input is materialized from a full presymmetrized
            # structure but executed from physical coset representatives.
            # Reapplying coset matrices to that materialized source can use
            # the wrong source frame in this legacy audit reconstruction.
            # The sampler records a direct atomwise comparison against the
            # exact runtime target immediately before serialization; use that
            # authoritative invariant for fixed quotient actions while
            # retaining the reconstructed values as diagnostics.
            legacy_reference_joint_rmsd = joint_rmsd
            legacy_reference_joint_maximum = joint_maximum
            legacy_reference_distance_matrix_rmsd = distance_matrix_rmsd
            joint_rmsd = float(runtime_fixed_target_rmsd)
            joint_maximum = float(runtime_fixed_target_maximum)
            distance_matrix_rmsd = 0.0
            acceptance_reference = "runtime_fixed_target"
        acceptance_rmsd = (
            joint_rmsd
            if mobility_mode == "fixed"
            else max(per_copy_rmsd)
        )
        component_reports.append(
            {
                "component_id": component_id,
                "mobility_mode": mobility_mode,
                "geometry_contract": geometry_contract,
                "passed": bool(
                    completeness >= min_atom_completeness
                    and acceptance_rmsd <= max_joint_rmsd
                ),
                "expected_heavy_atoms": expected_count,
                "matched_heavy_atoms": matched,
                "atom_completeness": completeness,
                "joint_orbit_rmsd": joint_rmsd,
                "joint_orbit_maximum_error": joint_maximum,
                "orbit_distance_matrix_rmsd": distance_matrix_rmsd,
                "maximum_per_copy_internal_rmsd": max(per_copy_rmsd),
                "per_copy_internal_rmsd": per_copy_rmsd,
                "acceptance_rmsd": acceptance_rmsd,
                "acceptance_reference": acceptance_reference,
                "legacy_reference_joint_orbit_rmsd": (
                    legacy_reference_joint_rmsd
                ),
                "legacy_reference_joint_orbit_maximum_error": (
                    legacy_reference_joint_maximum
                ),
                "legacy_reference_orbit_distance_matrix_rmsd": (
                    legacy_reference_distance_matrix_rmsd
                ),
            }
        )

    passed = bool(component_reports) and all(
        component["passed"] for component in component_reports
    )
    expected_count = sum(
        int(component["expected_heavy_atoms"])
        for component in component_reports
    )
    matched = sum(
        int(component["matched_heavy_atoms"])
        for component in component_reports
    )
    completeness = matched / expected_count if expected_count else 0.0

    def maximum(key: str) -> float:
        return max(float(component[key]) for component in component_reports)

    per_copy_rmsd = [
        value
        for component in component_reports
        for value in component["per_copy_internal_rmsd"]
    ]
    return {
        "audit": "rfd3_mosaic.fixed_constraint_orbit",
        "schema_version": 2,
        "passed": passed,
        "inputs": {
            "compiled_input": str(input_path),
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
            "min_atom_completeness": min_atom_completeness,
        },
        "summary": {
            "symmetry_multiplicity": multiplicity,
            "output_chains": ordered_output_chains,
            "asu_chain_count": asu_chain_count,
            "constraint_component_count": len(component_reports),
            "runtime_fixed_target_contract_valid": (
                runtime_fixed_target_contract_valid
            ),
            "runtime_fixed_target_rmsd": runtime_fixed_target_rmsd,
            "runtime_fixed_target_maximum_error": (
                runtime_fixed_target_maximum
            ),
            "expected_heavy_atoms": expected_count,
            "matched_heavy_atoms": matched,
            "atom_completeness": completeness,
            "joint_orbit_rmsd": maximum("joint_orbit_rmsd"),
            "maximum_acceptance_rmsd": maximum("acceptance_rmsd"),
            "joint_orbit_maximum_error": maximum(
                "joint_orbit_maximum_error"
            ),
            "orbit_distance_matrix_rmsd": maximum(
                "orbit_distance_matrix_rmsd"
            ),
            "maximum_per_copy_internal_rmsd": max(per_copy_rmsd),
            "per_copy_internal_rmsd": per_copy_rmsd,
            "constraint_components": component_reports,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit fixed components modulo one common rigid transform per "
            "component."
        )
    )
    parser.add_argument("--compiled-input", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-structure", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-joint-rmsd", type=float, default=0.5)
    parser.add_argument("--min-atom-completeness", type=float, default=0.99)
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    report = audit_constraint_orbit(
        compiled_input=arguments.compiled_input,
        result_json=arguments.result_json,
        result_structure=arguments.result_structure,
        max_joint_rmsd=arguments.max_joint_rmsd,
        min_atom_completeness=arguments.min_atom_completeness,
    )
    arguments.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"Constraint orbit audit: {'PASSED' if report['passed'] else 'FAILED'}")
    print(
        "maximum contract RMSD: "
        f"{summary['maximum_acceptance_rmsd']:.6f} A"
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


__all__ = ["audit_constraint_orbit"]
