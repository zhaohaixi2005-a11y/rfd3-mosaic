"""Compile standalone Interface-Seed artifacts into an RFD3 input JSON."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from rfd3_mosaic.compile import (
    expand_symmetry_instances,
    load_interface_seed_config,
)
from rfd3_mosaic.geometry import (
    build_transform_registry,
    validate_group_closure,
)
from rfd3_mosaic.output.standalone import compile_standalone
from rfd3_mosaic.schema import (
    ScaffoldLinkInstance,
    TerminalExtensionInstance,
)
from rfd3_mosaic.schema.specs import SymmetryType
from rfd3_mosaic.topology import (
    analyze_interleaved_interface_seed_topology,
)


@dataclass(frozen=True)
class RFD3AdapterOutputs:
    """Artifacts needed to prevalidate and run one native RFD3 design."""

    input_path: Path
    structure_path: Path
    mapping_path: Path
    manifest_path: Path
    example_id: str
    contig: str


@dataclass(frozen=True)
class _MaterializedScaffoldLink:
    """One link after choosing its concrete RFD3 linker length."""

    link: Any
    materialized_linker_length: int | None
    linker_length_policy: str
    contour_preflight: dict[str, Any]


@dataclass(frozen=True)
class _ASUScaffoldSegment:
    """One ordered fixed-fragment path emitted as one or more ASU chains."""

    links: tuple[_MaterializedScaffoldLink, ...]
    fragment_instance_ids: tuple[str, ...]
    selectors: tuple[str, ...]
    from_selector: str
    to_selector: str
    contig_chains: tuple[str, ...]
    materialized_linker_length: int | None
    linker_length_policy: str
    contour_preflight: dict[str, Any]


@dataclass(frozen=True)
class _ASUTerminalPath:
    """One motif with generated N/C flanks emitted as one ASU chain."""

    anchor_fragment_instance_id: str
    selector: str
    orbit_id: str
    n_extension: TerminalExtensionInstance | None
    c_extension: TerminalExtensionInstance | None
    contig: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_sample_overrides(
    manifest_path: Path,
    *,
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], int | None, str]:
    """Recover exact joint unit samples and validate config provenance."""

    manifest = _load_json(manifest_path)
    candidate_config = manifest.get("config", {})
    expected_config_sha = candidate_config.get("sha256")
    actual_config_sha = _sha256(config_path)
    if expected_config_sha != actual_config_sha:
        raise ValueError(
            "Pose candidate config SHA256 does not match the requested "
            f"config: {expected_config_sha!r} != {actual_config_sha!r}"
        )
    initialization_samples = manifest.get("initialization_samples", {})
    if not initialization_samples:
        raise ValueError("Pose candidate manifest has no initialization samples")
    overrides: dict[str, dict[str, Any]] = {}
    for group_id, sample in initialization_samples.items():
        unit_samples = sample.get("unit_samples")
        if not isinstance(unit_samples, dict):
            raise ValueError(
                f"Pose candidate group {group_id!r} lacks unit samples"
            )
        so3_unit = unit_samples.get("so3")
        if so3_unit is None:
            raise ValueError(
                f"Pose candidate group {group_id!r} lacks SO(3) unit samples"
            )
        overrides[group_id] = {
            "radius_unit": unit_samples.get("radius"),
            "axial_offset_unit": unit_samples.get("axial_offset"),
            "so3_unit": so3_unit,
        }
    expected_structure_sha = manifest.get("outputs", {}).get(
        "structure", {}
    ).get("sha256")
    if not expected_structure_sha:
        raise ValueError("Pose candidate manifest lacks structure SHA256")
    effective_seed = candidate_config.get("effective_random_seed")
    return overrides, effective_seed, str(expected_structure_sha)


def _fragment_selector(
    mapping: dict[str, Any],
    fragment_instance_id: str,
) -> str:
    """Return an RFD3 label-chain residue selector for one fragment copy.

    AtomWorks indexes mmCIF selections with ``label_seq_id``.  Original PDB
    residue numbers remain available under the mapping's ``source`` records
    for provenance, but using those author numbers in an RFD3 contig makes the
    motif unresolvable after CIF parsing.
    """

    records = [
        record
        for record in mapping["atom_mappings"]
        if record["instance"]["fragment_instance_id"]
        == fragment_instance_id
    ]
    if not records:
        raise ValueError(
            f"No atom mappings found for fragment {fragment_instance_id!r}"
        )

    compiled_chains = {
        record["compiled"]["chain_id"] for record in records
    }
    if len(compiled_chains) != 1:
        raise ValueError(
            f"Fragment {fragment_instance_id!r} spans multiple output chains"
        )
    chain_id = next(iter(compiled_chains))
    if len(chain_id) != 1:
        raise NotImplementedError(
            "The installed AtomWorks contig parser accepts only one-letter "
            f"chain identifiers, but fragment {fragment_instance_id!r} was "
            f"assigned {chain_id!r}. Reduce the number of materialized "
            "source fragments or use a local-neighbourhood backend that does "
            "not preexpand every physical component"
        )
    residue_numbers = sorted(
        {
            int(record["compiled"]["label_seq_id"])
            for record in records
        }
    )
    expected = list(range(residue_numbers[0], residue_numbers[-1] + 1))
    if residue_numbers != expected:
        raise ValueError(
            f"RFD3 indexed motifs must be residue-contiguous; "
            f"{fragment_instance_id!r} is not"
        )
    return f"{chain_id}{residue_numbers[0]}-{residue_numbers[-1]}"


def _rfd3_atom_selection(fixed_atoms: str | list[str] | None) -> str:
    if fixed_atoms is None:
        return "ALL"
    if isinstance(fixed_atoms, list):
        if not fixed_atoms:
            raise ValueError("fixed_atoms lists cannot be empty")
        return ",".join(fixed_atoms)
    aliases = {
        "all": "ALL",
        "backbone": "BKBN",
        "bkbn": "BKBN",
        "tip": "TIP",
    }
    return aliases.get(fixed_atoms.lower(), fixed_atoms)


def _selector_source_components(selector: str) -> list[str]:
    """Expand one contiguous RFD3 selector into residue source components."""

    match = re.fullmatch(r"([A-Za-z]+)(\d+)-(\d+)", selector)
    if match is None:
        raise ValueError(
            f"Runtime motif groups require a contiguous selector, got "
            f"{selector!r}"
        )
    chain_id, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"Selector range is reversed: {selector!r}")
    return [f"{chain_id}{residue_id}" for residue_id in range(start, end + 1)]


def _native_symmetry_id_and_multiplicity(transform_set: Any) -> tuple[str, int]:
    if transform_set.type == SymmetryType.CYCLIC:
        symmetry_id = f"C{transform_set.order}"
        multiplicity = transform_set.order
    elif transform_set.type == SymmetryType.DIHEDRAL:
        symmetry_id = f"D{transform_set.order}"
        multiplicity = 2 * transform_set.order
    elif transform_set.type == SymmetryType.TETRAHEDRAL:
        symmetry_id = "T"
        multiplicity = 12
    elif transform_set.type == SymmetryType.OCTAHEDRAL:
        symmetry_id = "O"
        multiplicity = 24
    elif transform_set.type == SymmetryType.ICOSAHEDRAL:
        symmetry_id = "I"
        multiplicity = 60
    else:
        raise NotImplementedError(
            f"Native RFD3 symmetry does not support "
            f"{transform_set.type.value!r}"
        )
    return symmetry_id, multiplicity


def _preflight_native_transform_registry(
    transform_set: Any,
    expected_multiplicity: int,
) -> list[str]:
    """Reject incomplete, duplicated, improper, or non-closed finite sets."""

    registry = build_transform_registry(transform_set)

    if registry.order != expected_multiplicity:
        raise ValueError(
            f"Transform registry {registry.group_name} contains "
            f"{registry.order} transforms; expected {expected_multiplicity}"
        )
    validate_group_closure(registry)
    matrices = [registry.transform(item) for item in registry.transform_ids]
    for index, matrix in enumerate(matrices):
        if not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6):
            raise ValueError(
                f"Transform {registry.transform_ids[index]} is not a "
                "proper rotation"
            )
        for previous in matrices[:index]:
            if np.allclose(matrix, previous, atol=1e-6):
                raise ValueError(
                    f"Transform registry {registry.group_name} contains "
                    "duplicate frames"
                )
    return list(registry.transform_ids)


def _runtime_action_index(
    registry,
    *,
    anchor_transform_id: str,
    target_transform_id: str,
    context: str,
    runtime_transform_order: tuple[str, ...] | list[str] | None = None,
    transform_to_runtime_representative: dict[str, str] | None = None,
) -> int:
    """Resolve the native left action carrying an ASU anchor to a target.

    Compiler instances retain their physical registry transforms, whereas
    RFD3 ``sym_transform_id`` values describe actions relative to the
    materialized ASU.  Keeping this conversion in one helper is essential for
    seam-crossing paths and for non-commutative D/T/O/I registries.
    """

    matching_action_ids = [
        action_id
        for action_id in registry.transform_ids
        if registry.compose_ids(action_id, anchor_transform_id)
        == target_transform_id
    ]
    if len(matching_action_ids) != 1:
        raise ValueError(
            f"Could not resolve one native runtime action for {context}: "
            f"anchor={anchor_transform_id!r}, "
            f"target={target_transform_id!r}, "
            f"matches={matching_action_ids}"
        )
    selected_order = tuple(runtime_transform_order or registry.transform_ids)
    representative_map = transform_to_runtime_representative or {
        transform_id: transform_id
        for transform_id in registry.transform_ids
    }
    action_id = matching_action_ids[0]
    try:
        representative_id = representative_map[action_id]
        return selected_order.index(representative_id)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Runtime action {action_id!r} for {context} does not map to "
            "a declared physical quotient copy"
        ) from error


def _runtime_interface_constraint_groups(
    instances,
    mapping: dict[str, Any],
    links,
    transform_set,
    runtime_transform_order=None,
    transform_to_runtime_representative=None,
    preexpanded_mixed_orbits: bool = False,
) -> list[dict[str, Any]]:
    """Describe complete cross-chain groups in post-symmetry RFD3 terms."""

    selector_by_source_id: dict[str, str] = {}
    selector_by_fragment_instance_id: dict[str, str] = {}
    canonical_transform_by_source_id: dict[str, str] = {}
    for link in links:
        for fragment_instance_id in (
            link.from_fragment_instance_id,
            link.to_fragment_instance_id,
        ):
            fragment = instances.fragments[fragment_instance_id]
            selector = _fragment_selector(mapping, fragment_instance_id)
            selector_by_fragment_instance_id[fragment_instance_id] = selector
            if preexpanded_mixed_orbits:
                # Every physical quotient copy is already a separate input
                # chain.  Repeated source_fragment ids therefore correctly
                # map to different selectors and must not be collapsed into
                # one canonical-ASU selector.
                continue
            previous = selector_by_source_id.get(fragment.source_id)
            if previous is not None and previous != selector:
                raise ValueError(
                    "One source fragment maps to multiple canonical ASU "
                    f"selectors: {fragment.source_id!r}"
                )
            selector_by_source_id[fragment.source_id] = selector
            canonical_transform_by_source_id[
                fragment.source_id
            ] = fragment.transform_id

    registry = build_transform_registry(transform_set)

    def correspondence_components(fragment) -> list[str]:
        if not preexpanded_mixed_orbits:
            return _selector_source_components(
                selector_by_source_id[fragment.source_id]
            )
        masters = [
            candidate
            for candidate in instances.fragments.values()
            if candidate.source_id == fragment.source_id
            and candidate.orbit_id == fragment.orbit_id
            and candidate.copy_index == 0
        ]
        if len(masters) != 1:
            raise ValueError(
                "Mixed-orbit atom correspondence requires exactly one "
                f"copy-zero fragment for {fragment.source_id!r} in "
                f"{fragment.orbit_id!r}; observed {len(masters)}"
            )
        return _selector_source_components(
            _fragment_selector(mapping, masters[0].id)
        )

    def runtime_transform_id(fragment) -> int:
        if preexpanded_mixed_orbits:
            try:
                return tuple(runtime_transform_order).index(
                    fragment.transform_id
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Physical fragment {fragment.id!r} uses transform "
                    f"{fragment.transform_id!r} outside the mixed runtime "
                    "registry"
                ) from error
        canonical_transform_id = canonical_transform_by_source_id[
            fragment.source_id
        ]
        return _runtime_action_index(
            registry,
            anchor_transform_id=canonical_transform_id,
            target_transform_id=fragment.transform_id,
            context=f"interface fragment {fragment.id!r}",
            runtime_transform_order=runtime_transform_order,
            transform_to_runtime_representative=(
                transform_to_runtime_representative
            ),
        )

    groups: list[dict[str, Any]] = []
    for edge in instances.interfaces.values():
        members: list[dict[str, Any]] = []
        for port_instance_id, copy_index, role in (
            (
                edge.left_port_instance_id,
                edge.source_copy_index,
                "left",
            ),
            (
                edge.right_port_instance_id,
                edge.target_copy_index,
                "right",
            ),
        ):
            port = instances.ports[port_instance_id]
            for fragment_instance_id in port.fragment_instance_ids:
                fragment = instances.fragments[fragment_instance_id]
                if preexpanded_mixed_orbits:
                    try:
                        source_component = (
                            selector_by_fragment_instance_id[
                                fragment_instance_id
                            ]
                        )
                    except KeyError as error:
                        raise NotImplementedError(
                            "Every mixed-orbit interface fragment must be "
                            "materialized by one physical scaffold path"
                        ) from error
                else:
                    try:
                        source_component = selector_by_source_id[
                            fragment.source_id
                        ]
                    except KeyError as error:
                        raise NotImplementedError(
                            "Runtime motif groups require every interface "
                            "fragment source to be present in the canonical "
                            "ASU scaffold link"
                        ) from error
                actual_components = _selector_source_components(
                    source_component
                )
                stable_components = correspondence_components(fragment)
                if len(actual_components) != len(stable_components):
                    raise ValueError(
                        f"Physical and canonical selectors for fragment "
                        f"{fragment.id!r} have different residue counts"
                    )
                members.append(
                    {
                        "role": role,
                        "source_fragment_id": fragment.source_id,
                        "src_components": actual_components,
                        "correspondence_components": stable_components,
                        "sym_transform_id": runtime_transform_id(fragment),
                    }
                )
        if not any(member["role"] == "left" for member in members):
            raise ValueError(f"Interface group {edge.id!r} has no left atoms")
        if not any(member["role"] == "right" for member in members):
            raise ValueError(f"Interface group {edge.id!r} has no right atoms")
        groups.append(
            {
                "group_id": edge.id,
                "constraint_kind": "interface",
                "source_interface_id": edge.source_id,
                "hyperedge_id": edge.hyperedge_id or edge.source_id,
                "orbit_id": edge.orbit_id,
                "transform_set_id": edge.transform_set_id,
                "action_copy_index": edge.action_copy_index,
                "action_transform_id": edge.action_transform_id,
                "left_orbit_id": edge.left_orbit_id,
                "right_orbit_id": edge.right_orbit_id,
                "left_transform_index": edge.left_transform_index,
                "right_transform_index": edge.right_transform_index,
                "source_copy_index": edge.source_copy_index,
                "target_copy_index": edge.target_copy_index,
                "members": members,
            }
        )
    return groups


def _runtime_interface_constraint_orbits(
    groups: list[dict[str, Any]],
    *,
    instances,
    transform_set,
    runtime_transform_order=None,
    transform_to_runtime_representative=None,
) -> list[dict[str, Any]]:
    """Resolve one master group and one group action per interface orbit."""

    registry = build_transform_registry(transform_set)
    selected_order = tuple(runtime_transform_order or registry.transform_ids)
    representative_map = transform_to_runtime_representative or {
        transform_id: transform_id
        for transform_id in registry.transform_ids
    }
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}
    for group in groups:
        key = (
            str(group["source_interface_id"]),
            str(group["orbit_id"]),
        )
        grouped.setdefault(key, []).append(group)

    orbits = []
    for (
        source_interface_id,
        symmetry_orbit_id,
    ), orbit_groups in grouped.items():
        cross_orbit = any(
            group.get("left_orbit_id")
            != group.get("right_orbit_id")
            for group in orbit_groups
        )
        if cross_orbit and not all(
            group.get("left_orbit_id")
            != group.get("right_orbit_id")
            and group.get("action_copy_index") is not None
            and group.get("action_transform_id") is not None
            for group in orbit_groups
        ):
            raise ValueError(
                f"Mixed-orbit interface {source_interface_id!r} has "
                "incomplete edge-action provenance"
            )
        masters = [
            group
            for group in orbit_groups
            if int(
                group.get("action_copy_index")
                if group.get("action_copy_index") is not None
                else group["source_copy_index"]
            ) == 0
        ]
        if len(masters) != 1:
            raise ValueError(
                "Each motif constraint orbit requires exactly one copy-zero "
                f"master group, observed {len(masters)} for "
                f"{source_interface_id!r}/{symmetry_orbit_id!r}"
            )
        master = masters[0]

        def member_map(group):
            resolved = {}
            for member in group["members"]:
                key = (
                    member["role"],
                    member["source_fragment_id"],
                )
                if key in resolved:
                    raise ValueError(
                        f"Constraint group {group['group_id']!r} has "
                        f"duplicate member identity {key!r}"
                    )
                resolved[key] = int(member["sym_transform_id"])
            return resolved

        master_members = member_map(master)
        group_transform_ids = []
        group_registry_transform_ids = []
        ordered_groups = sorted(
            orbit_groups,
            key=lambda item: int(
                item.get("action_copy_index")
                if item.get("action_copy_index") is not None
                else item["source_copy_index"]
            ),
        )
        constraint_orbit_id = (
            f"{source_interface_id}__{symmetry_orbit_id}"
        )
        for group in ordered_groups:
            target_members = member_map(group)
            if set(target_members) != set(master_members):
                raise ValueError(
                    "All groups in a motif constraint orbit must contain "
                    "the same member identities"
                )
            if cross_orbit:
                action_id = str(group["action_transform_id"])
                action_index = int(group["action_copy_index"])
                if action_index >= len(registry.transform_ids) or (
                    registry.transform_ids[action_index] != action_id
                ):
                    raise ValueError(
                        f"Constraint group {group['group_id']!r} edge "
                        "action index does not match the transform registry"
                    )
            else:
                matching_actions: dict[int, str] = {}
                for action_id in registry.transform_ids:
                    if all(
                        representative_map[
                            registry.compose_ids(
                                action_id,
                                selected_order[master_transform_id],
                            )
                        ]
                        == selected_order[target_members[member_key]]
                        for member_key, master_transform_id
                        in master_members.items()
                    ):
                        representative_id = representative_map[action_id]
                        matching_actions[
                            selected_order.index(representative_id)
                        ] = representative_id
                if len(matching_actions) != 1:
                    raise ValueError(
                        f"Constraint group {group['group_id']!r} has "
                        f"{len(matching_actions)} compatible group actions; "
                        "expected exactly one"
                    )
                action_index, action_id = next(
                    iter(matching_actions.items())
                )
            group["constraint_orbit_id"] = constraint_orbit_id
            group["orbit_transform_id"] = action_index
            group["orbit_registry_transform_id"] = action_id
            group_transform_ids.append(action_index)
            group_registry_transform_ids.append(action_id)

        if cross_orbit:
            component_orbit_ids = {
                str(group[side])
                for group in orbit_groups
                for side in ("left_orbit_id", "right_orbit_id")
            }
            component_mobilities = [
                instances.constraint_orbits[orbit_id].mobility
                for orbit_id in sorted(component_orbit_ids)
            ]
            mobility_payloads = {
                json.dumps(
                    mobility.model_dump(mode="json"),
                    sort_keys=True,
                )
                for mobility in component_mobilities
            }
            if len(mobility_payloads) != 1:
                raise NotImplementedError(
                    "A mixed-orbit interface requires identical component "
                    "mobility contracts before joint runtime control"
                )
            mobility = component_mobilities[0]
        else:
            mobility = instances.constraint_orbits[
                symmetry_orbit_id
            ].mobility
        bounds = mobility.bounds
        # Preserve legacy sampler overrides when the assembly does not own a
        # schedule explicitly.  New native inputs can move this policy into
        # the orbit; old campaigns continue to inherit sampler-level values.
        schedule = mobility.schedule
        orbits.append(
            {
                "constraint_orbit_id": constraint_orbit_id,
                "source_interface_id": source_interface_id,
                "symmetry_orbit_id": symmetry_orbit_id,
                "master_group_id": master["group_id"],
                "group_ids": [
                    group["group_id"] for group in ordered_groups
                ],
                "group_transform_ids": group_transform_ids,
                "group_registry_transform_ids": (
                    group_registry_transform_ids
                ),
                "mobility_mode": mobility.mode.value,
                "mobility_subspace": (
                    mobility.effective_subspace.value
                    if mobility.effective_subspace is not None
                    else None
                ),
                "mobility_proposal": (
                    mobility.effective_proposal.value
                    if mobility.effective_proposal is not None
                    else None
                ),
                "mobility_objectives": list(mobility.objectives),
                "mobility_schedule": (
                    schedule.model_dump(mode="json")
                    if schedule is not None
                    else None
                ),
                "max_translation": (
                    bounds.max_translation
                    if bounds is not None
                    else None
                ),
                "max_rotation_deg": (
                    bounds.max_rotation_deg
                    if bounds is not None
                    else None
                ),
            }
        )
    return orbits


def _runtime_fixed_motif_constraints(
    *,
    instances,
    mapping: dict[str, Any],
    anchor_fragment_instance_ids: str | tuple[str, ...],
    transform_set,
    runtime_transform_order=None,
    transform_to_runtime_representative=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lower fixed fragments into explicit rigid-geometry components.

    Fragments in one source motion group form one joint component and must
    retain their relative geometry.  Different motion groups become separate
    runtime constraint orbits, so downstream projection and audit code never
    has to infer coupling from selector order or topology names.
    """

    anchor_ids = (
        (anchor_fragment_instance_ids,)
        if isinstance(anchor_fragment_instance_ids, str)
        else tuple(dict.fromkeys(anchor_fragment_instance_ids))
    )
    if not anchor_ids:
        raise ValueError("A fixed constraint orbit requires anchor fragments")
    masters = [instances.fragments[item] for item in anchor_ids]
    orbit_ids = {fragment.orbit_id for fragment in masters}
    if None in orbit_ids or len(orbit_ids) != 1:
        raise ValueError(
            "A symmetric fixed constraint path requires anchors from one "
            "declared orbit"
        )
    orbit_id = next(iter(orbit_ids))
    if len({fragment.source_id for fragment in masters}) != len(masters):
        raise ValueError(
            "A fixed constraint orbit cannot repeat a source fragment"
        )
    registry = build_transform_registry(transform_set)
    masters_by_component: dict[str, list[Any]] = {}
    for master in masters:
        motion_group = instances.motion_groups[
            master.motion_group_instance_id
        ]
        masters_by_component.setdefault(
            motion_group.source_id, []
        ).append(master)

    groups: list[dict[str, Any]] = []
    orbits: list[dict[str, Any]] = []
    compiled_orbit = instances.constraint_orbits[orbit_id]
    selected_order = tuple(
        runtime_transform_order or compiled_orbit.transform_ids
    )
    if selected_order != tuple(compiled_orbit.transform_ids):
        raise ValueError(
            f"Runtime transform order for orbit {orbit_id!r} disagrees "
            "with its compiled physical copies"
        )
    single_component = len(masters_by_component) == 1
    for component_id, component_masters in masters_by_component.items():
        mobility = compiled_orbit.component_mobility.get(
            component_id,
            compiled_orbit.mobility,
        )
        bounds = mobility.bounds
        schedule = mobility.effective_schedule
        # The native RFD3 ASU is the materialized polymer path, which may
        # cross a compiler symmetry-copy seam.  Runtime ``src_component``
        # labels therefore come from the *actual path anchors* rather than
        # from their copy-zero counterparts.  Native symmetry expansion keeps
        # those labels and records the relative group action separately in
        # ``sym_transform_id``.
        source_components = {
            master.source_id: _selector_source_components(
                _fragment_selector(mapping, master.id)
            )
            for master in component_masters
        }
        copies_by_source_id: dict[str, dict[str, Any]] = {}
        for master in component_masters:
            source_copies = [
                fragment
                for fragment in instances.fragments.values()
                if fragment.source_id == master.source_id
                and fragment.orbit_id == orbit_id
            ]
            copies_by_transform_id = {
                fragment.transform_id: fragment
                for fragment in source_copies
            }
            if (
                len(source_copies) != len(selected_order)
                or len(copies_by_transform_id) != len(selected_order)
                or set(copies_by_transform_id) != set(
                    selected_order
                )
            ):
                raise ValueError(
                    f"Fixed motif {master.source_id!r} does not contain "
                    "exactly one fragment for every symmetry transform"
                )
            copies_by_source_id[master.source_id] = (
                copies_by_transform_id
            )

        runtime_orbit_id = (
            orbit_id
            if single_component
            else f"{orbit_id}__{component_id}"
        )
        component_groups: list[dict[str, Any]] = []
        group_transform_ids: list[int] = []
        group_registry_transform_ids: list[str] = []
        for transform_index, transform_id in enumerate(
            selected_order
        ):
            members: list[dict[str, Any]] = []
            for master in component_masters:
                target_fragment = copies_by_source_id[
                    master.source_id
                ][transform_id]
                runtime_action = _runtime_action_index(
                    registry,
                    anchor_transform_id=master.transform_id,
                    target_transform_id=target_fragment.transform_id,
                    context=(
                        f"fixed fragment {master.id!r} in physical "
                        f"transform {transform_id!r}"
                    ),
                    runtime_transform_order=selected_order,
                    transform_to_runtime_representative=(
                        transform_to_runtime_representative
                    ),
                )
                members.append(
                    {
                        "role": "motif",
                        "source_fragment_id": master.source_id,
                        "src_components": source_components[
                            master.source_id
                        ],
                        "sym_transform_id": runtime_action,
                    }
                )

            group_id = f"fixed@{runtime_orbit_id}[{transform_index}]"
            group = {
                "group_id": group_id,
                "constraint_kind": "fixed_motif",
                "geometry_lock": "joint_rigid",
                "coupling_group_id": component_id,
                "constraint_orbit_id": runtime_orbit_id,
                "orbit_id": orbit_id,
                "members": members,
            }
            component_groups.append(group)
            groups.append(group)
            group_transform_ids.append(transform_index)
            group_registry_transform_ids.append(transform_id)

        orbits.append(
            {
                "constraint_orbit_id": runtime_orbit_id,
                "symmetry_orbit_id": orbit_id,
                "coupling_group_id": component_id,
                "geometry_lock": "joint_rigid",
                "source_fragment_ids": [
                    master.source_id for master in component_masters
                ],
                "source_components": [
                    component
                    for master in component_masters
                    for component in source_components[master.source_id]
                ],
                "master_group_id": component_groups[0]["group_id"],
                "group_ids": [
                    group["group_id"] for group in component_groups
                ],
                "group_transform_ids": group_transform_ids,
                "group_registry_transform_ids": (
                    group_registry_transform_ids
                ),
                "mobility_mode": mobility.mode.value,
                "mobility_subspace": (
                    mobility.effective_subspace.value
                    if mobility.effective_subspace is not None
                    else None
                ),
                "mobility_proposal": (
                    mobility.effective_proposal.value
                    if mobility.effective_proposal is not None
                    else None
                ),
                "mobility_objectives": list(mobility.objectives),
                "mobility_schedule": (
                    schedule.model_dump(mode="json")
                    if schedule is not None
                    else None
                ),
                "max_translation": (
                    bounds.max_translation if bounds is not None else None
                ),
                "max_rotation_deg": (
                    bounds.max_rotation_deg if bounds is not None else None
                ),
            }
        )
    return groups, orbits


def _runtime_interface_relation_audit_plan(
    *,
    instances,
    mapping: dict[str, Any],
    preexpanded_mixed_orbits: bool = False,
) -> list[dict[str, Any]]:
    """Freeze topology-neutral interface relations for result auditing.

    RFD3 renumbers and merges the master-copy fragments into one or more ASU
    chains.  The source components below are those canonical RFD3 selectors,
    while the copy indices retain the compiler's exact group-action pairing.
    A post-diffusion audit can therefore reconstruct every declared edge
    without reopening the original public YAML or guessing chain mappings.
    """

    def source_components(port_instance_id: str) -> list[str]:
        port = instances.ports[port_instance_id]
        selectors: list[str] = []
        for fragment_instance_id in port.fragment_instance_ids:
            fragment = instances.fragments[fragment_instance_id]
            if preexpanded_mixed_orbits:
                selector = _fragment_selector(
                    mapping,
                    fragment_instance_id,
                )
            else:
                masters = [
                    candidate
                    for candidate in instances.fragments.values()
                    if candidate.source_id == fragment.source_id
                    and candidate.orbit_id == fragment.orbit_id
                    and candidate.copy_index == 0
                ]
                if len(masters) != 1:
                    raise ValueError(
                        "Interface relation audit requires exactly one "
                        "copy-zero fragment for source "
                        f"{fragment.source_id!r}; observed {len(masters)}"
                    )
                selector = _fragment_selector(mapping, masters[0].id)
            if selector not in selectors:
                selectors.append(selector)
        if not selectors:
            raise ValueError(
                f"Interface port {port_instance_id!r} contains no fragments"
            )
        return selectors

    return [
        {
            "edge_instance_id": edge.id,
            "source_interface_id": edge.source_id,
            "hyperedge_id": edge.hyperedge_id or edge.source_id,
            "required": edge.required,
            "satisfaction_stage": edge.satisfaction_stage,
            "source_copy_index": edge.source_copy_index,
            "target_copy_index": edge.target_copy_index,
            "action_copy_index": edge.action_copy_index,
            "action_transform_id": edge.action_transform_id,
            "left_orbit_id": edge.left_orbit_id,
            "right_orbit_id": edge.right_orbit_id,
            "left_transform_index": edge.left_transform_index,
            "right_transform_index": edge.right_transform_index,
            "left_source_components": source_components(
                edge.left_port_instance_id
            ),
            "right_source_components": source_components(
                edge.right_port_instance_id
            ),
            "reference_basis": (
                "compiled_presymmetrized_input"
                if edge.target_geometry.mode == "reference_transform"
                and edge.target_geometry.from_reference_seed
                else "declared_target_geometry"
            ),
            "target_geometry": edge.target_geometry.model_dump(mode="json"),
        }
        for edge in instances.interfaces.values()
    ]


def _materialize_length(
    minimum: int,
    maximum: int,
    *,
    label: str,
) -> int:
    if minimum > maximum:
        raise ValueError(f"Invalid generated length range for {label!r}")
    return (minimum + maximum) // 2


def _compile_asu_terminal_path(
    extensions: list[TerminalExtensionInstance],
    *,
    mapping: dict[str, Any],
) -> _ASUTerminalPath:
    """Compile terminal extensions around one copy-zero motif fragment."""

    if not extensions:
        raise ValueError("A terminal path requires at least one extension")
    anchor_ids = {
        extension.anchor_fragment_instance_id for extension in extensions
    }
    if len(anchor_ids) != 1:
        raise NotImplementedError(
            "One native RFD3 ASU path currently supports terminal "
            "extensions around exactly one motif fragment"
        )
    anchor_id = next(iter(anchor_ids))
    orbit_ids = {extension.orbit_id for extension in extensions}
    if len(orbit_ids) != 1 or None in orbit_ids:
        raise ValueError(
            "Native symmetry requires terminal extensions to belong to "
            "one expanded orbit"
        )
    by_terminus: dict[str, TerminalExtensionInstance] = {}
    for extension in extensions:
        key = extension.anchor_terminus.value
        if key in by_terminus:
            raise ValueError(
                f"Motif anchor {anchor_id!r} has multiple {key}-terminal "
                "extensions"
            )
        by_terminus[key] = extension

    selector = _fragment_selector(mapping, anchor_id)
    components: list[str] = []
    n_extension = by_terminus.get("N")
    c_extension = by_terminus.get("C")
    if n_extension is not None:
        length = _materialize_length(
            n_extension.minimum_length,
            n_extension.maximum_length,
            label=n_extension.id,
        )
        components.append(f"{length}-{length}")
    components.append(selector)
    if c_extension is not None:
        length = _materialize_length(
            c_extension.minimum_length,
            c_extension.maximum_length,
            label=c_extension.id,
        )
        components.append(f"{length}-{length}")
    return _ASUTerminalPath(
        anchor_fragment_instance_id=anchor_id,
        selector=selector,
        orbit_id=next(iter(orbit_ids)),
        n_extension=n_extension,
        c_extension=c_extension,
        contig=",".join(components),
    )


def _materialized_linker_contour_preflight(
    manifest_path: Path,
    *,
    source_link_id: str,
    materialized_length: int | None,
) -> dict[str, Any]:
    """Re-evaluate standalone endpoint spans at the emitted exact length."""

    if materialized_length is None:
        return {
            "status": "not_applicable",
            "passed": True,
            "materialized_linker_length": None,
            "evaluated_link_instances": [],
        }
    manifest = _load_json(manifest_path)
    try:
        geometry = manifest["validation"]["scaffold_link_geometry"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Standalone manifest is missing scaffold-link geometry"
        ) from error
    link_reports = geometry.get("links", [])
    matching = [
        report
        for report in link_reports
        if report.get("source_link_id") == source_link_id
    ]
    if not matching:
        raise ValueError(
            "Standalone manifest contains no geometry for scaffold link "
            f"{source_link_id!r}"
        )

    evaluated = []
    failed = []
    for report in matching:
        required = int(report["minimum_required_residues_at_3_8A"])
        passed = required <= materialized_length
        instance_id = str(report["link_instance_id"])
        evaluated.append(
            {
                "link_instance_id": instance_id,
                "minimum_required_residues_at_3_8A": required,
                "materialized_linker_length": materialized_length,
                "passed": passed,
            }
        )
        if not passed:
            failed.append(instance_id)
    if failed:
        raise ValueError(
            "Materialized linker length is geometrically insufficient for "
            f"scaffold-link instances {failed}: length="
            f"{materialized_length}"
        )
    return {
        "status": "passed",
        "passed": True,
        "materialized_linker_length": materialized_length,
        "evaluated_link_instances": evaluated,
    }


def _minimum_materialized_linker_length(
    manifest_path: Path,
    *,
    source_link_id: str,
) -> int:
    """Return the worst-case contour requirement over a link orbit.

    The public design declares a linker *range*, whereas native RFD3 needs
    one exact contig length.  Choosing the range midpoint without considering
    every symmetry-expanded link instance can make an otherwise feasible
    candidate fail only when it reaches the adapter.  Bind the exact length
    to the largest endpoint-contour requirement in the standalone manifest.
    """

    manifest = _load_json(manifest_path)
    try:
        geometry = manifest["validation"]["scaffold_link_geometry"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Standalone manifest is missing scaffold-link geometry"
        ) from error
    source_bindings = geometry.get("source_link_bindings", {})
    if source_link_id in source_bindings:
        binding = source_bindings[source_link_id]
        try:
            return int(
                binding[
                    "required_minimum_over_physical_instances"
                ]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Standalone source-link binding contains no valid contour "
                f"requirement for {source_link_id!r}"
            ) from error
    # Backward-compatible fallback for retained manifests produced before
    # source-level orbit bindings became explicit.
    link_reports = geometry.get("links", [])
    requirements = [
        int(report["minimum_required_residues_at_3_8A"])
        for report in link_reports
        if report.get("source_link_id") == source_link_id
        and not bool(report.get("chain_break", False))
    ]
    if not requirements:
        raise ValueError(
            "Standalone manifest contains no geometry for scaffold link "
            f"{source_link_id!r}"
        )
    return max(requirements)


def _compile_asu_scaffold_segments(
    links,
    *,
    mapping: dict[str, Any],
    manifest_path: Path,
    linker_length: int | None,
) -> tuple[_ASUScaffoldSegment, ...]:
    """Compile copy-zero links into non-duplicating ordered ASU paths.

    A fixed fragment may be the target of one link and the source of the next,
    as in ``A -> B -> C``.  The emitted contig contains B exactly once.  The
    first implementation deliberately rejects branching and cycles because a
    single protein chain has at most one N-side and one C-side neighbour.
    """

    automatic_lengths: dict[str, tuple[int, str]] = {}
    if linker_length is None:
        grouped_links: dict[str, list[Any]] = {}
        for link in links:
            if link.chain_break:
                continue
            tie_group = getattr(link, "tie_group", None)
            key = (
                f"tie:{tie_group}"
                if tie_group is not None
                else f"link:{link.source_id}"
            )
            grouped_links.setdefault(key, []).append(link)
        for group in grouped_links.values():
            common_minimum = max(link.minimum_length for link in group)
            common_maximum = min(link.maximum_length for link in group)
            if common_minimum > common_maximum:
                raise ValueError(
                    "Scaffold-link tie group has no common configured "
                    f"length: {[link.source_id for link in group]}"
                )
            common_midpoint = (common_minimum + common_maximum) // 2
            required = max(
                _minimum_materialized_linker_length(
                    manifest_path,
                    source_link_id=link.source_id,
                )
                for link in group
            )
            selected = max(common_midpoint, required)
            if selected > common_maximum:
                raise ValueError(
                    "No linker length in the common configured range "
                    f"[{common_minimum}, {common_maximum}] can span every "
                    "symmetry instance of scaffold links "
                    f"{[link.source_id for link in group]}; required "
                    f"length={selected}"
                )
            # A historical single link may carry a tie-group label even
            # though there is nothing to tie.  Preserve its legacy policy
            # name; use tie-group policies only when several source links
            # actually share one exact length.
            tied = (
                getattr(group[0], "tie_group", None) is not None
                and len({link.source_id for link in group}) > 1
            )
            for link in group:
                policy = (
                    "user_exact"
                    if link.minimum_length == link.maximum_length == selected
                    else "tie_group_common_midpoint"
                    if tied and selected == common_midpoint
                    else "tie_group_contour_sufficient"
                    if tied
                    else "configured_range_midpoint"
                    if selected
                    == (link.minimum_length + link.maximum_length) // 2
                    else "configured_range_contour_sufficient"
                )
                automatic_lengths[link.source_id] = (selected, policy)

    def materialize(link) -> _MaterializedScaffoldLink:
        if link.chain_break:
            if linker_length is not None:
                raise ValueError(
                    "linker_length cannot be set when an ASU scaffold "
                    "segment is a chain break"
                )
            materialized_length = None
            linker_policy = "not_applicable"
        else:
            if linker_length is None:
                materialized_length, linker_policy = automatic_lengths[
                    link.source_id
                ]
            else:
                if isinstance(linker_length, bool) or not isinstance(
                    linker_length,
                    int,
                ):
                    raise TypeError("linker_length must be an integer")
                materialized_length = linker_length
                linker_policy = "explicit"
            if not (
                link.minimum_length
                <= materialized_length
                <= link.maximum_length
            ):
                if linker_length is None:
                    raise ValueError(
                        "No linker length in the configured range "
                        f"[{link.minimum_length}, {link.maximum_length}] "
                        f"can span every symmetry instance of "
                        f"{link.source_id!r}; required length="
                        f"{materialized_length}"
                    )
                raise ValueError(
                    "linker_length must fall inside the configured range "
                    f"[{link.minimum_length}, {link.maximum_length}] for "
                    f"{link.id!r}, got {materialized_length}"
                )

        contour_preflight = _materialized_linker_contour_preflight(
            manifest_path,
            source_link_id=link.source_id,
            materialized_length=materialized_length,
        )
        return _MaterializedScaffoldLink(
            link=link,
            materialized_linker_length=materialized_length,
            linker_length_policy=linker_policy,
            contour_preflight=contour_preflight,
        )

    ordered_links = tuple(sorted(links, key=lambda item: item.id))
    if len({link.id for link in ordered_links}) != len(ordered_links):
        raise ValueError("ASU scaffold link instance IDs must be unique")
    materialized = {
        link.id: materialize(link) for link in ordered_links
    }

    continuous = tuple(link for link in ordered_links if not link.chain_break)
    breaks = tuple(link for link in ordered_links if link.chain_break)
    outgoing: dict[str, Any] = {}
    incoming: dict[str, Any] = {}
    for link in continuous:
        previous_out = outgoing.setdefault(
            link.from_fragment_instance_id,
            link,
        )
        if previous_out.id != link.id:
            raise NotImplementedError(
                "Generated scaffold topology branches from fixed fragment "
                f"{link.from_fragment_instance_id!r}; one protein chain "
                "cannot have two C-terminal outgoing links"
            )
        previous_in = incoming.setdefault(
            link.to_fragment_instance_id,
            link,
        )
        if previous_in.id != link.id:
            raise NotImplementedError(
                "Generated scaffold topology merges into fixed fragment "
                f"{link.to_fragment_instance_id!r}; one protein chain "
                "cannot have two N-terminal incoming links"
            )

    starts = sorted(
        (
            fragment_id
            for fragment_id in outgoing
            if fragment_id not in incoming
        ),
        key=lambda fragment_id: outgoing[fragment_id].id,
    )
    if continuous and not starts:
        raise NotImplementedError(
            "Generated scaffold topology contains a closed cycle; native "
            "RFD3 contigs currently require an open N-to-C path"
        )

    segments: list[_ASUScaffoldSegment] = []
    visited: set[str] = set()
    fragment_path_owners: dict[str, str] = {}
    for start in starts:
        path_links = []
        fragment_ids = [start]
        cursor = start
        while cursor in outgoing:
            link = outgoing[cursor]
            if link.id in visited:
                raise ValueError(
                    "Generated scaffold path traversal repeated link "
                    f"{link.id!r}"
                )
            visited.add(link.id)
            path_links.append(materialized[link.id])
            cursor = link.to_fragment_instance_id
            fragment_ids.append(cursor)

        selectors = tuple(
            _fragment_selector(mapping, fragment_id)
            for fragment_id in fragment_ids
        )
        contig_parts = [selectors[0]]
        for index, entry in enumerate(path_links):
            length = entry.materialized_linker_length
            assert length is not None
            contig_parts.extend((f"{length}-{length}", selectors[index + 1]))
        path_name = path_links[0].link.id
        for fragment_id in fragment_ids:
            previous = fragment_path_owners.setdefault(fragment_id, path_name)
            if previous != path_name:
                raise NotImplementedError(
                    "A fixed fragment belongs to multiple generated ASU "
                    f"paths: {fragment_id!r}"
                )
        contour_reports = tuple(
            entry.contour_preflight for entry in path_links
        )
        segments.append(
            _ASUScaffoldSegment(
                links=tuple(path_links),
                fragment_instance_ids=tuple(fragment_ids),
                selectors=selectors,
                from_selector=selectors[0],
                to_selector=selectors[-1],
                contig_chains=(",".join(contig_parts),),
                materialized_linker_length=(
                    path_links[0].materialized_linker_length
                    if len(path_links) == 1
                    else None
                ),
                linker_length_policy=(
                    path_links[0].linker_length_policy
                    if len(path_links) == 1
                    else "ordered_path_per_link"
                ),
                contour_preflight={
                    "status": "passed",
                    "passed": all(
                        report["passed"] for report in contour_reports
                    ),
                    "materialized_linker_length": (
                        path_links[0].materialized_linker_length
                        if len(path_links) == 1
                        else None
                    ),
                    "evaluated_link_instances": [
                        item
                        for report in contour_reports
                        for item in report["evaluated_link_instances"]
                    ],
                },
            )
        )

    if len(visited) != len(continuous):
        missing = sorted(
            link.id for link in continuous if link.id not in visited
        )
        raise NotImplementedError(
            "Generated scaffold topology contains a cycle or disconnected "
            f"non-path component: {missing}"
        )

    for link in breaks:
        entry = materialized[link.id]
        fragment_ids = (
            link.from_fragment_instance_id,
            link.to_fragment_instance_id,
        )
        for fragment_id in fragment_ids:
            previous = fragment_path_owners.setdefault(fragment_id, link.id)
            if previous != link.id:
                raise NotImplementedError(
                    "A fixed fragment cannot participate in both a chain "
                    "break and another generated ASU path: "
                    f"{fragment_id!r}"
                )
        selectors = tuple(
            _fragment_selector(mapping, fragment_id)
            for fragment_id in fragment_ids
        )
        segments.append(
            _ASUScaffoldSegment(
                links=(entry,),
                fragment_instance_ids=fragment_ids,
                selectors=selectors,
                from_selector=selectors[0],
                to_selector=selectors[1],
                contig_chains=selectors,
                materialized_linker_length=None,
                linker_length_policy="not_applicable",
                contour_preflight=entry.contour_preflight,
            )
        )

    return tuple(
        sorted(segments, key=lambda segment: segment.links[0].link.id)
    )


def _preexpanded_stabilizer_path_layout(
    *,
    finite_action,
    segments: tuple[_ASUScaffoldSegment, ...],
    registry_transform_order: list[str],
    identity_transform_id: str,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Bind already materialized G/G scaffold paths to stabilizer frames.

    A complete supplied oligomer may have one physical quotient copy while
    containing one generated polymer path for every stabilizer participant.
    RFD3 must not expand that oligomer again.  Instead, each existing chain is
    annotated as one transform of a single preexpanded entity.
    """

    declared = dict(finite_action.stabilizer_path_transform_ids)
    if not declared:
        raise ValueError(
            "A one-copy stabilized ASU requires frozen scaffold-path to "
            "stabilizer-transform assignments"
        )
    selected = set(registry_transform_order)
    unknown = set(declared.values()) - selected
    if unknown:
        raise ValueError(
            "Stabilized ASU path assignments reference transforms outside "
            f"the runtime stabilizer order: {sorted(unknown)}"
        )

    segment_transform_ids: list[str] = []
    consumed_path_ids: set[str] = set()
    layout: list[dict[str, Any]] = []

    def declared_path_id(runtime_path_id: str) -> str | None:
        """Resolve a compiler link ID back to its public connection ID.

        Public assembly connections are lowered into generated segments named
        ``connection__<public id>``.  The finite-action contract intentionally
        stores the stable public ID, while scaffold instances expose the
        lowered segment ID.  Accept exactly those two representations here;
        do not use a fuzzy suffix match that could bind the wrong path.
        """

        if runtime_path_id in declared:
            return runtime_path_id
        prefix = "connection__"
        if runtime_path_id.startswith(prefix):
            public_path_id = runtime_path_id[len(prefix):]
            if public_path_id in declared:
                return public_path_id
        return None

    for segment in segments:
        if len(segment.contig_chains) != 1:
            raise NotImplementedError(
                "The first preexpanded stabilized-ASU backend requires "
                "one continuous output chain per stabilizer path"
            )
        runtime_path_ids = {
            entry.link.source_id for entry in segment.links
        }
        resolved_path_ids = {
            runtime_path_id: declared_path_id(runtime_path_id)
            for runtime_path_id in runtime_path_ids
        }
        missing = sorted(
            runtime_path_id
            for runtime_path_id, public_path_id
            in resolved_path_ids.items()
            if public_path_id is None
        )
        if missing:
            raise ValueError(
                "Stabilized ASU scaffold paths are missing transform "
                f"assignments: {missing}"
            )
        public_path_ids = {
            public_path_id
            for public_path_id in resolved_path_ids.values()
            if public_path_id is not None
        }
        transform_ids = {
            declared[path_id] for path_id in public_path_ids
        }
        if len(transform_ids) != 1:
            raise ValueError(
                "Every continuous stabilized-ASU path must remain inside "
                "one stabilizer transform"
            )
        transform_id = next(iter(transform_ids))
        transform_index = registry_transform_order.index(transform_id)
        segment_transform_ids.append(transform_id)
        consumed_path_ids.update(public_path_ids)
        layout.append(
            {
                "entity_id": 0,
                "transform_index": transform_index,
                "transform_id": transform_id,
                "orbit_id": segment.links[0].link.orbit_id,
                "is_asu": transform_id == identity_transform_id,
            }
        )

    unused = sorted(set(declared) - consumed_path_ids)
    if unused:
        raise ValueError(
            "Stabilizer path assignments do not correspond to generated "
            f"scaffold paths: {unused}"
        )
    if len(layout) != len(registry_transform_order) or {
        record["transform_id"] for record in layout
    } != selected:
        raise ValueError(
            "A preexpanded G/G ASU requires exactly one continuous chain "
            "for every stabilizer transform"
        )
    return layout, tuple(segment_transform_ids)


def _runtime_preexpanded_stabilizer_constraints(
    *,
    instances,
    mapping: dict[str, Any],
    segments: tuple[_ASUScaffoldSegment, ...],
    segment_transform_ids: tuple[str, ...],
    registry_transform_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create exact fixed-motif groups for one internally stabilized ASU."""

    if len(segments) != len(segment_transform_ids):
        raise ValueError("Stabilized ASU segment/transform counts disagree")
    identity_transform_id = registry_transform_order[0]
    try:
        canonical_index = segment_transform_ids.index(identity_transform_id)
    except ValueError as error:
        raise ValueError(
            "Stabilized ASU paths omit the identity-transform master"
        ) from error
    canonical = segments[canonical_index]
    if not canonical.fragment_instance_ids:
        raise ValueError("Stabilized ASU master path contains no fixed motif")

    canonical_fragments = [
        instances.fragments[fragment_id]
        for fragment_id in canonical.fragment_instance_ids
    ]
    canonical_components = [
        _selector_source_components(
            _fragment_selector(mapping, fragment.id)
        )
        for fragment in canonical_fragments
    ]
    orbit_ids = {
        entry.link.orbit_id
        for segment in segments
        for entry in segment.links
    }
    if len(orbit_ids) != 1 or None in orbit_ids:
        raise ValueError(
            "A stabilized ASU must belong to one declared symmetry orbit"
        )
    orbit_id = next(iter(orbit_ids))
    component_ids = {
        instances.motion_groups[
            instances.fragments[fragment_id].motion_group_instance_id
        ].source_id
        for segment in segments
        for fragment_id in segment.fragment_instance_ids
    }
    if len(component_ids) != 1:
        raise ValueError(
            "A preexpanded stabilized ASU must remain one joint-rigid "
            "component"
        )
    component_id = next(iter(component_ids))

    groups: list[dict[str, Any]] = []
    for segment, transform_id in zip(
        segments,
        segment_transform_ids,
        strict=True,
    ):
        if len(segment.fragment_instance_ids) != len(canonical_fragments):
            raise ValueError(
                "Stabilizer-related paths contain different fixed-fragment "
                "counts"
            )
        transform_index = registry_transform_order.index(transform_id)
        members = []
        for slot, (fragment_id, canonical_fragment) in enumerate(
            zip(
                segment.fragment_instance_ids,
                canonical_fragments,
                strict=True,
            )
        ):
            fragment = instances.fragments[fragment_id]
            actual_components = _selector_source_components(
                _fragment_selector(mapping, fragment.id)
            )
            correspondence_components = canonical_components[slot]
            if len(actual_components) != len(correspondence_components):
                raise ValueError(
                    "Stabilizer-related fixed fragments have different "
                    "residue counts"
                )
            members.append(
                {
                    "role": "motif",
                    "source_fragment_id": canonical_fragment.source_id,
                    "src_components": actual_components,
                    "correspondence_components": (
                        correspondence_components
                    ),
                    "sym_transform_id": transform_index,
                }
            )
        groups.append(
            {
                "group_id": f"fixed@{orbit_id}[{transform_index}]",
                "constraint_kind": "fixed_motif",
                "geometry_lock": "joint_rigid",
                "coupling_group_id": component_id,
                "constraint_orbit_id": orbit_id,
                "orbit_id": orbit_id,
                "members": members,
            }
        )

    compiled_orbit = instances.constraint_orbits[orbit_id]
    mobility = compiled_orbit.component_mobility.get(
        component_id,
        compiled_orbit.mobility,
    )
    bounds = mobility.bounds
    schedule = mobility.effective_schedule
    ordered_group_ids = [
        f"fixed@{orbit_id}[{index}]"
        for index in range(len(registry_transform_order))
    ]
    group_by_id = {group["group_id"]: group for group in groups}
    groups = [group_by_id[group_id] for group_id in ordered_group_ids]
    orbit = {
        "constraint_orbit_id": orbit_id,
        "symmetry_orbit_id": orbit_id,
        "coupling_group_id": component_id,
        "geometry_lock": "joint_rigid",
        "source_fragment_ids": [
            fragment.source_id for fragment in canonical_fragments
        ],
        "source_components": [
            component
            for components in canonical_components
            for component in components
        ],
        "master_group_id": ordered_group_ids[0],
        "group_ids": ordered_group_ids,
        "group_transform_ids": list(range(len(ordered_group_ids))),
        "group_registry_transform_ids": list(registry_transform_order),
        "mobility_mode": mobility.mode.value,
        "mobility_subspace": (
            mobility.effective_subspace.value
            if mobility.effective_subspace is not None
            else None
        ),
        "mobility_proposal": (
            mobility.effective_proposal.value
            if mobility.effective_proposal is not None
            else None
        ),
        "mobility_objectives": list(mobility.objectives),
        "mobility_schedule": (
            schedule.model_dump(mode="json")
            if schedule is not None
            else None
        ),
        "max_translation": (
            bounds.max_translation if bounds is not None else None
        ),
        "max_rotation_deg": (
            bounds.max_rotation_deg if bounds is not None else None
        ),
    }
    return groups, [orbit]


def compile_assembly_rfd3_input(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    base_directory: str | Path = ".",
    example_id: str = "lhd101_c3_interface_seed",
    is_non_loopy: bool = True,
    pose_seed: int | None = None,
    pose_candidate_manifest: str | Path | None = None,
    linker_length: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> RFD3AdapterOutputs:
    """Emit one native-symmetry RFD3 task from the common Assembly IR.

    Both cross-interface scaffold links and central-motif terminal extensions
    are lowered from ``CompiledInstanceSet.generated_segments`` into one ASU
    contig compiler.  RFD3 then expands that ASU with the compiler-validated
    symmetry registry.
    """

    config = Path(config_path).resolve()
    base = Path(base_directory).resolve()
    output = Path(output_directory)
    candidate_manifest_path: Path | None = None
    sample_overrides: dict[str, dict[str, Any]] | None = None
    candidate_structure_sha: str | None = None
    if pose_candidate_manifest is not None:
        if pose_seed is not None:
            raise ValueError(
                "pose_seed and pose_candidate_manifest are mutually exclusive"
            )
        candidate_manifest_path = Path(pose_candidate_manifest).resolve()
        (
            sample_overrides,
            pose_seed,
            candidate_structure_sha,
        ) = _candidate_sample_overrides(
            candidate_manifest_path,
            config_path=config,
        )
    spec = load_interface_seed_config(config)
    standalone = compile_standalone(
        config,
        output,
        base_directory=base,
        random_seed=pose_seed,
        sample_overrides=sample_overrides,
    )
    rebuilt_structure_sha = _sha256(standalone.structure_path)
    if (
        candidate_structure_sha is not None
        and rebuilt_structure_sha != candidate_structure_sha
    ):
        raise ValueError(
            "Rebuilt RFD3 adapter structure does not exactly reproduce the "
            f"pose candidate: {rebuilt_structure_sha} != "
            f"{candidate_structure_sha}"
        )
    instances = expand_symmetry_instances(
        spec,
        master_transforms=None,
    )
    mapping = _load_json(standalone.mapping_path)

    generated_orbit_ids = {
        segment.orbit_id
        for segment in instances.generated_segments.values()
        if segment.orbit_id is not None
    }
    generated_orbit_ids.update(
        link.orbit_id
        for link in instances.scaffold_links.values()
        if link.orbit_id is not None
    )
    generated_action_payloads = {
        json.dumps(
            (
                spec.symmetry.orbits[orbit_id].finite_action.model_dump(
                    mode="json"
                )
                if spec.symmetry.orbits[orbit_id].finite_action is not None
                else None
            ),
            sort_keys=True,
        )
        for orbit_id in generated_orbit_ids
    }
    preexpanded_mixed_orbits = len(generated_action_payloads) > 1
    generated_stabilized_actions = [
        spec.symmetry.orbits[orbit_id].finite_action
        for orbit_id in generated_orbit_ids
        if spec.symmetry.orbits[orbit_id].finite_action is not None
        and spec.symmetry.orbits[
            orbit_id
        ].finite_action.stabilizer_path_transform_ids
    ]
    preexpanded_stabilizer_asu = bool(generated_stabilized_actions)
    if preexpanded_stabilizer_asu and (
        len(generated_orbit_ids) != 1
        or len(generated_stabilized_actions) != 1
    ):
        raise NotImplementedError(
            "A preexpanded stabilized ASU currently requires exactly one "
            "generated component orbit"
        )
    preexpanded_runtime_layout = (
        preexpanded_mixed_orbits or preexpanded_stabilizer_asu
    )

    link_by_id = {
        link.id: link
        for link in instances.scaffold_links.values()
        if preexpanded_runtime_layout or link.copy_index == 0
    }
    for segment in instances.generated_segments.values():
        if (
            isinstance(segment, ScaffoldLinkInstance)
            and (preexpanded_runtime_layout or segment.copy_index == 0)
        ):
            link_by_id.setdefault(segment.id, segment)
    orbit_links = list(link_by_id.values())
    terminal_extensions = [
        segment
        for segment in instances.generated_segments.values()
        if isinstance(segment, TerminalExtensionInstance)
        and (preexpanded_runtime_layout or segment.copy_index == 0)
    ]
    if preexpanded_runtime_layout and terminal_extensions:
        raise NotImplementedError(
            "Preexpanded symmetry execution currently requires explicit "
            "scaffold-link paths; terminal-extension paths are not yet "
            "materialized"
        )
    if orbit_links and terminal_extensions:
        raise NotImplementedError(
            "A single native ASU path cannot yet mix scaffold links and "
            "terminal extensions"
        )
    if not orbit_links and not terminal_extensions:
        raise ValueError(
            "The native RFD3 adapter requires at least one copy-zero "
            "generated segment"
        )

    terminal_path: _ASUTerminalPath | None = None
    transform_set_ids = set()
    for link in orbit_links:
        if link.orbit_id is None:
            raise ValueError(
                "Native symmetry requires every ASU scaffold relation to "
                "belong to an expanded orbit"
            )
        transform_set_ids.add(
            spec.symmetry.orbits[link.orbit_id].transform_set
        )
    if terminal_extensions:
        terminal_path = _compile_asu_terminal_path(
            terminal_extensions,
            mapping=mapping,
        )
        transform_set_ids.add(
            spec.symmetry.orbits[terminal_path.orbit_id].transform_set
        )
    if len(transform_set_ids) != 1:
        raise ValueError(
            "All ASU scaffold relations must use one native symmetry "
            f"transform set, observed {sorted(transform_set_ids)}"
        )
    transform_set_id = next(iter(transform_set_ids))
    transform_set = spec.symmetry.transform_sets[transform_set_id]
    symmetry_id, full_symmetry_multiplicity = (
        _native_symmetry_id_and_multiplicity(transform_set)
    )
    full_registry_transform_order = _preflight_native_transform_registry(
        transform_set,
        full_symmetry_multiplicity,
    )
    native_registry = build_transform_registry(transform_set)
    full_registry_transform_matrices = {
        transform_id: native_registry.transform(transform_id).tolist()
        for transform_id in native_registry.transform_ids
    }
    active_orbit_ids = (
        {terminal_path.orbit_id}
        if terminal_path is not None
        else {link.orbit_id for link in orbit_links}
    )
    active_orbits = {
        orbit_id: spec.symmetry.orbits[orbit_id]
        for orbit_id in active_orbit_ids
    }
    finite_action_payloads = {
        json.dumps(
            orbit.finite_action.model_dump(mode="json"),
            sort_keys=True,
        )
        for orbit in active_orbits.values()
        if orbit.finite_action is not None
    }
    if len(finite_action_payloads) > 1 and not preexpanded_mixed_orbits:
        raise NotImplementedError(
            "Different finite quotient actions require the compiler-owned "
            "preexpanded mixed-entity backend"
        )
    has_full_orbit = any(
        orbit.finite_action is None for orbit in active_orbits.values()
    )
    has_quotient_orbit = bool(finite_action_payloads)
    if (
        has_full_orbit
        and has_quotient_orbit
        and not preexpanded_mixed_orbits
    ):
        raise NotImplementedError(
            "One native RFD3 task cannot mix full-group and quotient-group "
            "physical copy sets"
        )
    finite_action = (
        next(
            orbit.finite_action
            for orbit in active_orbits.values()
            if orbit.finite_action is not None
        )
        if has_quotient_orbit and not preexpanded_mixed_orbits
        else None
    )
    registry_transform_order = (
        full_registry_transform_order
        if preexpanded_mixed_orbits
        else list(finite_action.stabilizer_transform_ids)
        if preexpanded_stabilizer_asu and finite_action is not None
        else list(finite_action.coset_representative_ids)
        if finite_action is not None
        else full_registry_transform_order
    )
    transform_to_runtime_representative = (
        {
            transform_id: transform_id
            for transform_id in full_registry_transform_order
        }
        if preexpanded_runtime_layout
        else dict(finite_action.transform_to_coset_representative)
        if finite_action is not None
        else {
            transform_id: transform_id
            for transform_id in full_registry_transform_order
        }
    )
    registry_transform_matrices = {
        transform_id: full_registry_transform_matrices[transform_id]
        for transform_id in registry_transform_order
    }
    for orbit_id in active_orbit_ids:
        compiled_order = tuple(
            instances.constraint_orbits[orbit_id].transform_ids
        )
        if preexpanded_runtime_layout:
            if not set(compiled_order).issubset(registry_transform_order):
                raise ValueError(
                    f"Compiled mixed-orbit frames for {orbit_id!r} are not "
                    "a subset of the full runtime transform registry"
                )
        elif compiled_order != tuple(registry_transform_order):
            raise ValueError(
                f"Compiled physical copies for orbit {orbit_id!r} do not "
                "match the RFD3 runtime transform order: "
                f"{compiled_order} != {tuple(registry_transform_order)}"
            )
    symmetry_multiplicity = len(registry_transform_order)
    if symmetry_multiplicity < 2 and not preexpanded_stabilizer_asu:
        raise NotImplementedError(
            "RFD3 quotient execution requires at least two physical copies"
        )

    segments: tuple[_ASUScaffoldSegment, ...] = ()
    if terminal_path is not None:
        if linker_length is not None:
            raise ValueError(
                "linker_length does not apply to terminal extensions; set "
                "their ranges in the AssemblySpecification"
            )
        contig = terminal_path.contig
        asu_chain_count = 1
        scaffold_mode = "terminal_extensions"
        configured_linker_range = None
        materialized_linker_length = None
        linker_length_policy = "per_terminal_extension"
        linker_contour_preflight = {
            "status": "not_applicable",
            "passed": True,
            "materialized_linker_length": None,
            "evaluated_link_instances": [],
        }
    else:
        segments = _compile_asu_scaffold_segments(
            orbit_links,
            mapping=mapping,
            manifest_path=standalone.manifest_path,
            linker_length=linker_length,
        )
        contig = ",/0,".join(
            chain
            for segment in segments
            for chain in segment.contig_chains
        )
        asu_chain_count = sum(
            len(segment.contig_chains) for segment in segments
        )
    preexpanded_chain_layout: list[dict[str, Any]] | None = None
    stabilized_segment_transform_ids: tuple[str, ...] | None = None
    if preexpanded_stabilizer_asu:
        if finite_action is None:
            raise ValueError(
                "Preexpanded stabilized ASU lost its finite action"
            )
        (
            preexpanded_chain_layout,
            stabilized_segment_transform_ids,
        ) = _preexpanded_stabilizer_path_layout(
            finite_action=finite_action,
            segments=segments,
            registry_transform_order=registry_transform_order,
            identity_transform_id=native_registry.identity_id,
        )
    elif preexpanded_mixed_orbits:
        entity_ids: dict[tuple[tuple[str, ...], int], int] = {}
        preexpanded_chain_layout = []
        entity_transform_indices: dict[int, set[int]] = {}
        entity_asu_counts: dict[int, int] = {}
        for segment in segments:
            segment_orbit_ids = {
                entry.link.orbit_id for entry in segment.links
            }
            if len(segment_orbit_ids) != 1 or None in segment_orbit_ids:
                raise ValueError(
                    "A preexpanded mixed-entity scaffold path must belong "
                    "to exactly one component orbit"
                )
            orbit_id = next(iter(segment_orbit_ids))
            orbit = instances.constraint_orbits[orbit_id]
            copy_indices = {entry.link.copy_index for entry in segment.links}
            if len(copy_indices) != 1:
                raise NotImplementedError(
                    "A preexpanded mixed-entity protein path cannot change "
                    "component-orbit copy inside one covalent chain"
                )
            copy_index = next(iter(copy_indices))
            transform_id = orbit.transform_ids[copy_index]
            transform_index = full_registry_transform_order.index(
                transform_id
            )
            source_path_key = tuple(
                entry.link.source_id for entry in segment.links
            )
            for chain_offset, _ in enumerate(segment.contig_chains):
                entity_key = (source_path_key, chain_offset)
                entity_id = entity_ids.setdefault(
                    entity_key,
                    len(entity_ids),
                )
                observed = entity_transform_indices.setdefault(
                    entity_id,
                    set(),
                )
                if transform_index in observed:
                    raise ValueError(
                        "Preexpanded mixed-entity layout contains duplicate "
                        f"transform {transform_id!r} for entity {entity_id}"
                    )
                observed.add(transform_index)
                chain_is_asu = transform_id == native_registry.identity_id
                if chain_is_asu:
                    entity_asu_counts[entity_id] = (
                        entity_asu_counts.get(entity_id, 0) + 1
                    )
                preexpanded_chain_layout.append({
                    "entity_id": entity_id,
                    "transform_index": transform_index,
                    "transform_id": transform_id,
                    "orbit_id": orbit_id,
                    "is_asu": chain_is_asu,
                })
        invalid_asu = {
            entity_id: entity_asu_counts.get(entity_id, 0)
            for entity_id in entity_transform_indices
            if entity_asu_counts.get(entity_id, 0) != 1
        }
        if invalid_asu:
            raise ValueError(
                "Each preexpanded mixed symmetry entity requires exactly "
                f"one identity-transform chain: {invalid_asu}"
            )
    materialized_links = tuple(
        entry
        for segment in segments
        for entry in segment.links
    )
    compiled_links = tuple(entry.link for entry in materialized_links)

    if (
        terminal_path is None
        and len(segments) == 1
        and len(compiled_links) == 1
    ):
        only_segment = segments[0]
        only_link = compiled_links[0]
        scaffold_mode = (
            "independent_chains"
            if only_link.chain_break
            else "continuous_linker"
        )
        configured_linker_range = [
            only_link.minimum_length,
            only_link.maximum_length,
        ]
        materialized_linker_length = (
            only_segment.materialized_linker_length
        )
        linker_length_policy = only_segment.linker_length_policy
        linker_contour_preflight = only_segment.contour_preflight
    elif terminal_path is None and len(segments) == 1:
        scaffold_mode = "ordered_asu_scaffold_path"
        configured_linker_range = None
        materialized_linker_length = None
        linker_length_policy = "ordered_path_per_link"
        linker_contour_preflight = segments[0].contour_preflight
    elif terminal_path is None:
        scaffold_mode = "multiple_asu_scaffold_segments"
        configured_linker_range = None
        materialized_linker_length = None
        linker_length_policy = "per_segment"
        linker_contour_preflight = {
            "status": "passed",
            "passed": all(
                segment.contour_preflight["passed"]
                for segment in segments
            ),
            "materialized_linker_length": None,
            "evaluated_link_instances": [
                item
                for segment in segments
                for item in segment.contour_preflight[
                    "evaluated_link_instances"
                ]
            ],
        }

    fixed_atoms: dict[str, str] = {}
    if terminal_path is not None:
        fragment = instances.fragments[
            terminal_path.anchor_fragment_instance_id
        ]
        fixed_atoms[terminal_path.selector] = _rfd3_atom_selection(
            spec.fragments[fragment.source_id].fixed_atoms
        )
        (
            motif_constraint_groups,
            motif_constraint_orbits,
        ) = _runtime_fixed_motif_constraints(
            instances=instances,
            mapping=mapping,
            anchor_fragment_instance_ids=(
                terminal_path.anchor_fragment_instance_id
            ),
            transform_set=transform_set,
            runtime_transform_order=registry_transform_order,
            transform_to_runtime_representative=(
                transform_to_runtime_representative
            ),
        )
    else:
        for segment in segments:
            for selector, fragment_instance_id in zip(
                segment.selectors,
                segment.fragment_instance_ids,
                strict=True,
            ):
                fragment = instances.fragments[fragment_instance_id]
                fixed_atoms[selector] = _rfd3_atom_selection(
                    spec.fragments[fragment.source_id].fixed_atoms
                )
        use_interface_constraint_groups = (
            spec.constraint_group_strategy == "interface_edges"
            or (
                spec.constraint_group_strategy == "auto"
                and bool(instances.interfaces)
            )
        )
        if preexpanded_stabilizer_asu:
            if stabilized_segment_transform_ids is None:
                raise ValueError(
                    "Preexpanded stabilized ASU lost its path transforms"
                )
            (
                motif_constraint_groups,
                motif_constraint_orbits,
            ) = _runtime_preexpanded_stabilizer_constraints(
                instances=instances,
                mapping=mapping,
                segments=segments,
                segment_transform_ids=(
                    stabilized_segment_transform_ids
                ),
                registry_transform_order=registry_transform_order,
            )
        elif use_interface_constraint_groups:
            motif_constraint_groups = _runtime_interface_constraint_groups(
                instances,
                mapping,
                list(compiled_links),
                transform_set,
                runtime_transform_order=registry_transform_order,
                transform_to_runtime_representative=(
                    transform_to_runtime_representative
                ),
                preexpanded_mixed_orbits=preexpanded_runtime_layout,
            )
            motif_constraint_orbits = _runtime_interface_constraint_orbits(
                motif_constraint_groups,
                instances=instances,
                transform_set=transform_set,
                runtime_transform_order=registry_transform_order,
                transform_to_runtime_representative=(
                    transform_to_runtime_representative
                ),
            )
        else:
            path_fragment_ids = tuple(
                dict.fromkeys(
                    fragment_id
                    for segment in segments
                    for fragment_id in segment.fragment_instance_ids
                )
            )
            (
                motif_constraint_groups,
                motif_constraint_orbits,
            ) = _runtime_fixed_motif_constraints(
                instances=instances,
                mapping=mapping,
                anchor_fragment_instance_ids=path_fragment_ids,
                transform_set=transform_set,
                runtime_transform_order=registry_transform_order,
                transform_to_runtime_representative=(
                    transform_to_runtime_representative
                ),
            )
    legacy_link = compiled_links[0] if len(compiled_links) == 1 else None
    selected_orbit_ids = (
        {terminal_path.orbit_id}
        if terminal_path is not None
        else {link.orbit_id for link in compiled_links}
    )
    configured_linker_ranges = {
        entry.link.id: [
            entry.link.minimum_length,
            entry.link.maximum_length,
        ]
        for entry in materialized_links
    }
    materialized_linker_lengths = {
        entry.link.id: entry.materialized_linker_length
        for entry in materialized_links
    }
    linker_length_policies = {
        entry.link.id: entry.linker_length_policy
        for entry in materialized_links
    }
    linker_contour_preflights = {
        entry.link.id: entry.contour_preflight
        for entry in materialized_links
    }
    symmetry = {
        "id": symmetry_id,
        "is_symmetric_motif": True,
    }
    # A compiler-owned multi-chain ASU may contain identical entities more
    # than once. Foundry's legacy frame recovery counts those occurrences as
    # additional symmetry copies. Mosaic has already validated the exact
    # transform registry, so request declared frames only for that case. Keep
    # the legacy single-chain JSON byte-level schema free of a redundant false
    # field.
    if (
        finite_action is not None
        or asu_chain_count > 1
        or terminal_path is not None
        or not instances.interfaces
        or transform_set.type
        in {
            SymmetryType.TETRAHEDRAL,
            SymmetryType.OCTAHEDRAL,
            SymmetryType.ICOSAHEDRAL,
        }
    ):
        symmetry["use_declared_frames"] = True
        if finite_action is not None and not preexpanded_stabilizer_asu:
            symmetry["declared_action_is_quotient"] = True
        symmetry["declared_transform_order"] = registry_transform_order
        symmetry["declared_transform_matrices"] = (
            registry_transform_matrices
        )
        if preexpanded_chain_layout is not None:
            symmetry["declared_preexpanded_chain_layout"] = (
                preexpanded_chain_layout
            )
    adapter_extra = {
        "compiler": "rfd3_mosaic.static_adapter",
        "native_compiler_path": "assembly_ir_to_rfd3_features",
        "pose_seed": (
            spec.random_seed if pose_seed is None else pose_seed
        ),
        "pose_source": (
            "candidate_manifest"
            if candidate_manifest_path is not None
            else "random_seed"
        ),
        "pose_candidate_manifest": (
            str(candidate_manifest_path)
            if candidate_manifest_path is not None
            else None
        ),
        "pose_candidate_structure_sha256": candidate_structure_sha,
        "adapter_structure_sha256": rebuilt_structure_sha,
        "assembly_config": str(config),
        # Compatibility key consumed by the existing interface audit.
        "interface_seed_config": str(config),
        "asu_scaffold_link_instance": (
            legacy_link.id if legacy_link is not None else None
        ),
        "asu_source_copy_index": (
            legacy_link.copy_index if legacy_link is not None else None
        ),
        "asu_target_copy_index": (
            legacy_link.target_copy_index if legacy_link is not None else None
        ),
        "asu_scaffold_link_instances": [
            link.id for link in compiled_links
        ],
        "asu_scaffold_segments": [
            {
                "link_instance_id": (
                    segment.links[0].link.id
                    if len(segment.links) == 1
                    else None
                ),
                "source_link_id": (
                    segment.links[0].link.source_id
                    if len(segment.links) == 1
                    else None
                ),
                "link_instance_ids": [
                    entry.link.id for entry in segment.links
                ],
                "source_link_ids": [
                    entry.link.source_id for entry in segment.links
                ],
                "source_copy_index": segment.links[0].link.copy_index,
                "target_copy_index": segment.links[-1].link.target_copy_index,
                "from_selector": segment.from_selector,
                "to_selector": segment.to_selector,
                "path_selectors": list(segment.selectors),
                "contig_chains": list(segment.contig_chains),
            }
            for segment in segments
        ],
        "asu_terminal_extensions": (
            [
                {
                    "extension_instance_id": extension.id,
                    "source_extension_id": extension.source_id,
                    "terminus": extension.anchor_terminus.value,
                    "configured_length_range": [
                        extension.minimum_length,
                        extension.maximum_length,
                    ],
                    "materialized_length": _materialize_length(
                        extension.minimum_length,
                        extension.maximum_length,
                        label=extension.id,
                    ),
                }
                for extension in (
                    terminal_path.n_extension,
                    terminal_path.c_extension,
                )
                if extension is not None
            ]
            if terminal_path is not None
            else []
        ),
        "symmetry_multiplicity": symmetry_multiplicity,
        "full_symmetry_multiplicity": full_symmetry_multiplicity,
        "symmetry_action_kind": (
            "mixed_stabilizer_quotients"
            if preexpanded_mixed_orbits
            else "preexpanded_stabilized_asu"
            if preexpanded_stabilizer_asu
            else "stabilizer_quotient"
            if finite_action is not None
            else "regular_full_group"
        ),
        "preexpanded_chain_layout": preexpanded_chain_layout,
        "finite_orbit_action": (
            finite_action.model_dump(mode="json")
            if finite_action is not None
            else None
        ),
        "asu_chain_count": asu_chain_count,
        "scaffold_mode": scaffold_mode,
        "configured_linker_length_range": configured_linker_range,
        "materialized_linker_length": materialized_linker_length,
        "linker_length_policy": linker_length_policy,
        "configured_linker_length_ranges": configured_linker_ranges,
        "materialized_linker_lengths": materialized_linker_lengths,
        "linker_length_policies": linker_length_policies,
        "contig_linker_is_deterministic": True,
        "materialized_linker_contour_preflight": (
            linker_contour_preflight
        ),
        "materialized_linker_contour_preflights": (
            linker_contour_preflights
        ),
        "registry_preflight": "passed",
        "registry_transform_order": registry_transform_order,
        "registry_transform_matrices": registry_transform_matrices,
        "full_registry_transform_order": full_registry_transform_order,
        "full_registry_transform_matrices": (
            full_registry_transform_matrices
        ),
        "mosaic_transform_order": (
            list(registry_transform_order)
            if preexpanded_stabilizer_asu
            else list(
                dict.fromkeys(
                    group.transform_id
                    for group in instances.motion_groups.values()
                    if group.orbit_id in selected_orbit_ids
                )
            )
        ),
        "motif_constraint_groups": motif_constraint_groups,
        "motif_constraint_orbits": motif_constraint_orbits,
        "assembly_interface_relations": (
            _runtime_interface_relation_audit_plan(
                instances=instances,
                mapping=mapping,
                preexpanded_mixed_orbits=preexpanded_runtime_layout,
            )
            if instances.interfaces
            else []
        ),
    }
    if instances.interfaces and compiled_links:
        adapter_extra["interleaved_interface_seed_topology"] = (
            analyze_interleaved_interface_seed_topology(
                instances
            ).to_dict()
        )
    if extra_metadata:
        protected = set(adapter_extra) & set(extra_metadata)
        if protected:
            raise ValueError(
                "extra_metadata cannot overwrite compiler-owned fields: "
                f"{sorted(protected)}"
            )
        adapter_extra.update(extra_metadata)

    payload = {
        example_id: {
            "dialect": 2,
            "input": standalone.structure_path.name,
            "contig": contig,
            "select_fixed_atoms": fixed_atoms,
            "redesign_motif_sidechains": False,
            "is_non_loopy": is_non_loopy,
            "symmetry": symmetry,
            "extra": adapter_extra,
        }
    }
    if finite_action is not None or preexpanded_mixed_orbits:
        # RFD3 otherwise centers fixed-motif inputs on their fixed-atom COM.
        # A complete group orbit has its COM at the group origin, whereas a
        # stabilizer quotient generally does not.  Applying that translation
        # without conjugating the declared frames changes the represented
        # action and invalidates exact fixed-target projection.  The compiler
        # registry matrices are expressed about the group-frame origin.
        payload[example_id]["ori_token"] = [0.0, 0.0, 0.0]
    input_path = output / "rfd3_input.json"
    input_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RFD3AdapterOutputs(
        input_path=input_path,
        structure_path=standalone.structure_path,
        mapping_path=standalone.mapping_path,
        manifest_path=standalone.manifest_path,
        example_id=example_id,
        contig=contig,
    )


def compile_rfd3_input(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    base_directory: str | Path = ".",
    example_id: str = "lhd101_c3_interface_seed",
    is_non_loopy: bool = True,
    pose_seed: int | None = None,
    pose_candidate_manifest: str | Path | None = None,
    linker_length: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> RFD3AdapterOutputs:
    """Compatibility name for the native Assembly IR emitter."""

    return compile_assembly_rfd3_input(
        config_path,
        output_directory,
        base_directory=base_directory,
        example_id=example_id,
        is_non_loopy=is_non_loopy,
        pose_seed=pose_seed,
        pose_candidate_manifest=pose_candidate_manifest,
        linker_length=linker_length,
        extra_metadata=extra_metadata,
    )
