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
class _ASUScaffoldSegment:
    """One deterministic scaffold segment emitted as one ASU chain."""

    link: Any
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
    """Reject incomplete, duplicated, improper, or non-closed Cn/Dn sets."""

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


def _runtime_interface_constraint_groups(
    instances,
    mapping: dict[str, Any],
    links,
    transform_set,
) -> list[dict[str, Any]]:
    """Describe complete cross-chain groups in post-symmetry RFD3 terms."""

    selector_by_source_id: dict[str, str] = {}
    canonical_transform_by_source_id: dict[str, str] = {}
    for link in links:
        for fragment_instance_id in (
            link.from_fragment_instance_id,
            link.to_fragment_instance_id,
        ):
            fragment = instances.fragments[fragment_instance_id]
            selector = _fragment_selector(mapping, fragment_instance_id)
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

    def runtime_transform_id(fragment) -> int:
        canonical_transform_id = canonical_transform_by_source_id[
            fragment.source_id
        ]
        for index, relation_id in enumerate(registry.transform_ids):
            if (
                registry.compose_ids(
                    relation_id,
                    canonical_transform_id,
                )
                == fragment.transform_id
            ):
                return index
        raise ValueError(
            "Could not resolve native runtime transform for fragment "
            f"{fragment.id!r}"
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
                try:
                    source_component = selector_by_source_id[
                        fragment.source_id
                    ]
                except KeyError as error:
                    raise NotImplementedError(
                        "Runtime motif groups require every interface "
                        "fragment source to be present in the canonical ASU "
                        "scaffold link"
                    ) from error
                members.append(
                    {
                        "role": role,
                        "source_fragment_id": fragment.source_id,
                        "src_components": _selector_source_components(
                            source_component
                        ),
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
                "orbit_id": edge.orbit_id,
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
) -> list[dict[str, Any]]:
    """Resolve one master group and one group action per interface orbit."""

    registry = build_transform_registry(transform_set)
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
        masters = [
            group
            for group in orbit_groups
            if int(group["source_copy_index"]) == 0
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
            key=lambda item: int(item["source_copy_index"]),
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
            matching_actions = []
            for action_index, action_id in enumerate(
                registry.transform_ids
            ):
                if all(
                    registry.compose_ids(
                        action_id,
                        registry.transform_ids[
                            master_transform_id
                        ],
                    )
                    == registry.transform_ids[
                        target_members[member_key]
                    ]
                    for member_key, master_transform_id
                    in master_members.items()
                ):
                    matching_actions.append(
                        (action_index, action_id)
                    )
            if len(matching_actions) != 1:
                raise ValueError(
                    f"Constraint group {group['group_id']!r} has "
                    f"{len(matching_actions)} compatible group actions; "
                    "expected exactly one"
                )
            action_index, action_id = matching_actions[0]
            group["constraint_orbit_id"] = constraint_orbit_id
            group["orbit_transform_id"] = action_index
            group["orbit_registry_transform_id"] = action_id
            group_transform_ids.append(action_index)
            group_registry_transform_ids.append(action_id)

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
    if (
        any(fragment.copy_index != 0 for fragment in masters)
        or None in orbit_ids
        or len(orbit_ids) != 1
    ):
        raise ValueError(
            "A symmetric fixed constraint path requires copy-zero anchors "
            "from one orbit"
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
    single_component = len(masters_by_component) == 1
    for component_id, component_masters in masters_by_component.items():
        mobility = compiled_orbit.component_mobility.get(
            component_id,
            compiled_orbit.mobility,
        )
        bounds = mobility.bounds
        schedule = mobility.effective_schedule
        source_components = {
            master.source_id: _selector_source_components(
                _fragment_selector(mapping, master.id)
            )
            for master in component_masters
        }
        copies_by_source_id = {
            master.source_id: sorted(
                (
                    fragment
                    for fragment in instances.fragments.values()
                    if fragment.source_id == master.source_id
                    and fragment.orbit_id == orbit_id
                ),
                key=lambda fragment: fragment.copy_index,
            )
            for master in component_masters
        }
        for source_id, source_copies in copies_by_source_id.items():
            if len(source_copies) != registry.order:
                raise ValueError(
                    f"Fixed motif {source_id!r} has {len(source_copies)} "
                    f"copies but symmetry registry has {registry.order} "
                    "transforms"
                )

        runtime_orbit_id = (
            orbit_id
            if single_component
            else f"{orbit_id}__{component_id}"
        )
        component_groups: list[dict[str, Any]] = []
        group_transform_ids: list[int] = []
        group_registry_transform_ids: list[str] = []
        for copy_index in range(registry.order):
            copy_fragments = [
                copies_by_source_id[master.source_id][copy_index]
                for master in component_masters
            ]
            matches = [
                (index, action_id)
                for index, action_id in enumerate(registry.transform_ids)
                if all(
                    registry.compose_ids(action_id, master.transform_id)
                    == fragment.transform_id
                    for master, fragment in zip(
                        component_masters, copy_fragments
                    )
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Could not resolve one group action for fixed "
                    f"component {component_id!r} copy {copy_index}"
                )
            transform_index, transform_id = matches[0]
            group_id = f"fixed@{runtime_orbit_id}[{copy_index}]"
            group = {
                "group_id": group_id,
                "constraint_kind": "fixed_motif",
                "geometry_lock": "joint_rigid",
                "coupling_group_id": component_id,
                "constraint_orbit_id": runtime_orbit_id,
                "orbit_id": orbit_id,
                "members": [
                    {
                        "role": "motif",
                        "source_fragment_id": fragment.source_id,
                        "src_components": source_components[
                            fragment.source_id
                        ],
                        "sym_transform_id": transform_index,
                    }
                    for fragment in copy_fragments
                ],
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
        link_reports = manifest["validation"][
            "scaffold_link_geometry"
        ]["links"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Standalone manifest is missing scaffold-link geometry"
        ) from error
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


def _compile_asu_scaffold_segments(
    links,
    *,
    mapping: dict[str, Any],
    manifest_path: Path,
    linker_length: int | None,
) -> tuple[_ASUScaffoldSegment, ...]:
    """Compile disjoint copy-zero links into deterministic ASU chains.

    This first multi-interface adapter stage intentionally supports disjoint
    segments only.  A fragment that participates in two links requires a
    later ordered-path compiler; emitting it twice in an RFD3 contig would
    silently duplicate fixed motif atoms.
    """

    segments: list[_ASUScaffoldSegment] = []
    selector_owners: dict[str, str] = {}
    for link in sorted(links, key=lambda item: item.id):
        from_selector = _fragment_selector(
            mapping,
            link.from_fragment_instance_id,
        )
        to_selector = _fragment_selector(
            mapping,
            link.to_fragment_instance_id,
        )
        for selector in (from_selector, to_selector):
            previous = selector_owners.get(selector)
            if previous is not None:
                raise NotImplementedError(
                    "Multi-link RFD3 contigs currently require disjoint "
                    "fixed-fragment segments; selector "
                    f"{selector!r} is used by both {previous!r} and "
                    f"{link.id!r}"
                )
            selector_owners[selector] = link.id

        if link.chain_break:
            if linker_length is not None:
                raise ValueError(
                    "linker_length cannot be set when an ASU scaffold "
                    "segment is a chain break"
                )
            contig_chains = (from_selector, to_selector)
            materialized_length = None
            linker_policy = "not_applicable"
        else:
            if linker_length is None:
                materialized_length = (
                    link.minimum_length + link.maximum_length
                ) // 2
                linker_policy = "configured_range_midpoint"
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
                raise ValueError(
                    "linker_length must fall inside the configured range "
                    f"[{link.minimum_length}, {link.maximum_length}] for "
                    f"{link.id!r}, got {materialized_length}"
                )
            linker = f"{materialized_length}-{materialized_length}"
            contig_chains = (
                f"{from_selector},{linker},{to_selector}",
            )

        contour_preflight = _materialized_linker_contour_preflight(
            manifest_path,
            source_link_id=link.source_id,
            materialized_length=materialized_length,
        )
        segments.append(
            _ASUScaffoldSegment(
                link=link,
                from_selector=from_selector,
                to_selector=to_selector,
                contig_chains=contig_chains,
                materialized_linker_length=materialized_length,
                linker_length_policy=linker_policy,
                contour_preflight=contour_preflight,
            )
        )
    return tuple(segments)


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
    spec = load_interface_seed_config(config)
    instances = expand_symmetry_instances(
        spec,
        master_transforms=None,
    )
    mapping = _load_json(standalone.mapping_path)

    link_by_id = {
        link.id: link
        for link in instances.scaffold_links.values()
        if link.copy_index == 0
    }
    for segment in instances.generated_segments.values():
        if (
            isinstance(segment, ScaffoldLinkInstance)
            and segment.copy_index == 0
        ):
            link_by_id.setdefault(segment.id, segment)
    orbit_links = list(link_by_id.values())
    terminal_extensions = [
        segment
        for segment in instances.generated_segments.values()
        if isinstance(segment, TerminalExtensionInstance)
        and segment.copy_index == 0
    ]
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
    symmetry_id, symmetry_multiplicity = (
        _native_symmetry_id_and_multiplicity(transform_set)
    )
    registry_transform_order = _preflight_native_transform_registry(
        transform_set,
        symmetry_multiplicity,
    )
    native_registry = build_transform_registry(transform_set)
    registry_transform_matrices = {
        transform_id: native_registry.transform(transform_id).tolist()
        for transform_id in native_registry.transform_ids
    }

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

    if terminal_path is None and len(segments) == 1:
        only_segment = segments[0]
        scaffold_mode = (
            "independent_chains"
            if only_segment.link.chain_break
            else "continuous_linker"
        )
        configured_linker_range = [
            only_segment.link.minimum_length,
            only_segment.link.maximum_length,
        ]
        materialized_linker_length = (
            only_segment.materialized_linker_length
        )
        linker_length_policy = only_segment.linker_length_policy
        linker_contour_preflight = only_segment.contour_preflight
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
        )
    else:
        for segment in segments:
            for selector, fragment_instance_id in (
                (
                    segment.from_selector,
                    segment.link.from_fragment_instance_id,
                ),
                (
                    segment.to_selector,
                    segment.link.to_fragment_instance_id,
                ),
            ):
                fragment = instances.fragments[fragment_instance_id]
                fixed_atoms[selector] = _rfd3_atom_selection(
                    spec.fragments[fragment.source_id].fixed_atoms
                )
        if instances.interfaces:
            motif_constraint_groups = _runtime_interface_constraint_groups(
                instances,
                mapping,
                [segment.link for segment in segments],
                transform_set,
            )
            motif_constraint_orbits = _runtime_interface_constraint_orbits(
                motif_constraint_groups,
                instances=instances,
                transform_set=transform_set,
            )
        else:
            anchor_fragment_ids = tuple(
                dict.fromkeys(
                    fragment_id
                    for segment in segments
                    for fragment_id in (
                        segment.link.from_fragment_instance_id,
                        segment.link.to_fragment_instance_id,
                    )
                )
            )
            (
                motif_constraint_groups,
                motif_constraint_orbits,
            ) = _runtime_fixed_motif_constraints(
                instances=instances,
                mapping=mapping,
                anchor_fragment_instance_ids=anchor_fragment_ids,
                transform_set=transform_set,
            )
    legacy_link = segments[0].link if len(segments) == 1 else None
    selected_orbit_ids = (
        {terminal_path.orbit_id}
        if terminal_path is not None
        else {segment.link.orbit_id for segment in segments}
    )
    configured_linker_ranges = {
        segment.link.id: [
            segment.link.minimum_length,
            segment.link.maximum_length,
        ]
        for segment in segments
    }
    materialized_linker_lengths = {
        segment.link.id: segment.materialized_linker_length
        for segment in segments
    }
    linker_length_policies = {
        segment.link.id: segment.linker_length_policy
        for segment in segments
    }
    linker_contour_preflights = {
        segment.link.id: segment.contour_preflight
        for segment in segments
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
        asu_chain_count > 1
        or terminal_path is not None
        or not instances.interfaces
    ):
        symmetry["use_declared_frames"] = True
        symmetry["declared_transform_order"] = registry_transform_order
        symmetry["declared_transform_matrices"] = (
            registry_transform_matrices
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
            segment.link.id for segment in segments
        ],
        "asu_scaffold_segments": [
            {
                "link_instance_id": segment.link.id,
                "source_link_id": segment.link.source_id,
                "source_copy_index": segment.link.copy_index,
                "target_copy_index": segment.link.target_copy_index,
                "from_selector": segment.from_selector,
                "to_selector": segment.to_selector,
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
        "mosaic_transform_order": list(
            dict.fromkeys(
                group.transform_id
                for group in instances.motion_groups.values()
                if group.orbit_id in selected_orbit_ids
            )
        ),
        "motif_constraint_groups": motif_constraint_groups,
        "motif_constraint_orbits": motif_constraint_orbits,
    }
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
