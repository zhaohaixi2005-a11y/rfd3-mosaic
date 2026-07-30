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
    if multiplicity > 10:
        raise ValueError(
            f"Native RFD3 symmetric-motif inference currently allows at "
            f"most 10 transforms, but {symmetry_id} requires {multiplicity}"
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
    link,
    transform_set,
) -> list[dict[str, Any]]:
    """Describe complete cross-chain groups in post-symmetry RFD3 terms."""

    selector_by_source_id: dict[str, str] = {}
    canonical_transform_by_source_id: dict[str, str] = {}
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
    spec,
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

        mobility = spec.interfaces[source_interface_id].mobility
        bounds = mobility.bounds
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
) -> RFD3AdapterOutputs:
    """Emit a native-symmetry RFD3 task from one orbit of scaffold links.

    RFD3 constructs one asymmetric unit from the contig and then applies its
    symmetry sampler.  Consequently, only the copy-zero scaffold path belongs
    in the contig; the pre-symmetrized CIF supplies the complete motif geometry
    used by RFD3 to infer the symmetry frames.
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

    orbit_links = [
        link
        for link in instances.scaffold_links.values()
        if link.copy_index == 0
    ]
    if len(orbit_links) != 1:
        raise NotImplementedError(
            "The static RFD3 adapter currently requires exactly one "
            "copy-zero scaffold relation"
        )
    link = orbit_links[0]
    if link.orbit_id is None:
        raise ValueError("Native symmetry requires an orbit-expanded link")

    orbit = spec.symmetry.orbits[link.orbit_id]
    transform_set = spec.symmetry.transform_sets[orbit.transform_set]
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

    from_selector = _fragment_selector(
        mapping, link.from_fragment_instance_id
    )
    to_selector = _fragment_selector(mapping, link.to_fragment_instance_id)
    configured_linker_range = [
        link.minimum_length,
        link.maximum_length,
    ]
    materialized_linker_length: int | None
    linker_length_policy: str
    if link.chain_break:
        if linker_length is not None:
            raise ValueError(
                "linker_length cannot be set for a chain-break scaffold link"
            )
        contig = f"{from_selector},/0,{to_selector}"
        scaffold_mode = "independent_chains"
        asu_chain_count = 2
        materialized_linker_length = None
        linker_length_policy = "not_applicable"
    else:
        if linker_length is None:
            materialized_linker_length = (
                link.minimum_length + link.maximum_length
            ) // 2
            linker_length_policy = "configured_range_midpoint"
        else:
            if isinstance(linker_length, bool) or not isinstance(
                linker_length, int
            ):
                raise TypeError("linker_length must be an integer")
            materialized_linker_length = linker_length
            linker_length_policy = "explicit"
        if not (
            link.minimum_length
            <= materialized_linker_length
            <= link.maximum_length
        ):
            raise ValueError(
                "linker_length must fall inside the configured range "
                f"[{link.minimum_length}, {link.maximum_length}], got "
                f"{materialized_linker_length}"
            )
        # Foundry samples an N-M contig range whenever it builds an input.
        # Adapter prevalidation and inference run in separate processes, so
        # leaving a range here can silently produce different AtomArrays.
        # N-N keeps the native generated-residue grammar while making both
        # builds identical.
        linker = (
            f"{materialized_linker_length}-"
            f"{materialized_linker_length}"
        )
        contig = f"{from_selector},{linker},{to_selector}"
        scaffold_mode = "continuous_linker"
        asu_chain_count = 1

    linker_contour_preflight = _materialized_linker_contour_preflight(
        standalone.manifest_path,
        source_link_id=link.source_id,
        materialized_length=materialized_linker_length,
    )

    from_fragment = instances.fragments[link.from_fragment_instance_id]
    to_fragment = instances.fragments[link.to_fragment_instance_id]
    fixed_atoms = {
        from_selector: _rfd3_atom_selection(
            spec.fragments[from_fragment.source_id].fixed_atoms
        ),
        to_selector: _rfd3_atom_selection(
            spec.fragments[to_fragment.source_id].fixed_atoms
        ),
    }
    motif_constraint_groups = _runtime_interface_constraint_groups(
        instances,
        mapping,
        link,
        transform_set,
    )
    motif_constraint_orbits = _runtime_interface_constraint_orbits(
        motif_constraint_groups,
        spec=spec,
        transform_set=transform_set,
    )
    payload = {
        example_id: {
            "dialect": 2,
            "input": standalone.structure_path.name,
            "contig": contig,
            "select_fixed_atoms": fixed_atoms,
            "redesign_motif_sidechains": False,
            "is_non_loopy": is_non_loopy,
            "symmetry": {
                "id": symmetry_id,
                "is_symmetric_motif": True,
            },
            "extra": {
                "compiler": "rfd3_mosaic.static_adapter",
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
                "interface_seed_config": str(config),
                "asu_scaffold_link_instance": link.id,
                "asu_source_copy_index": link.copy_index,
                "asu_target_copy_index": link.target_copy_index,
                "symmetry_multiplicity": symmetry_multiplicity,
                "asu_chain_count": asu_chain_count,
                "scaffold_mode": scaffold_mode,
                "configured_linker_length_range": (
                    configured_linker_range
                ),
                "materialized_linker_length": (
                    materialized_linker_length
                ),
                "linker_length_policy": linker_length_policy,
                "contig_linker_is_deterministic": True,
                "materialized_linker_contour_preflight": (
                    linker_contour_preflight
                ),
                "registry_preflight": "passed",
                "registry_transform_order": registry_transform_order,
                "registry_transform_matrices": (
                    registry_transform_matrices
                ),
                "mosaic_transform_order": list(
                    dict.fromkeys(
                        group.transform_id
                        for group in instances.motion_groups.values()
                        if group.orbit_id == link.orbit_id
                    )
                ),
                "motif_constraint_groups": motif_constraint_groups,
                "motif_constraint_orbits": motif_constraint_orbits,
            },
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
