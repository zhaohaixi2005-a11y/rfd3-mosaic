"""Validate and install a complete-scaffold partial-diffusion input artifact.

This module changes initialization, not the learned model or guidance loss.
Artifacts are bound to a compiler contract and may remap only an explicit
set of native fields.  The first supported workflow is locked, regular full
Cn/Dn assemblies with a complete, already expanded scaffold.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from rfd3_mosaic.structure import read_structure_atoms


_NATIVE_FIELDS = {
    "contig",
    "select_fixed_atoms",
    "select_unfixed_sequence",
    "symmetry",
    "extra",
}
_EXTRA_FIELDS = {
    "motif_constraint_groups",
    "motif_constraint_orbits",
    "assembly_interface_relations",
    "asu_scaffold_segments",
    "asu_terminal_extensions",
    "preexpanded_chain_layout",
    "asu_chain_count",
    "asu_polymer_chain_count",
    "generated_coordinate_initialization",
    "mosaic_scaffold_contract",
    "mosaic_reference_transport",
}
_BINDING_EXTRA_FIELDS = (
    "adapter_structure_sha256",
    "full_standalone_structure_sha256",
    "symmetry_action_kind",
    "registry_transform_order",
    "registry_transform_matrices",
    "materialized_linker_lengths",
    "materialized_linker_length",
    "asu_terminal_extensions",
    "asu_scaffold_segments",
    "motif_constraint_groups",
    "motif_constraint_orbits",
    "assembly_interface_relations",
    "cylindrical_constraints",
)
_MEMBER_REMAP_FIELDS = {
    "src_components",
    "correspondence_components",
    "sym_transform_id",
}
_SEED_MAXIMUM_ERROR = 0.01
_SEED_RMSD_TOLERANCE = 0.005
_AMINO_ACIDS = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)


def scaffold_transport_plan(native, registry_extra, residues):
    from rfd3_mosaic.validation.reference_transport import build_reference_transport_plan

    fixed = _fixed_selection(native["select_fixed_atoms"], residues)
    records = [{"chain_id": residue["chain"], "residue_number": residue["number"],
                "atom_name": atom, "coordinate": np.asarray(residue["atoms"][atom]).tolist()}
               for component, residue in residues.items() if component in fixed
               for atom in sorted(fixed[component])]
    extra = native["extra"]
    return build_reference_transport_plan(extra["mosaic_scaffold_contract"],
        extra["motif_constraint_groups"], extra["motif_constraint_orbits"],
        registry_extra["registry_transform_order"], registry_extra["registry_transform_matrices"], records)


def _example(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Scaffold handoff requires exactly one compiled example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled example must be an object")
    return example


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def compiled_scaffold_contract_sha256(payload: dict[str, Any]) -> str:
    """Fingerprint graph, pose, atom policies and lengths, excluding run paths."""

    example = _example(payload)
    extra = example.get("extra") or {}
    contract = {
        "native": {
            key: example.get(key)
            for key in (
                "dialect",
                "contig",
                "select_fixed_atoms",
                "select_unfixed_sequence",
                "redesign_motif_sidechains",
                "symmetry",
            )
        },
        "extra": {key: extra.get(key) for key in _BINDING_EXTRA_FIELDS},
    }
    return hashlib.sha256(_canonical_bytes(contract)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _components(selection: str, *, chain_breaks: bool = False) -> list[str]:
    if not isinstance(selection, str) or not selection:
        raise ValueError(
            "Scaffold residue selections must be explicit nonempty strings"
        )
    result = []
    for token in selection.split(","):
        if chain_breaks and token == "/0":
            continue
        match = re.fullmatch(r"([A-Za-z]+)(-?\d+)(?:-(-?\d+))?", token)
        if match is None:
            raise ValueError(
                f"Scaffold selections cannot contain generated lengths: {token!r}"
            )
        chain, start, end = match.groups()
        first, last = int(start), int(end) if end is not None else int(start)
        if last < first:
            raise ValueError("Scaffold residue ranges must be increasing")
        result.extend(f"{chain}{residue}" for residue in range(first, last + 1))
    if len(result) != len(set(result)):
        raise ValueError("Scaffold residue selection contains duplicates")
    return result


def _residues(path: Path) -> dict[str, dict[str, Any]]:
    atoms = read_structure_atoms(path, mmcif_identifier_namespace="label")
    residues: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        if atom.record_type != "ATOM" or atom.insertion_code:
            raise ValueError(
                "Complete-scaffold inputs require protein ATOM records without insertion codes"
            )
        if atom.residue_name not in _AMINO_ACIDS:
            raise ValueError(
                "Complete-scaffold inputs currently require standard protein residues"
            )
        if not all(math.isfinite(value) for value in atom.coordinate):
            raise ValueError("Complete scaffold contains nonfinite coordinates")
        component = f"{atom.chain_id}{atom.residue_number}"
        residue = residues.setdefault(
            component,
            {
                "chain": atom.chain_id,
                "number": atom.residue_number,
                "name": atom.residue_name,
                "atoms": {},
            },
        )
        if atom.atom_name in residue["atoms"] or residue["name"] != atom.residue_name:
            raise ValueError("Complete scaffold has ambiguous residue/atom identities")
        residue["atoms"][atom.atom_name] = np.asarray(atom.coordinate, dtype=float)
    return residues


def _fixed_selection(selection: Any, residues: dict[str, dict]) -> dict[str, set[str]]:
    if not isinstance(selection, dict) or not selection:
        raise ValueError("Scaffold handoff requires explicit select_fixed_atoms")
    result: dict[str, set[str]] = {}
    for selector, atom_selection in selection.items():
        for component in _components(selector):
            if component not in residues or component in result:
                raise ValueError(f"Fixed residue absent or selected twice: {component}")
            atoms = residues[component]["atoms"]
            if atom_selection == "ALL":
                selected = set(atoms)
            elif atom_selection == "BKBN":
                selected = {"N", "CA", "C", "O"}
            elif isinstance(atom_selection, str):
                selected = set(atom_selection.split(","))
            else:
                raise ValueError(
                    "Fixed atom selectors must be ALL, BKBN or explicit atom names"
                )
            if not selected or not selected <= set(atoms) or "CA" not in selected:
                raise ValueError(f"Incomplete fixed atom selection for {component}")
            result[component] = selected
    return result


def _compiled_contig_chains(contig: str) -> list[list[str | None]]:
    """Expand exact compiler lengths without interpreting a native random range."""
    chains: list[list[str | None]] = [[]]
    for token in contig.split(","):
        if token == "/0":
            if not chains[-1]:
                raise ValueError("Compiled contig contains an empty chain")
            chains.append([])
            continue
        numeric = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if numeric:
            first, last = numeric.groups()
            if last is not None and int(first) != int(last):
                raise ValueError(
                    "Scaffold handoff requires materialized exact generated lengths"
                )
            chains[-1].extend([None] * int(first))
        else:
            chains[-1].extend(_components(token))
    if any(not chain for chain in chains):
        raise ValueError("Compiled contig contains an empty chain")
    return chains


def compiled_scaffold_residue_map(
    payload: dict, *, chain_layout: list[dict], chain_ids: list[str]
) -> dict[tuple[str, int], str]:
    """Map (canonical source component, symmetry index) to complete selectors.

    Complete input chains are numbered from one; sorted entity IDs correspond
    to the compiler's ASU contig chain order. Generated slots affect numbering
    but have no source selector. The preparer and validator share this mapping.
    """
    compiled = _compiled_contig_chains(_example(payload)["contig"])
    entities = sorted({record["entity_id"] for record in chain_layout})
    if len(entities) != len(compiled) or len(chain_layout) != len(chain_ids):
        raise ValueError("Complete scaffold changes the number of compiled ASU chains")
    by_entity = dict(zip(entities, compiled, strict=True))
    result = {}
    for chain, record in zip(chain_ids, chain_layout, strict=True):
        for number, source in enumerate(by_entity[record["entity_id"]], start=1):
            if source is None:
                continue
            key = (source, record["transform_index"])
            if key in result:
                raise ValueError(
                    "Compiled source residue occurs twice in one physical copy"
                )
            result[key] = f"{chain}{number}"
    return result


def _validate_chain_patterns(
    base: dict,
    residues: dict,
    fixed: dict,
    source: dict,
    source_fixed: dict,
    layout: list,
) -> None:
    chains = list(dict.fromkeys(residue["chain"] for residue in residues.values()))
    compiled = _compiled_contig_chains(base["contig"])
    entities = sorted({record["entity_id"] for record in layout})
    if len(entities) != len(compiled):
        raise ValueError("Complete scaffold changes the number of compiled ASU chains")
    by_entity = dict(zip(entities, compiled, strict=True))
    for chain, record in zip(chains, layout, strict=True):
        observed = [
            key for key, residue in residues.items() if residue["chain"] == chain
        ]
        expected = by_entity[record["entity_id"]]
        if len(observed) != len(expected):
            raise ValueError(
                "Complete scaffold changes a compiled per-chain generated length"
            )
        if [residues[component]["number"] for component in observed] != list(
            range(1, len(expected) + 1)
        ):
            raise ValueError(
                "Complete scaffold chain residue numbers must start at one without gaps"
            )
        for component, original in zip(observed, expected, strict=True):
            if original is not None and original not in source:
                raise ValueError("Compiled contig refers to a missing source residue")
            if (component in fixed) != (original in source_fixed):
                raise ValueError(
                    "Complete scaffold changes the compiled fixed/generated residue order"
                )


def _validate_symmetry(
    base: dict, native: dict, residues: dict
) -> tuple[list, dict, list]:
    original = base.get("symmetry") or {}
    extra = base.get("extra") or {}
    symmetry = native["symmetry"]
    if not isinstance(symmetry, dict):
        raise ValueError("Scaffold symmetry must be an object")
    match = re.fullmatch(r"([CD])([2-9]|[1-9]\d+)", str(original.get("id", "")))
    if match is None or symmetry.get("id") != original.get("id"):
        raise ValueError(
            "Complete-scaffold partial diffusion supports matching Cn/Dn (n>=2) only"
        )
    if extra.get("symmetry_action_kind") != "regular_full_group":
        raise ValueError(
            "Complete-scaffold partial diffusion does not support quotient or mixed actions"
        )
    if original.get("declared_preexpanded_chain_layout") or extra.get(
        "preexpanded_chain_layout"
    ):
        raise ValueError(
            "Scaffold handoff currently requires a canonical ASU compiler payload"
        )
    allowed = {
        "id",
        "is_symmetric_motif",
        "use_declared_frames",
        "declared_transform_order",
        "declared_transform_matrices",
        "declared_preexpanded_chain_layout",
    }
    if set(symmetry) - allowed or symmetry.get("use_declared_frames") is not True:
        raise ValueError(
            "Scaffold symmetry must use explicit full-group preexpanded frames"
        )
    if symmetry.get("is_symmetric_motif") is not True:
        raise ValueError("Scaffold input requires exact symmetric motifs")
    order = extra.get("registry_transform_order")
    matrices = extra.get("registry_transform_matrices")
    if not isinstance(order, list) or not isinstance(matrices, dict):
        raise ValueError("Compiled input lacks a declared symmetry registry")
    multiplicity = int(match[2]) * (2 if match[1] == "D" else 1)
    if (
        len(order) != multiplicity
        or len(set(order)) != multiplicity
        or set(matrices) != set(order)
    ):
        raise ValueError("Scaffold input requires the full Cn/Dn action")
    if (
        symmetry.get("declared_transform_order") != order
        or symmetry.get("declared_transform_matrices") != matrices
    ):
        raise ValueError(
            "Scaffold artifact changes the compiler symmetry matrices/order"
        )
    for matrix in matrices.values():
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("Invalid symmetry matrix")
        if (
            not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(
                value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-8, rtol=0
            )
            or not math.isclose(np.linalg.det(value[:3, :3]), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("Symmetry matrices must be proper rigid transforms")
    chains = list(dict.fromkeys(residue["chain"] for residue in residues.values()))
    layout = symmetry.get("declared_preexpanded_chain_layout")
    if not isinstance(layout, list) or len(layout) != len(chains):
        raise ValueError(
            "Preexpanded chain layout does not cover complete scaffold chains"
        )
    entities: dict[int, list[dict]] = {}
    for record in layout:
        required = {"transform_index", "entity_id", "is_asu"}
        if (
            not isinstance(record, dict)
            or not required <= set(record)
            or set(record) - required - {"transform_id", "orbit_id"}
        ):
            raise ValueError(
                "Each preexpanded chain needs transform_index, entity_id and is_asu"
            )
        transform, entity = record["transform_index"], record["entity_id"]
        if (
            type(transform) is not int
            or not 0 <= transform < multiplicity
            or type(entity) is not int
            or entity < 0
            or type(record["is_asu"]) is not bool
        ):
            raise ValueError("Invalid preexpanded chain identity")
        if "transform_id" in record and record["transform_id"] != order[transform]:
            raise ValueError("Preexpanded transform_id disagrees with transform_index")
        entities.setdefault(entity, []).append(record)
    for records in entities.values():
        if {record["transform_index"] for record in records} != set(
            range(multiplicity)
        ) or len(records) != multiplicity:
            raise ValueError(
                "Each scaffold entity must contain exactly one complete symmetry orbit"
            )
        if sum(record["is_asu"] for record in records) != 1:
            raise ValueError("Each scaffold entity must declare exactly one ASU")
    # Declared frames are not proof that the supplied complete scaffold has
    # those frames. Check every atom slot, including generated backbone atoms.
    # Otherwise the first symmetry projection silently replaces the template.
    for entity in entities:
        members = [
            (chain, record)
            for chain, record in zip(chains, layout, strict=True)
            if record["entity_id"] == entity
        ]
        reference_chain, reference_record = next(
            (chain, record) for chain, record in members if record["is_asu"]
        )
        reference_residues = [
            residue
            for residue in residues.values()
            if residue["chain"] == reference_chain
        ]
        reference_keys = [
            (i, name)
            for i, residue in enumerate(reference_residues)
            for name in sorted(residue["atoms"])
        ]
        reference_xyz = np.asarray(
            [reference_residues[i]["atoms"][name] for i, name in reference_keys]
        )
        reference_frame = np.asarray(
            matrices[order[reference_record["transform_index"]]], dtype=float
        )
        for chain, record in members:
            current = [
                residue for residue in residues.values() if residue["chain"] == chain
            ]
            keys = [
                (i, name)
                for i, residue in enumerate(current)
                for name in sorted(residue["atoms"])
            ]
            if keys != reference_keys:
                raise ValueError(
                    "Complete scaffold symmetry copies have different residue/atom slots"
                )
            relative = np.asarray(
                matrices[order[record["transform_index"]]], dtype=float
            ) @ np.linalg.inv(reference_frame)
            expected = reference_xyz @ relative[:3, :3].T + relative[:3, 3]
            actual = np.asarray([current[i]["atoms"][name] for i, name in keys])
            if (
                float(np.linalg.norm(expected - actual, axis=1).max())
                > _SEED_MAXIMUM_ERROR
            ):
                raise ValueError(
                    "Complete scaffold coordinates violate the declared symmetry frames"
                )
    return order, matrices, layout


def _validate_group_remapping(
    base: dict,
    native: dict,
    source: dict,
    target: dict,
    source_fixed: dict,
    target_fixed: dict,
    order: list,
    matrices: dict,
    layout: list,
    component_map: dict,
) -> list[dict]:
    original = (base.get("extra") or {}).get("motif_constraint_groups")
    remapped = native["extra"].get("motif_constraint_groups")
    if not isinstance(original, list) or not original or not isinstance(remapped, list):
        raise ValueError("Scaffold handoff requires complete motif constraint groups")
    old_by_id = {group["group_id"]: group for group in original}
    new_by_id = {group["group_id"]: group for group in remapped}
    if (
        len(old_by_id) != len(original)
        or len(new_by_id) != len(remapped)
        or set(old_by_id) != set(new_by_id)
    ):
        raise ValueError("Scaffold handoff changes motif group identities")
    seen_target_atoms = set()
    reports = []
    source_unfixed = (
        set(_components(base["select_unfixed_sequence"]))
        if base.get("select_unfixed_sequence")
        else set()
    )
    target_unfixed = set(_components(native["select_unfixed_sequence"]))
    chains = list(dict.fromkeys(residue["chain"] for residue in target.values()))
    chain_transforms = {
        chain: record["transform_index"]
        for chain, record in zip(chains, layout, strict=True)
    }
    asu_transforms = {
        record["entity_id"]: record["transform_index"]
        for record in layout
        if record["is_asu"]
    }
    chain_records = dict(zip(chains, layout, strict=True))
    for group_id, old in old_by_id.items():
        new = new_by_id[group_id]
        if {k: v for k, v in old.items() if k != "members"} != {
            k: v for k, v in new.items() if k != "members"
        }:
            raise ValueError(
                f"Scaffold handoff changes motif group contract {group_id}"
            )
        if len(old["members"]) != len(new.get("members", [])):
            raise ValueError("Scaffold motif member count changed")
        original_xyz, target_xyz = [], []
        for old_member, new_member in zip(old["members"], new["members"], strict=True):
            if {
                k: v for k, v in old_member.items() if k not in _MEMBER_REMAP_FIELDS
            } != {k: v for k, v in new_member.items() if k not in _MEMBER_REMAP_FIELDS}:
                raise ValueError("Scaffold motif member identity changed")
            old_components, new_components = (
                old_member["src_components"],
                new_member["src_components"],
            )
            if (
                not old_components
                or len(old_components) != len(new_components)
                or len(set(new_components)) != len(new_components)
            ):
                raise ValueError("Scaffold fixed fragment length changed")
            transform_id = old_member["sym_transform_id"]
            new_transform = new_member.get("sym_transform_id")
            if type(transform_id) is not int or not 0 <= transform_id < len(order):
                raise ValueError(
                    "Compiled motif member transform is outside the registry"
                )
            if type(new_transform) is not int or new_transform != transform_id:
                raise ValueError(
                    "Scaffold motif member transform is outside the registry"
                )
            correspondence = new_member.get("correspondence_components")
            if (
                not isinstance(correspondence, list)
                or len(correspondence) != len(old_components)
                or len(set(correspondence)) != len(correspondence)
            ):
                raise ValueError(
                    "Scaffold motif correspondence must preserve residue count and identity"
                )
            expected_components = [
                component_map.get((value, transform_id)) for value in old_components
            ]
            if new_components != expected_components:
                raise ValueError(
                    "Scaffold motif components do not follow the compiled contig/copy mapping"
                )
            expected_correspondence = []
            for old_component, new_component in zip(
                old_components, new_components, strict=True
            ):
                record = chain_records[target[new_component]["chain"]]
                expected_correspondence.append(
                    component_map[(old_component, asu_transforms[record["entity_id"]])]
                )
            if correspondence != expected_correspondence:
                raise ValueError(
                    "Scaffold motif correspondence does not follow the canonical ASU mapping"
                )
            matrix = np.asarray(matrices[order[transform_id]], dtype=float)
            for old_component, new_component in zip(
                old_components, new_components, strict=True
            ):
                if (
                    old_component not in source_fixed
                    or new_component not in target_fixed
                ):
                    raise ValueError(
                        "Scaffold group refers to an unfixed or missing residue"
                    )
                if chain_transforms[target[new_component]["chain"]] != new_transform:
                    raise ValueError(
                        "Scaffold motif selector disagrees with its chain transform"
                    )
                atom_names = source_fixed[old_component]
                if atom_names != target_fixed[new_component]:
                    raise ValueError(
                        "Scaffold handoff changes fixed atom identity/count"
                    )
                if (old_component in source_unfixed) != (
                    new_component in target_unfixed
                ):
                    raise ValueError(
                        "Scaffold handoff changes the seed sequence conditioning policy"
                    )
                if (
                    old_component not in source_unfixed
                    and source[old_component]["name"] != target[new_component]["name"]
                ):
                    raise ValueError(
                        "Scaffold template changes a sequence-fixed seed residue"
                    )
                for atom_name in sorted(atom_names):
                    original_xyz.append(
                        source[old_component]["atoms"][atom_name] @ matrix[:3, :3].T
                        + matrix[:3, 3]
                    )
                    target_xyz.append(target[new_component]["atoms"][atom_name])
                    seen_target_atoms.add((new_component, atom_name))
        left, right = np.asarray(original_xyz), np.asarray(target_xyz)
        error = np.linalg.norm(left - right, axis=1)
        rmsd, maximum = float(np.sqrt(np.mean(error**2))), float(error.max())
        if rmsd > _SEED_RMSD_TOLERANCE or maximum > _SEED_MAXIMUM_ERROR:
            raise ValueError(
                f"Complete scaffold does not preserve compiled seed pose: {group_id}, RMSD={rmsd:.6f}, max={maximum:.6f} A"
            )
        reports.append(
            {
                "group_id": group_id,
                "fixed_atom_count": len(error),
                "rmsd_angstrom": rmsd,
                "maximum_error_angstrom": maximum,
            }
        )
    expected = {
        (component, name) for component, names in target_fixed.items() for name in names
    }
    if seen_target_atoms != expected:
        raise ValueError("Scaffold fixed atoms are not exactly covered by motif groups")
    return reports


def _validate_orbit_remapping(
    base: dict, native: dict, component_map: dict, layout: list, residues: dict
) -> None:
    original = base["extra"]["motif_constraint_orbits"]
    remapped = native["extra"].get("motif_constraint_orbits")
    if not isinstance(remapped, list) or len(remapped) != len(original):
        raise ValueError("Scaffold artifact changes motif orbit/mobility contracts")
    chains = list(dict.fromkeys(residue["chain"] for residue in residues.values()))
    asu_chains = {
        chain for chain, record in zip(chains, layout, strict=True) if record["is_asu"]
    }
    canonical = {
        source: target
        for (source, _), target in component_map.items()
        if residues[target]["chain"] in asu_chains
    }
    for before, after in zip(original, remapped, strict=True):
        if (
            not isinstance(after, dict)
            or set(before) != set(after)
            or {
                key: value
                for key, value in before.items()
                if key != "source_components"
            }
            != {
                key: value for key, value in after.items() if key != "source_components"
            }
        ):
            raise ValueError("Scaffold artifact changes motif orbit/mobility contracts")
        if "source_components" in before:
            expected = [
                canonical.get(component) for component in before["source_components"]
            ]
            if None in expected or after["source_components"] != expected:
                raise ValueError(
                    "Scaffold orbit source_components do not follow canonical ASU remapping"
                )


def remap_scaffold_interface_relations(
    relations: list, original_groups: list, remapped_groups: list
) -> list[dict]:
    """Bind each edge to its actual physical group members after expansion.

    Source-copy indices describe the compiler graph; member transform indices
    describe native chains relative to the selected ASU. They need not agree.
    Only unambiguous edge/group/role correspondence is supported here.
    """
    old_by_id = {group["group_id"]: group for group in original_groups}
    new_by_id = {group["group_id"]: group for group in remapped_groups}
    result = copy.deepcopy(relations)
    for relation in result:
        identity = relation.get("edge_instance_id")
        old_group, new_group = old_by_id.get(identity), new_by_id.get(identity)
        if old_group is None or new_group is None:
            raise ValueError(
                f"Complete-scaffold relation {identity!r} requires an unambiguous matching motif interface group"
            )
        for role in ("left", "right"):
            selector_key = role + "_source_components"
            before = [
                member for member in old_group["members"] if member["role"] == role
            ]
            after = [
                member for member in new_group["members"] if member["role"] == role
            ]
            selectors = relation.get(selector_key)
            if (
                not before
                or len(before) != len(after)
                or not isinstance(selectors, list)
                or len(selectors) != len(before)
            ):
                raise ValueError(
                    f"Complete-scaffold relation {identity!r} has ambiguous {role} fragment correspondence"
                )
            transforms = {member["sym_transform_id"] for member in after}
            if len(transforms) != 1:
                raise ValueError(
                    f"Complete-scaffold relation {identity!r} {role} spans multiple native transforms"
                )
            for selector, old, new in zip(selectors, before, after, strict=True):
                if (
                    len(_components(selector)) != len(old["src_components"])
                    or old["source_fragment_id"] != new["source_fragment_id"]
                ):
                    raise ValueError(
                        f"Complete-scaffold relation {identity!r} changes a {role} fragment"
                    )
            relation[selector_key] = [
                ",".join(member["src_components"]) for member in after
            ]
            relation[role + "_transform_index"] = next(iter(transforms))
    return result


def _validate_metadata_remapping(
    base: dict, native: dict, component_map: dict, layout: list, residues: dict
) -> None:
    """Allow selector movement, never changes to graph or acceptance policy."""
    original, remapped = base.get("extra") or {}, native["extra"]
    if (
        remapped.get("generated_coordinate_initialization")
        != "complete_scaffold_partial_diffusion"
    ):
        raise ValueError(
            "Scaffold initialization metadata must declare complete_scaffold_partial_diffusion"
        )
    if remapped.get("asu_terminal_extensions") != original.get(
        "asu_terminal_extensions", []
    ):
        raise ValueError(
            "Scaffold artifact changes compiled terminal extension lengths/identities"
        )
    expected_relations = remap_scaffold_interface_relations(
        original.get("assembly_interface_relations", []),
        original["motif_constraint_groups"],
        remapped["motif_constraint_groups"],
    )
    if remapped.get("assembly_interface_relations") != expected_relations:
        raise ValueError(
            "Scaffold artifact changes compiled assembly_interface_relations graph/geometry policy or physical mapping"
        )
    chains = list(dict.fromkeys(residue["chain"] for residue in residues.values()))
    asu_chains = {
        chain for chain, record in zip(chains, layout, strict=True) if record["is_asu"]
    }
    canonical = {
        source: target
        for (source, _), target in component_map.items()
        if residues[target]["chain"] in asu_chains
    }

    def selector(value):
        components = _components(value)
        if any(component not in canonical for component in components):
            raise ValueError(
                "Compiled scaffold segment selector lies outside the canonical ASU"
            )
        return ",".join(canonical[component] for component in components)

    expected_segments = copy.deepcopy(original.get("asu_scaffold_segments", []))
    for segment in expected_segments:
        for field in ("from_selector", "to_selector"):
            if segment.get(field):
                segment[field] = selector(segment[field])
        if segment.get("path_selectors"):
            segment["path_selectors"] = [
                selector(value) for value in segment["path_selectors"]
            ]
        if "contig_chains" in segment:
            segment["contig_chains"] = [
                ",".join(
                    selector(token) if re.match(r"[A-Za-z]", token) else token
                    for token in value.split(",")
                )
                for value in segment["contig_chains"]
            ]
    if remapped.get("asu_scaffold_segments") != expected_segments:
        raise ValueError(
            "Scaffold artifact changes compiled segment graph/length policy or canonical selector mapping"
        )


def apply_scaffold_input(
    payload: dict[str, Any], *, artifact_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Return a private native payload after validating/installing the artifact."""

    from rfd3_mosaic.validation.scaffold_contract import (
        audit_scaffold_contract,
        validate_scaffold_contract,
    )

    artifact_path = Path(artifact_path).resolve()
    output_directory = Path(output_directory).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "compiled_contract_sha256",
        "structure_path",
        "structure_sha256",
        "partial_t",
        "native_input",
    }
    if (
        not isinstance(artifact, dict)
        or set(artifact) != fields
        or type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
    ):
        raise ValueError("Unsupported or incomplete scaffold input artifact schema")
    if artifact["compiled_contract_sha256"] != compiled_scaffold_contract_sha256(
        payload
    ):
        raise ValueError("Scaffold artifact compiler contract SHA256 mismatch")
    noise = artifact["partial_t"]
    if (
        isinstance(noise, bool)
        or not isinstance(noise, (int, float))
        or not math.isfinite(noise)
        or noise <= 0
    ):
        raise ValueError(
            "Scaffold partial_t must be a positive finite noise level in Angstroms"
        )
    native = artifact["native_input"]
    if not isinstance(native, dict) or set(native) != _NATIVE_FIELDS:
        raise ValueError(
            "Scaffold native_input contains unsupported or missing override fields"
        )
    remapped_extra = native["extra"]
    if not isinstance(remapped_extra, dict) or set(remapped_extra) - _EXTRA_FIELDS:
        raise ValueError("Scaffold artifact contains unsupported extra overrides")
    base = _example(payload)
    extra = base.get("extra") or {}
    if base.get("dialect", 2) != 2 or base.get("partial_t") is not None:
        raise ValueError(
            "Scaffold handoff requires an unmodified dialect-2 compiler payload"
        )
    if (
        base.get("ligand")
        or base.get("unindex")
        or extra.get("cylindrical_constraints")
    ):
        raise ValueError(
            "Scaffold partial diffusion does not yet support ligands, unindexed motifs or cylindrical constraints"
        )
    unsupported_selectors = [
        key
        for key, value in base.items()
        if key.startswith("select_")
        and key not in {"select_fixed_atoms", "select_unfixed_sequence"}
        and value not in (None, False, "", {}, [])
    ]
    if unsupported_selectors or base.get("redesign_motif_sidechains") not in (
        None,
        False,
    ):
        raise ValueError(
            "Scaffold partial diffusion does not yet remap auxiliary conditioning selectors "
            "or motif-sidechain redesign; remove these from the scaffold task: "
            + ", ".join(unsupported_selectors or ["redesign_motif_sidechains"])
        )
    if base.get("cif_parser_args"):
        raise ValueError(
            "Scaffold partial diffusion requires the standard native structure parser"
        )
    if base.get("infer_ori_strategy") is not None or base.get("ori_token") not in (
        None,
        [0, 0, 0],
    ):
        raise ValueError(
            "Complete-scaffold input requires the compiler group origin without recentering"
        )
    original_orbits = extra.get("motif_constraint_orbits")
    if (
        not isinstance(original_orbits, list)
        or not original_orbits
        or any(orbit.get("mobility_mode") not in {"fixed", "orbit_rigid"} for orbit in original_orbits)
    ):
        raise ValueError(
            "Complete-scaffold partial diffusion requires fixed or bounded rigid seed orbits"
        )
    structure_path = Path(artifact["structure_path"])
    if not structure_path.is_absolute():
        structure_path = artifact_path.parent / structure_path
    structure_path = structure_path.resolve()
    if (
        not structure_path.is_file()
        or _sha256(structure_path) != artifact["structure_sha256"]
    ):
        raise ValueError("Scaffold structure SHA256 mismatch or file missing")
    residues = _residues(structure_path)
    if _components(native["contig"], chain_breaks=True) != list(residues):
        raise ValueError(
            "Complete-scaffold contig must cover every input residue in structure order"
        )
    if any(
        not {"N", "CA", "C", "O"} <= set(residue["atoms"])
        for residue in residues.values()
    ):
        raise ValueError("Complete scaffold requires N/CA/C/O for every residue")
    fixed = _fixed_selection(native["select_fixed_atoms"], residues)
    unfixed_sequence = set(_components(native["select_unfixed_sequence"]))
    if (
        not unfixed_sequence <= set(residues)
        or not (set(residues) - set(fixed)) <= unfixed_sequence
    ):
        raise ValueError(
            "Generated scaffold residues must explicitly have unfixed sequence"
        )
    order, matrices, layout = _validate_symmetry(base, native, residues)
    if remapped_extra.get("preexpanded_chain_layout") != layout:
        raise ValueError(
            "Scaffold extra and symmetry preexpanded chain layouts disagree"
        )
    chain_count = len(layout)
    if any(
        remapped_extra.get(key) != chain_count
        for key in ("asu_chain_count", "asu_polymer_chain_count")
    ):
        raise ValueError(
            "Complete-scaffold parser chain counts must match the complete input"
        )
    source_path = Path(base["input"])
    if not source_path.is_absolute():
        source_path = output_directory / source_path
    if _sha256(source_path) != extra.get("adapter_structure_sha256"):
        raise ValueError("Compiled fixed-seed structure SHA256 mismatch")
    source = _residues(source_path)
    source_fixed = _fixed_selection(base.get("select_fixed_atoms"), source)
    _validate_chain_patterns(base, residues, fixed, source, source_fixed, layout)
    component_map = compiled_scaffold_residue_map(
        payload,
        chain_layout=layout,
        chain_ids=list(
            dict.fromkeys(residue["chain"] for residue in residues.values())
        ),
    )
    _validate_orbit_remapping(base, native, component_map, layout, residues)
    seed_reports = _validate_group_remapping(
        base,
        native,
        source,
        residues,
        source_fixed,
        fixed,
        order,
        matrices,
        layout,
        component_map,
    )
    _validate_metadata_remapping(base, native, component_map, layout, residues)
    contract = remapped_extra.get("mosaic_scaffold_contract")
    validate_scaffold_contract(contract)
    mobile = any(o["mobility_mode"] == "orbit_rigid" for o in original_orbits)
    transport = remapped_extra.get("mosaic_reference_transport")
    if mobile or transport is not None:
        if transport != scaffold_transport_plan(native, extra, residues):
            raise ValueError("Scaffold mobility transport plan differs from the exact compiler seed ownership")
    ca_residues = list(residues.values())
    contract_audit = audit_scaffold_contract(
        contract=contract,
        coordinates=np.asarray([residue["atoms"]["CA"] for residue in ca_residues]),
        chain_ids=[residue["chain"] for residue in ca_residues],
        residue_numbers=[residue["number"] for residue in ca_residues],
        fixed_mask=np.asarray([component in fixed for component in residues]),
        align_fixed=False,
        backbone_coordinates=(np.asarray([[r["atoms"][a] for a in ("N", "CA", "C", "O")] for r in ca_residues])
                              if contract.get("schema_version") == 2 else None),
        residue_names=[r["name"] for r in ca_residues],
    )
    if not contract_audit.get("passed"):
        raise ValueError("Complete scaffold fails its declared fold/geometry contract")

    installed = output_directory / "scaffold_input"
    installed.mkdir(parents=True, exist_ok=True)
    suffix = "".join(structure_path.suffixes)
    installed_structure = installed / ("complete_scaffold" + suffix)
    structure_bytes = structure_path.read_bytes()
    if hashlib.sha256(structure_bytes).hexdigest() != artifact["structure_sha256"]:
        raise ValueError("Scaffold structure changed during validation")
    if (
        installed_structure.exists()
        and installed_structure.read_bytes() != structure_bytes
    ):
        raise ValueError(
            "Refusing to replace a different frozen scaffold in the same task"
        )
    installed_artifact = installed / "artifact.json"
    frozen_artifact = copy.deepcopy(artifact)
    frozen_artifact["structure_path"] = installed_structure.name
    artifact_bytes = _canonical_bytes(frozen_artifact) + b"\n"
    if (
        installed_artifact.exists()
        and installed_artifact.read_bytes() != artifact_bytes
    ):
        raise ValueError(
            "Refusing to replace a different frozen scaffold artifact in the same task"
        )
    installed_structure.write_bytes(structure_bytes)
    installed_artifact.write_bytes(artifact_bytes)

    result = copy.deepcopy(payload)
    example = _example(result)
    example.update(
        {key: copy.deepcopy(native[key]) for key in _NATIVE_FIELDS if key != "extra"}
    )
    example["extra"].update(copy.deepcopy(remapped_extra))
    example["input"] = str(installed_structure.relative_to(output_directory))
    example["partial_t"] = float(noise)
    example["ori_token"] = [0.0, 0.0, 0.0]
    example.pop("length", None)
    example["extra"]["adapter_structure_sha256"] = artifact["structure_sha256"]
    example["extra"]["scaffold_input"] = {
        "schema_version": 1,
        "mode": "complete_scaffold_partial_diffusion",
        "compiled_contract_sha256": artifact["compiled_contract_sha256"],
        "structure_sha256": artifact["structure_sha256"],
        "artifact_path": str(installed_artifact.relative_to(output_directory)),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "partial_t_angstrom": float(noise),
        "seed_atom_checks": seed_reports,
        "fixed_atom_maximum_error_tolerance_angstrom": _SEED_MAXIMUM_ERROR,
        "fixed_atom_rmsd_tolerance_angstrom": _SEED_RMSD_TOLERANCE,
        "input_contract_audit": contract_audit,
    }
    return result


def audit_prepared_scaffold_input(example: dict, *, input_directory: Path) -> dict:
    """Recheck the actual frozen scaffold before replacing the chord pose gate."""
    from rfd3_mosaic.validation.scaffold_contract import audit_scaffold_contract

    extra = example.get("extra") or {}
    binding = extra.get("scaffold_input") or {}
    if (
        binding.get("mode") != "complete_scaffold_partial_diffusion"
        or example.get("partial_t") is None
    ):
        raise ValueError(
            "A scaffold contract needs a validated complete partial-diffusion artifact"
        )
    if extra.get("mosaic_scaffold_contract") is None:
        raise ValueError("Frozen scaffold input lost its mandatory geometry contract")
    artifact_path = Path(binding.get("artifact_path", ""))
    if not artifact_path.is_absolute():
        artifact_path = input_directory / artifact_path
    if not artifact_path.is_file() or _sha256(artifact_path) != binding.get(
        "artifact_sha256"
    ):
        raise ValueError("Frozen scaffold artifact changed before pose preflight")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        artifact.get("compiled_contract_sha256")
        != binding.get("compiled_contract_sha256")
        or artifact.get("partial_t") != example["partial_t"]
    ):
        raise ValueError("Scaffold runtime and frozen artifact identity disagree")
    for key, value in artifact["native_input"].items():
        matches = (
            all(extra.get(k) == v for k, v in value.items())
            if key == "extra"
            else example.get(key) == value
        )
        if not matches:
            raise ValueError(
                f"Scaffold runtime overrides differ from the frozen artifact: {key}"
            )
    path = Path(example["input"])
    if not path.is_absolute():
        path = input_directory / path
    if _sha256(path) != binding.get("structure_sha256") or binding.get(
        "structure_sha256"
    ) != artifact.get("structure_sha256"):
        raise ValueError("Frozen complete scaffold changed before pose preflight")
    residues = _residues(path)
    fixed = _fixed_selection(example["select_fixed_atoms"], residues)
    report = audit_scaffold_contract(
        contract=extra["mosaic_scaffold_contract"],
        coordinates=np.asarray([r["atoms"]["CA"] for r in residues.values()]),
        chain_ids=[r["chain"] for r in residues.values()],
        residue_numbers=[r["number"] for r in residues.values()],
        fixed_mask=np.asarray([component in fixed for component in residues]),
        backbone_coordinates=(np.asarray([[r["atoms"][a] for a in ("N", "CA", "C", "O")] for r in residues.values()])
                              if extra["mosaic_scaffold_contract"].get("schema_version") == 2 else None),
        residue_names=[r["name"] for r in residues.values()],
    )
    if not report["passed"]:
        raise ValueError(
            "Frozen complete scaffold fails pre-diffusion geometry contract"
        )
    return {
        "evaluated": True,
        "passed": True,
        "measurement": "complete_scaffold_input",
        "scaffold_contract": report,
    }


__all__ = [
    "apply_scaffold_input",
    "compiled_scaffold_contract_sha256",
    "compiled_scaffold_residue_map",
    "remap_scaffold_interface_relations",
    "audit_prepared_scaffold_input",
]
