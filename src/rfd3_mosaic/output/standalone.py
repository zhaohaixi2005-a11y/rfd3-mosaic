"""RFD3-independent structure, mapping, and manifest output."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from rfd3_mosaic.compile import (
    build_master_group_transforms,
    expand_symmetry_instances,
    load_interface_seed_config,
    resolve_reference_port_frames,
)
from rfd3_mosaic.geometry import (
    apply_transform,
    compose_transforms,
    invert_transform,
)
from rfd3_mosaic.objectives import (
    build_static_metric_map,
    evaluate_objectives,
)
from rfd3_mosaic.provenance import build_mapping_registry
from rfd3_mosaic.structure import AtomRecord, load_selected_atoms
from rfd3_mosaic.topology import compile_scaffold_graph


@dataclass(frozen=True)
class CompilationOutputs:
    structure_path: Path
    mapping_path: Path
    manifest_path: Path
    atom_count: int
    residue_count: int
    chain_count: int


@dataclass(frozen=True)
class _CompiledAtom:
    atom_index: int
    chain_id: str
    entity_id: str
    label_seq_id: int
    coordinate: tuple[float, float, float]
    source_atom: AtomRecord
    fragment_instance_id: str
    source_fragment_id: str
    motion_group_instance_id: str
    orbit_id: str | None
    copy_index: int
    transform_id: str


def _classify_symmetry_pair(
    left_orbit_id: str | None,
    left_transform_id: str,
    right_orbit_id: str | None,
    right_transform_id: str,
) -> str:
    """Classify a copy pair without assuming that Dn is one linear ring."""

    if left_orbit_id is None or right_orbit_id is None:
        return "unsymmetrized"
    if left_orbit_id != right_orbit_id:
        return "cross_orbit"

    left_group, left_element = left_transform_id.split(":", maxsplit=1)
    right_group, right_element = right_transform_id.split(":", maxsplit=1)
    if left_group != right_group:
        return "cross_transform_set"
    if left_group.startswith("C"):
        return "cyclic_intra"
    if left_group.startswith("D"):
        left_is_flipped = left_element.startswith("s")
        right_is_flipped = right_element.startswith("s")
        if left_is_flipped == right_is_flipped:
            return "dihedral_intra_coset"
        return "dihedral_inter_coset"
    return "other_symmetry"


def _analyze_inter_group_clashes(
    atoms: list[_CompiledAtom],
    *,
    hard_cutoff: float = 2.0,
) -> dict[str, Any]:
    """Measure severe atom overlaps between independently placed group copies."""

    grouped: dict[str, list[_CompiledAtom]] = {}
    for atom in atoms:
        grouped.setdefault(atom.motion_group_instance_id, []).append(atom)

    pair_reports: list[dict[str, Any]] = []
    category_reports: dict[str, dict[str, Any]] = {}
    total_hard_clashes = 0
    minimum_distance: float | None = None
    group_ids = tuple(grouped)
    for left_index, left_group_id in enumerate(group_ids):
        left_coordinates = np.asarray(
            [atom.coordinate for atom in grouped[left_group_id]],
            dtype=np.float64,
        )
        for right_group_id in group_ids[left_index + 1 :]:
            right_coordinates = np.asarray(
                [atom.coordinate for atom in grouped[right_group_id]],
                dtype=np.float64,
            )
            distances = np.linalg.norm(
                left_coordinates[:, None, :]
                - right_coordinates[None, :, :],
                axis=-1,
            )
            pair_minimum = float(distances.min())
            hard_clashes = int((distances < hard_cutoff).sum())
            left_representative = grouped[left_group_id][0]
            right_representative = grouped[right_group_id][0]
            pair_class = _classify_symmetry_pair(
                left_representative.orbit_id,
                left_representative.transform_id,
                right_representative.orbit_id,
                right_representative.transform_id,
            )
            total_hard_clashes += hard_clashes
            minimum_distance = (
                pair_minimum
                if minimum_distance is None
                else min(minimum_distance, pair_minimum)
            )
            pair_reports.append(
                {
                    "left_group_instance_id": left_group_id,
                    "right_group_instance_id": right_group_id,
                    "minimum_atom_distance": pair_minimum,
                    "hard_clash_count": hard_clashes,
                    "symmetry_pair_class": pair_class,
                }
            )
            category = category_reports.setdefault(
                pair_class,
                {
                    "group_pair_count": 0,
                    "total_hard_clashes": 0,
                    "minimum_atom_distance": None,
                },
            )
            category["group_pair_count"] += 1
            category["total_hard_clashes"] += hard_clashes
            category_minimum = category["minimum_atom_distance"]
            category["minimum_atom_distance"] = (
                pair_minimum
                if category_minimum is None
                else min(category_minimum, pair_minimum)
            )

    return {
        "hard_cutoff": hard_cutoff,
        "total_hard_clashes": total_hard_clashes,
        "minimum_inter_group_distance": minimum_distance,
        "categories": category_reports,
        "group_pairs": pair_reports,
    }


def _terminal_anchor(
    atoms: list[_CompiledAtom],
    *,
    terminus: str,
) -> tuple[np.ndarray, str, int]:
    if not atoms:
        raise ValueError("Cannot resolve a terminus from an empty fragment")
    label_seq_id = (
        min(atom.label_seq_id for atom in atoms)
        if terminus == "N"
        else max(atom.label_seq_id for atom in atoms)
    )
    residue_atoms = [
        atom for atom in atoms if atom.label_seq_id == label_seq_id
    ]
    preferred_names = (
        ("N", "CA") if terminus == "N" else ("C", "CA")
    )
    for atom_name in preferred_names:
        for atom in residue_atoms:
            if atom.source_atom.atom_name.strip().upper() == atom_name:
                return (
                    np.asarray(atom.coordinate, dtype=np.float64),
                    atom_name,
                    label_seq_id,
                )
    coordinates = np.asarray(
        [atom.coordinate for atom in residue_atoms],
        dtype=np.float64,
    )
    return coordinates.mean(axis=0), "residue_centroid", label_seq_id


def _analyze_scaffold_link_geometry(
    atoms: list[_CompiledAtom],
    instances: Any,
) -> dict[str, Any]:
    """Report endpoint spans before RFD3 generates missing scaffold atoms."""

    atoms_by_fragment: dict[str, list[_CompiledAtom]] = {}
    for atom in atoms:
        atoms_by_fragment.setdefault(atom.fragment_instance_id, []).append(atom)

    reports: list[dict[str, Any]] = []
    infeasible_links: list[str] = []
    for link in instances.scaffold_links.values():
        from_coordinate, from_anchor, from_residue = _terminal_anchor(
            atoms_by_fragment[link.from_fragment_instance_id],
            terminus=link.from_terminus.value,
        )
        to_coordinate, to_anchor, to_residue = _terminal_anchor(
            atoms_by_fragment[link.to_fragment_instance_id],
            terminus=link.to_terminus.value,
        )
        endpoint_distance = float(
            np.linalg.norm(to_coordinate - from_coordinate)
        )
        minimum_required_residues = max(
            0,
            int(np.ceil(endpoint_distance / 3.8)) - 1,
        )
        within_maximum_contour = (
            minimum_required_residues <= link.maximum_length
        )
        if not within_maximum_contour and not link.chain_break:
            infeasible_links.append(link.id)
        reports.append(
            {
                "link_instance_id": link.id,
                "source_link_id": link.source_id,
                "from_fragment_instance_id": (
                    link.from_fragment_instance_id
                ),
                "to_fragment_instance_id": link.to_fragment_instance_id,
                "from_anchor": from_anchor,
                "to_anchor": to_anchor,
                "from_label_seq_id": from_residue,
                "to_label_seq_id": to_residue,
                "endpoint_distance": endpoint_distance,
                "configured_minimum_length": link.minimum_length,
                "configured_maximum_length": link.maximum_length,
                "minimum_required_residues_at_3_8A": (
                    minimum_required_residues
                ),
                "within_maximum_contour": within_maximum_contour,
                "chain_break": link.chain_break,
            }
        )
    return {
        "all_continuous_links_within_maximum_contour": not infeasible_links,
        "infeasible_link_instances": infeasible_links,
        "links": reports,
        "note": (
            "Contour feasibility is a necessary geometric check, not a "
            "prediction that RFD3 will generate a folded linker."
        ),
    }


def _analyze_symmetry_cavities(
    atoms: list[_CompiledAtom],
    spec: Any,
) -> dict[str, Any]:
    """Measure central and axial clearance for every declared orbit."""

    reports: list[dict[str, Any]] = []
    for orbit_id, orbit in spec.symmetry.orbits.items():
        transform_set = spec.symmetry.transform_sets[orbit.transform_set]
        orbit_coordinates = np.asarray(
            [
                atom.coordinate
                for atom in atoms
                if atom.orbit_id == orbit_id
            ],
            dtype=np.float64,
        )
        if not orbit_coordinates.size:
            continue
        axis = np.asarray(transform_set.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        center = np.asarray(transform_set.center, dtype=np.float64)
        relative = orbit_coordinates - center
        axial_coordinates = relative @ axis
        radial_vectors = relative - axial_coordinates[:, None] * axis
        reports.append(
            {
                "orbit_id": orbit_id,
                "transform_set_id": orbit.transform_set,
                "symmetry_type": transform_set.type.value,
                "symmetry_order": transform_set.order,
                "copy_count": len(
                    {
                        atom.copy_index
                        for atom in atoms
                        if atom.orbit_id == orbit_id
                    }
                ),
                "central_void_radius": float(
                    np.linalg.norm(relative, axis=1).min()
                ),
                "minimum_axis_clearance": float(
                    np.linalg.norm(radial_vectors, axis=1).min()
                ),
                "minimum_axial_coordinate": float(axial_coordinates.min()),
                "maximum_axial_coordinate": float(axial_coordinates.max()),
            }
        )
    return {
        "orbits": reports,
        "note": (
            "Clearance values are geometric descriptors of motif atoms, "
            "not solvent-accessible cavity calculations."
        ),
    }


def _rotation_error_deg(observed: np.ndarray, target: np.ndarray) -> float:
    relative = target.T @ observed
    cosine = float(
        np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    )
    return float(np.rad2deg(np.arccos(cosine)))


def _analyze_interface_edges(
    atoms: list[_CompiledAtom],
    instances: Any,
    reference_frames: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate resolved port pairs against their declared target geometry."""

    atoms_by_fragment: dict[str, list[_CompiledAtom]] = {}
    for atom in atoms:
        atoms_by_fragment.setdefault(atom.fragment_instance_id, []).append(atom)

    reports: list[dict[str, Any]] = []
    failed_required_edges: list[str] = []
    for edge in instances.interfaces.values():
        left_port = instances.ports[edge.left_port_instance_id]
        right_port = instances.ports[edge.right_port_instance_id]
        left_world_frame = compose_transforms(
            left_port.transform,
            reference_frames[left_port.source_id],
        )
        right_world_frame = compose_transforms(
            right_port.transform,
            reference_frames[right_port.source_id],
        )
        observed_relative = compose_transforms(
            invert_transform(left_world_frame),
            right_world_frame,
        )
        left_coordinates = np.asarray(
            [
                atom.coordinate
                for fragment_id in left_port.fragment_instance_ids
                for atom in atoms_by_fragment[fragment_id]
            ],
            dtype=np.float64,
        )
        right_coordinates = np.asarray(
            [
                atom.coordinate
                for fragment_id in right_port.fragment_instance_ids
                for atom in atoms_by_fragment[fragment_id]
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm(
            left_coordinates[:, None, :] - right_coordinates[None, :, :],
            axis=-1,
        )
        normal_cosine = float(
            np.clip(
                np.dot(left_world_frame[:3, 2], right_world_frame[:3, 2]),
                -1.0,
                1.0,
            )
        )
        report: dict[str, Any] = {
            "edge_instance_id": edge.id,
            "source_edge_id": edge.source_id,
            "left_port_instance_id": edge.left_port_instance_id,
            "right_port_instance_id": edge.right_port_instance_id,
            "source_copy_index": edge.source_copy_index,
            "target_copy_index": edge.target_copy_index,
            "centroid_distance": float(
                np.linalg.norm(
                    left_coordinates.mean(axis=0)
                    - right_coordinates.mean(axis=0)
                )
            ),
            "minimum_atom_distance": float(distances.min()),
            "heavy_atom_contacts_below_4_5A": int((distances < 4.5).sum()),
            "hard_clashes_below_2_0A": int((distances < 2.0).sum()),
            "normal_angle_deg": float(np.rad2deg(np.arccos(normal_cosine))),
        }

        geometry = edge.target_geometry
        if geometry.mode == "reference_transform":
            if geometry.from_reference_seed:
                left_reference = np.asarray(
                    reference_frames[left_port.source_id],
                    dtype=np.float64,
                )
                right_reference = np.asarray(
                    reference_frames[right_port.source_id],
                    dtype=np.float64,
                )
                target_relative = compose_transforms(
                    invert_transform(left_reference),
                    right_reference,
                )
            else:
                target_relative = np.asarray(
                    geometry.target_transform,
                    dtype=np.float64,
                )
            translation_error = float(
                np.linalg.norm(
                    observed_relative[:3, 3] - target_relative[:3, 3]
                )
            )
            rotation_error = _rotation_error_deg(
                observed_relative[:3, :3],
                target_relative[:3, :3],
            )
            satisfied = (
                translation_error <= geometry.translation_tolerance
                and rotation_error <= geometry.rotation_tolerance_deg
                and report["hard_clashes_below_2_0A"] == 0
            )
            report.update(
                {
                    "target_mode": geometry.mode,
                    "translation_error": translation_error,
                    "translation_tolerance": geometry.translation_tolerance,
                    "rotation_error_deg": rotation_error,
                    "rotation_tolerance_deg": (
                        geometry.rotation_tolerance_deg
                    ),
                    "satisfied": satisfied,
                }
            )
        else:
            report.update(
                {
                    "target_mode": geometry.mode,
                    "satisfied": False,
                    "diagnostic": (
                        "Geometric-constraint validation is not implemented"
                    ),
                }
            )
        if edge.required and not report["satisfied"]:
            failed_required_edges.append(edge.id)
        reports.append(report)

    return {
        "all_required_satisfied": not failed_required_edges,
        "failed_required_edge_instances": failed_required_edges,
        "edges": reports,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chain_id(index: int) -> str:
    """Return A..Z, AA..AZ, BA... for a zero-based chain index."""

    if index < 0:
        raise ValueError("Chain index cannot be negative")
    value = index + 1
    characters: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        characters.append(chr(ord("A") + remainder))
    return "".join(reversed(characters))


def _element(atom: AtomRecord) -> str:
    letters = re.sub(r"[^A-Za-z]", "", atom.element)
    if letters:
        return letters[:2].upper()
    inferred = re.sub(r"[^A-Za-z]", "", atom.atom_name)
    if not inferred:
        raise ValueError(
            f"Cannot infer element for source atom serial {atom.serial}"
        )
    return inferred[0].upper()


def _cif_value(value: str) -> str:
    if not value:
        return "?"
    if re.fullmatch(r"[A-Za-z0-9_.+\-]+", value):
        return value
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _compile_atoms(
    config_path: Path,
    base_directory: Path,
    *,
    strict_validation: bool = True,
    random_seed: int | None = None,
    sample_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[_CompiledAtom], Any, Any, dict[str, Any]]:
    spec = load_interface_seed_config(config_path)
    initialization_samples: dict[str, Any] = {}
    master_transforms = build_master_group_transforms(
        spec,
        base_directory=base_directory,
        random_seed=random_seed,
        sample_metadata=initialization_samples,
        sample_overrides=sample_overrides,
    )
    instances = expand_symmetry_instances(
        spec,
        master_transforms=master_transforms,
    )
    scaffold_graph = compile_scaffold_graph(instances)
    registry = build_mapping_registry(spec, instances)
    frames = resolve_reference_port_frames(
        spec,
        base_directory=base_directory,
    )
    source_atoms = {
        fragment_id: load_selected_atoms(
            fragment_spec,
            base_directory=base_directory,
        )
        for fragment_id, fragment_spec in spec.fragments.items()
    }
    entity_ids = {
        fragment_id: str(index + 1)
        for index, fragment_id in enumerate(spec.fragments)
    }

    compiled_atoms: list[_CompiledAtom] = []
    fragment_ranges: dict[str, dict[str, Any]] = {}
    global_residue_index = 0
    for chain_index, fragment in enumerate(instances.fragments.values()):
        atoms = source_atoms[fragment.source_id]
        coordinates = np.asarray(
            [atom.coordinate for atom in atoms],
            dtype=np.float64,
        )
        transformed = apply_transform(coordinates, fragment.transform)
        chain_id = _chain_id(chain_index)
        residue_labels: dict[tuple[str, int, str], int] = {}
        fragment_atom_start = len(compiled_atoms)
        fragment_residue_indices: list[int] = []
        for atom, coordinate in zip(atoms, transformed, strict=True):
            if atom.residue_id not in residue_labels:
                residue_labels[atom.residue_id] = len(residue_labels) + 1
                fragment_residue_indices.append(global_residue_index)
                global_residue_index += 1
            compiled_atoms.append(
                _CompiledAtom(
                    atom_index=len(compiled_atoms),
                    chain_id=chain_id,
                    entity_id=entity_ids[fragment.source_id],
                    label_seq_id=residue_labels[atom.residue_id],
                    coordinate=tuple(float(value) for value in coordinate),
                    source_atom=atom,
                    fragment_instance_id=fragment.id,
                    source_fragment_id=fragment.source_id,
                    motion_group_instance_id=(
                        fragment.motion_group_instance_id
                    ),
                    orbit_id=fragment.orbit_id,
                    copy_index=fragment.copy_index,
                    transform_id=fragment.transform_id,
                )
            )
        fragment_ranges[fragment.id] = {
            "chain_id": chain_id,
            "entity_id": entity_ids[fragment.source_id],
            "compiled_atom_indices": list(
                range(fragment_atom_start, len(compiled_atoms))
            ),
            "compiled_residue_indices": fragment_residue_indices,
        }

    compilation = {
        "spec": spec,
        "instances": instances,
        "scaffold_graph": scaffold_graph,
        "registry": registry,
        "frames": frames,
        "fragment_ranges": fragment_ranges,
        "residue_count": global_residue_index,
        "master_transforms": {
            group_id: transform.tolist()
            for group_id, transform in master_transforms.items()
        },
        "initialization_samples": initialization_samples,
    }
    clash_report = _analyze_inter_group_clashes(compiled_atoms)
    compilation["clash_report"] = clash_report
    interface_report = _analyze_interface_edges(
        compiled_atoms,
        instances,
        frames,
    )
    compilation["interface_report"] = interface_report
    compilation["linker_geometry_report"] = (
        _analyze_scaffold_link_geometry(
            compiled_atoms,
            instances,
        )
    )
    compilation["symmetry_cavity_report"] = _analyze_symmetry_cavities(
        compiled_atoms,
        spec,
    )
    static_metrics = build_static_metric_map(
        clash_report=clash_report,
        interface_report=interface_report,
        linker_report=compilation["linker_geometry_report"],
        cavity_report=compilation["symmetry_cavity_report"],
    )
    compilation["static_metrics"] = static_metrics
    compilation["objective_report"] = evaluate_objectives(
        spec.objectives,
        static_metrics,
    ).to_dict()
    compilation["strict_validation"] = strict_validation

    if strict_validation and clash_report["total_hard_clashes"]:
        raise ValueError(
            "Standalone compilation rejected severe inter-group clashes: "
            f"{clash_report['total_hard_clashes']} atom pairs are closer "
            f"than {clash_report['hard_cutoff']:.2f} A; minimum distance "
            f"is {clash_report['minimum_inter_group_distance']:.3f} A"
        )
    if strict_validation and not interface_report["all_required_satisfied"]:
        raise ValueError(
            "Standalone compilation rejected unsatisfied required interface "
            f"edges: {interface_report['failed_required_edge_instances']}"
        )
    return compiled_atoms, spec, instances, compilation


def _write_cif(path: Path, atoms: list[_CompiledAtom]) -> None:
    headers = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    )
    lines = [
        "data_rfd3_mosaic",
        "#",
        "_rfd3_mosaic.schema_version 2",
        "_rfd3_mosaic.generated_linker_coordinates no",
        "#",
        "loop_",
        *headers,
    ]
    for atom in atoms:
        source = atom.source_atom
        x, y, z = atom.coordinate
        lines.append(
            " ".join(
                (
                    source.record_type,
                    str(atom.atom_index + 1),
                    _element(source),
                    _cif_value(source.atom_name),
                    ".",
                    _cif_value(source.residue_name),
                    atom.chain_id,
                    atom.entity_id,
                    str(atom.label_seq_id),
                    (
                        _cif_value(source.insertion_code)
                        if source.insertion_code
                        else "?"
                    ),
                    f"{x:.3f}",
                    f"{y:.3f}",
                    f"{z:.3f}",
                    "1.00",
                    "0.00",
                    str(source.residue_number),
                    _cif_value(source.residue_name),
                    atom.chain_id,
                    _cif_value(source.atom_name),
                    "1",
                )
            )
        )
    lines.extend(("#", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def _atom_mapping(atom: _CompiledAtom) -> dict[str, Any]:
    source = atom.source_atom
    return {
        "compiled": {
            "atom_index": atom.atom_index,
            "chain_id": atom.chain_id,
            "entity_id": atom.entity_id,
            "label_seq_id": atom.label_seq_id,
            "atom_name": source.atom_name,
        },
        "source": {
            "fragment_id": atom.source_fragment_id,
            "chain_id": source.chain_id,
            "residue_number": source.residue_number,
            "insertion_code": source.insertion_code,
            "residue_name": source.residue_name,
            "atom_name": source.atom_name,
            "atom_serial": source.serial,
        },
        "instance": {
            "fragment_instance_id": atom.fragment_instance_id,
            "motion_group_instance_id": atom.motion_group_instance_id,
            "orbit_id": atom.orbit_id,
            "copy_index": atom.copy_index,
            "transform_id": atom.transform_id,
        },
    }


def compile_standalone(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    base_directory: str | Path = ".",
    strict_validation: bool = True,
    random_seed: int | None = None,
    sample_overrides: dict[str, dict[str, Any]] | None = None,
) -> CompilationOutputs:
    """Compile a validated Interface-Seed config into standalone artifacts."""

    config = Path(config_path).resolve()
    base = Path(base_directory).resolve()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    structure_path = output / "presymmetrized_input.cif"
    mapping_path = output / "mapping.json"
    manifest_path = output / "manifest.json"

    atoms, spec, instances, compilation = _compile_atoms(
        config,
        base,
        strict_validation=strict_validation,
        random_seed=random_seed,
        sample_overrides=sample_overrides,
    )
    _write_cif(structure_path, atoms)

    mapping_payload = {
        "schema_version": 1,
        "coordinate_indexing": "zero_based",
        "object_registry": compilation["registry"].model_dump(mode="json"),
        "fragment_ranges": compilation["fragment_ranges"],
        "port_frames": compilation["frames"],
        "master_transforms": compilation["master_transforms"],
        "initialization_samples": compilation["initialization_samples"],
        "atom_mappings": [_atom_mapping(atom) for atom in atoms],
    }
    mapping_path.write_text(
        json.dumps(mapping_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    input_paths = sorted(
        {
            (
                fragment.source
                if fragment.source.is_absolute()
                else base / fragment.source
            ).resolve()
            for fragment in spec.fragments.values()
        }
    )
    manifest_payload = {
        "schema_version": 1,
        "compiler": "rfd3_mosaic.standalone",
        "config": {
            "path": str(config),
            "sha256": _sha256(config),
            "interface_seed_schema_version": spec.schema_version,
            "configured_random_seed": spec.random_seed,
            "effective_random_seed": (
                spec.random_seed if random_seed is None else random_seed
            ),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths
        ],
        "outputs": {
            "structure": {
                "path": str(structure_path.resolve()),
                "sha256": _sha256(structure_path),
            },
            "mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": _sha256(mapping_path),
            },
        },
        "initialization_samples": compilation["initialization_samples"],
        "counts": {
            "atoms": len(atoms),
            "residues": compilation["residue_count"],
            "chains": len(instances.fragments),
            "fragment_instances": len(instances.fragments),
            "motion_group_instances": len(instances.motion_groups),
            "port_instances": len(instances.ports),
            "interface_edge_instances": len(instances.interfaces),
            "scaffold_link_instances": len(instances.scaffold_links),
        },
        "interface_edges": [
            edge.model_dump(mode="json")
            for edge in instances.interfaces.values()
        ],
        "scaffold_links": [
            link.model_dump(mode="json")
            for link in instances.scaffold_links.values()
        ],
        "validation": {
            "strict_validation": compilation["strict_validation"],
            "inter_group_clashes": compilation["clash_report"],
            "interfaces": compilation["interface_report"],
            "scaffold_link_geometry": compilation[
                "linker_geometry_report"
            ],
            "symmetry_cavities": compilation[
                "symmetry_cavity_report"
            ],
            "static_metrics": compilation["static_metrics"],
            "objectives": compilation["objective_report"],
        },
        "limitations": [
            "Scaffold linker coordinates are not generated in this artifact.",
            "Compiled indices are RFD3-independent until adapter validation.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return CompilationOutputs(
        structure_path=structure_path,
        mapping_path=mapping_path,
        manifest_path=manifest_path,
        atom_count=len(atoms),
        residue_count=compilation["residue_count"],
        chain_count=len(instances.fragments),
    )
