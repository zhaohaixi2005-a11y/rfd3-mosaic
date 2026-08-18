"""RFD3-independent structure, mapping, and manifest output."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
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
from rfd3_mosaic.topology import (
    compile_scaffold_graph,
    compiled_scaffold_links,
)


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


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def _vector_angle_degrees(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    sign_invariant: bool = False,
) -> float | None:
    if left is None or right is None:
        return None
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if sign_invariant:
        cosine = abs(cosine)
    return float(np.degrees(np.arccos(cosine)))


def _terminal_backbone_geometry(
    atoms: list[_CompiledAtom],
    *,
    terminus: str,
) -> dict[str, Any]:
    anchor, anchor_name, label_seq_id = _terminal_anchor(
        atoms,
        terminus=terminus,
    )
    residue_atoms = {
        atom.source_atom.atom_name.strip().upper(): np.asarray(
            atom.coordinate, dtype=np.float64
        )
        for atom in atoms
        if atom.label_seq_id == label_seq_id
    }
    n_coordinate = residue_atoms.get("N")
    ca_coordinate = residue_atoms.get("CA")
    c_coordinate = residue_atoms.get("C")
    if terminus == "C":
        tangent = (
            _unit_vector(c_coordinate - ca_coordinate)
            if c_coordinate is not None and ca_coordinate is not None
            else None
        )
    else:
        tangent = (
            _unit_vector(ca_coordinate - n_coordinate)
            if n_coordinate is not None and ca_coordinate is not None
            else None
        )
    plane_normal = (
        _unit_vector(
            np.cross(
                ca_coordinate - n_coordinate,
                c_coordinate - ca_coordinate,
            )
        )
        if (
            n_coordinate is not None
            and ca_coordinate is not None
            and c_coordinate is not None
        )
        else None
    )
    return {
        "anchor": anchor,
        "anchor_name": anchor_name,
        "label_seq_id": label_seq_id,
        "tangent": tangent,
        "plane_normal": plane_normal,
    }


def _minimum_point_to_segment_distance(
    coordinates: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float | None:
    if not coordinates.size:
        return None
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-12:
        return float(np.linalg.norm(coordinates - start, axis=1).min())
    fractions = np.clip(
        ((coordinates - start) @ segment) / squared_length,
        0.0,
        1.0,
    )
    projections = start + fractions[:, None] * segment
    return float(np.linalg.norm(coordinates - projections, axis=1).min())


def _minimum_segment_axis_clearance(
    start: np.ndarray,
    end: np.ndarray,
    *,
    axis: np.ndarray,
    center: np.ndarray,
) -> float:
    relative_start = start - center
    segment = end - start
    radial_start = relative_start - np.dot(relative_start, axis) * axis
    radial_segment = segment - np.dot(segment, axis) * axis
    squared_radial_length = float(
        np.dot(radial_segment, radial_segment)
    )
    fraction = (
        float(
            np.clip(
                -np.dot(radial_start, radial_segment)
                / squared_radial_length,
                0.0,
                1.0,
            )
        )
        if squared_radial_length > 1e-12
        else 0.0
    )
    closest_radial = radial_start + fraction * radial_segment
    return float(np.linalg.norm(closest_radial))


def _analyze_scaffold_link_geometry(
    atoms: list[_CompiledAtom],
    instances: Any,
    spec: Any,
) -> dict[str, Any]:
    """Report unit-boundary geometry before RFD3 generates scaffold atoms."""

    atoms_by_fragment: dict[str, list[_CompiledAtom]] = {}
    for atom in atoms:
        atoms_by_fragment.setdefault(atom.fragment_instance_id, []).append(atom)

    reports: list[dict[str, Any]] = []
    infeasible_links: list[str] = []
    for link in compiled_scaffold_links(instances).values():
        from_geometry = _terminal_backbone_geometry(
            atoms_by_fragment[link.from_fragment_instance_id],
            terminus=link.from_terminus.value,
        )
        to_geometry = _terminal_backbone_geometry(
            atoms_by_fragment[link.to_fragment_instance_id],
            terminus=link.to_terminus.value,
        )
        from_coordinate = from_geometry["anchor"]
        to_coordinate = to_geometry["anchor"]
        endpoint_vector = to_coordinate - from_coordinate
        endpoint_distance = float(
            np.linalg.norm(endpoint_vector)
        )
        endpoint_direction = _unit_vector(endpoint_vector)
        minimum_required_residues = max(
            0,
            int(np.ceil(endpoint_distance / 3.8)) - 1,
        )
        within_maximum_contour = (
            minimum_required_residues <= link.maximum_length
        )
        if not within_maximum_contour and not link.chain_break:
            infeasible_links.append(link.id)
        excluded_fragments = {
            link.from_fragment_instance_id,
            link.to_fragment_instance_id,
        }
        corridor_coordinates = np.asarray(
            [
                atom.coordinate
                for atom in atoms
                if atom.fragment_instance_id not in excluded_fragments
            ],
            dtype=np.float64,
        )
        corridor_start = from_coordinate + 0.1 * endpoint_vector
        corridor_end = from_coordinate + 0.9 * endpoint_vector
        interior_chord_clearance = _minimum_point_to_segment_distance(
            corridor_coordinates,
            corridor_start,
            corridor_end,
        )
        chord_axis_clearance: float | None = None
        chord_axial_fraction: float | None = None
        chord_out_of_plane_angle: float | None = None
        if link.orbit_id is not None:
            orbit = spec.symmetry.orbits[link.orbit_id]
            transform_set = spec.symmetry.transform_sets[
                orbit.transform_set
            ]
            axis = np.asarray(transform_set.axis, dtype=np.float64)
            axis /= np.linalg.norm(axis)
            center = np.asarray(transform_set.center, dtype=np.float64)
            chord_axis_clearance = _minimum_segment_axis_clearance(
                from_coordinate,
                to_coordinate,
                axis=axis,
                center=center,
            )
            if endpoint_distance > 1e-8:
                chord_axial_fraction = float(
                    abs(np.dot(endpoint_vector, axis))
                    / endpoint_distance
                )
                chord_out_of_plane_angle = float(
                    np.degrees(
                        np.arcsin(
                            np.clip(chord_axial_fraction, 0.0, 1.0)
                        )
                    )
                )
        reports.append(
            {
                "link_instance_id": link.id,
                "source_link_id": link.source_id,
                "tie_group": link.tie_group,
                "from_fragment_instance_id": (
                    link.from_fragment_instance_id
                ),
                "to_fragment_instance_id": link.to_fragment_instance_id,
                "from_anchor": from_geometry["anchor_name"],
                "to_anchor": to_geometry["anchor_name"],
                "from_label_seq_id": from_geometry["label_seq_id"],
                "to_label_seq_id": to_geometry["label_seq_id"],
                "endpoint_distance": endpoint_distance,
                "from_terminal_tangent_to_chord_angle_deg": (
                    _vector_angle_degrees(
                        from_geometry["tangent"],
                        endpoint_direction,
                    )
                ),
                "to_terminal_tangent_to_chord_angle_deg": (
                    _vector_angle_degrees(
                        to_geometry["tangent"],
                        endpoint_direction,
                    )
                ),
                "terminal_tangent_relative_angle_deg": (
                    _vector_angle_degrees(
                        from_geometry["tangent"],
                        to_geometry["tangent"],
                    )
                ),
                "terminal_plane_normal_relative_angle_deg": (
                    _vector_angle_degrees(
                        from_geometry["plane_normal"],
                        to_geometry["plane_normal"],
                        sign_invariant=True,
                    )
                ),
                "endpoint_chord_axial_fraction": chord_axial_fraction,
                "endpoint_chord_out_of_plane_angle_deg": (
                    chord_out_of_plane_angle
                ),
                "minimum_endpoint_chord_axis_clearance": (
                    chord_axis_clearance
                ),
                "minimum_interior_chord_fixed_atom_clearance": (
                    interior_chord_clearance
                ),
                "corridor_excluded_fragment_instance_ids": sorted(
                    excluded_fragments
                ),
                "configured_minimum_length": link.minimum_length,
                "configured_maximum_length": link.maximum_length,
                "minimum_required_residues_at_3_8A": (
                    minimum_required_residues
                ),
                "within_maximum_contour": within_maximum_contour,
                "chain_break": link.chain_break,
            }
        )
    source_link_bindings: dict[str, dict[str, Any]] = {}
    for report in reports:
        if bool(report["chain_break"]):
            continue
        source_id = str(report["source_link_id"])
        entry = source_link_bindings.setdefault(
            source_id,
            {
                "source_link_id": source_id,
                "tie_group": report.get("tie_group"),
                "configured_minimum_length": int(
                    report["configured_minimum_length"]
                ),
                "configured_maximum_length": int(
                    report["configured_maximum_length"]
                ),
                "physical_instance_ids": [],
                "minimum_required_residues_by_instance": {},
            },
        )
        configured = (
            int(report["configured_minimum_length"]),
            int(report["configured_maximum_length"]),
        )
        expected = (
            entry["configured_minimum_length"],
            entry["configured_maximum_length"],
        )
        if configured != expected:
            raise ValueError(
                "Symmetry instances of scaffold link "
                f"{source_id!r} have inconsistent configured ranges"
            )
        if report.get("tie_group") != entry["tie_group"]:
            raise ValueError(
                "Symmetry instances of scaffold link "
                f"{source_id!r} have inconsistent tie groups"
            )
        instance_id = str(report["link_instance_id"])
        entry["physical_instance_ids"].append(instance_id)
        entry["minimum_required_residues_by_instance"][instance_id] = int(
            report["minimum_required_residues_at_3_8A"]
        )
    for entry in source_link_bindings.values():
        minimum = int(entry["configured_minimum_length"])
        maximum = int(entry["configured_maximum_length"])
        midpoint = (minimum + maximum) // 2
        requirements = tuple(
            int(value)
            for value in entry[
                "minimum_required_residues_by_instance"
            ].values()
        )
        required = max(requirements)
        selected = max(midpoint, required)
        entry["required_minimum_over_physical_instances"] = required
        entry["automatic_materialized_length"] = selected
        entry["automatic_materialization_feasible"] = selected <= maximum
        entry["automatic_materialization_policy"] = (
            "user_exact"
            if minimum == maximum
            else "configured_range_midpoint"
            if selected == midpoint
            else "configured_range_contour_sufficient"
        )
    tie_group_bindings: dict[str, dict[str, Any]] = {}
    for source in source_link_bindings.values():
        tie_group = source.get("tie_group")
        if tie_group is None:
            continue
        entry = tie_group_bindings.setdefault(
            str(tie_group),
            {
                "tie_group": str(tie_group),
                "source_link_ids": [],
                "common_configured_minimum_length": 0,
                "common_configured_maximum_length": 2**31 - 1,
                "required_minimum_over_all_physical_instances": 0,
            },
        )
        entry["source_link_ids"].append(source["source_link_id"])
        entry["common_configured_minimum_length"] = max(
            int(entry["common_configured_minimum_length"]),
            int(source["configured_minimum_length"]),
        )
        entry["common_configured_maximum_length"] = min(
            int(entry["common_configured_maximum_length"]),
            int(source["configured_maximum_length"]),
        )
        entry["required_minimum_over_all_physical_instances"] = max(
            int(entry["required_minimum_over_all_physical_instances"]),
            int(source["required_minimum_over_physical_instances"]),
        )
    for entry in tie_group_bindings.values():
        minimum = int(entry["common_configured_minimum_length"])
        maximum = int(entry["common_configured_maximum_length"])
        required = int(
            entry["required_minimum_over_all_physical_instances"]
        )
        midpoint = (minimum + maximum) // 2
        selected = max(midpoint, required)
        entry["automatic_materialized_length"] = selected
        entry["automatic_materialization_feasible"] = (
            minimum <= maximum and selected <= maximum
        )
        entry["source_link_ids"].sort()
        entry["is_effective_multi_link_tie"] = (
            len(entry["source_link_ids"]) > 1
        )
    infeasible_tie_groups = sorted(
        tie_group
        for tie_group, entry in tie_group_bindings.items()
        if not bool(entry["automatic_materialization_feasible"])
    )
    return {
        "all_continuous_links_within_maximum_contour": not infeasible_links,
        "all_generated_link_constraints_materializable": (
            not infeasible_links and not infeasible_tie_groups
        ),
        "infeasible_link_instances": infeasible_links,
        "infeasible_tie_groups": infeasible_tie_groups,
        "links": reports,
        "source_link_bindings": source_link_bindings,
        "tie_group_bindings": tie_group_bindings,
        "note": (
            "These are fixed-boundary and straight-chord proxy descriptors "
            "for one generated protomer segment. They are necessary CPU "
            "checks, not predictions of the folded RFD3 scaffold path."
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
        radial_distances = np.linalg.norm(radial_vectors, axis=1)
        axial_span = float(
            axial_coordinates.max() - axial_coordinates.min()
        )
        maximum_axis_extent = float(radial_distances.max())
        center_distances = np.linalg.norm(relative, axis=1)
        maximum_center_extent = float(center_distances.max())
        radial_thickness = float(
            maximum_axis_extent - radial_distances.min()
        )
        centered_coordinates = orbit_coordinates - orbit_coordinates.mean(
            axis=0
        )
        covariance = (
            centered_coordinates.T @ centered_coordinates
        ) / len(centered_coordinates)
        shape_eigenvalues = np.linalg.eigvalsh(covariance)
        largest_eigenvalue = float(shape_eigenvalues[-1])
        shape_sphericity = (
            float(
                np.sqrt(
                    max(float(shape_eigenvalues[0]), 0.0)
                    / largest_eigenvalue
                )
            )
            if largest_eigenvalue > 0.0
            else 0.0
        )
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
                    center_distances.min()
                ),
                "maximum_center_extent": maximum_center_extent,
                "outer_diameter": 2.0 * maximum_center_extent,
                "minimum_axis_clearance": float(
                    radial_distances.min()
                ),
                "mean_axis_clearance": float(radial_distances.mean()),
                "maximum_axis_extent": maximum_axis_extent,
                "radial_thickness": radial_thickness,
                "radial_thickness_fraction": (
                    radial_thickness / maximum_axis_extent
                    if maximum_axis_extent > 0.0
                    else 0.0
                ),
                "minimum_axial_coordinate": float(axial_coordinates.min()),
                "maximum_axial_coordinate": float(axial_coordinates.max()),
                "axial_span": axial_span,
                "axial_to_radial_aspect_ratio": (
                    axial_span / (2.0 * maximum_axis_extent)
                    if maximum_axis_extent > 0.0
                    else 0.0
                ),
                "shape_covariance_eigenvalues": [
                    float(value) for value in shape_eigenvalues
                ],
                "shape_sphericity": shape_sphericity,
            }
        )
    return {
        "orbits": reports,
        "note": (
            "Clearance and morphology values are geometric descriptors of "
            "motif atoms, not solvent-accessible cavity or designability "
            "calculations."
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
            "action_copy_index": edge.action_copy_index,
            "action_transform_id": edge.action_transform_id,
            "physical_edge_index": edge.physical_edge_index,
            "equivalent_action_transform_ids": list(
                edge.equivalent_action_transform_ids
            ),
            "edge_stabilizer_order": edge.edge_stabilizer_order,
            "left_orbit_id": edge.left_orbit_id,
            "right_orbit_id": edge.right_orbit_id,
            "left_transform_index": edge.left_transform_index,
            "right_transform_index": edge.right_transform_index,
            "satisfaction_stage": edge.satisfaction_stage,
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
                # ``preserve_input`` freezes the relation in the compiled
                # reference assembly, not the relation between the two raw
                # source-file port frames.  Those are equivalent only when
                # both ports address the same symmetry copy.  A named
                # cross-copy edge (for example face_a -> face_b@T:g01) must
                # include both the sampled master pose and the target group
                # action in its reference relation.
                #
                # ``left_world_frame`` and ``right_world_frame`` already
                # contain exactly those transforms because the instance set
                # was expanded with ``master_transforms``.  Freezing this
                # initialized relation also matches the post-diffusion audit,
                # whose reference is the emitted presymmetrized input.
                target_relative = observed_relative.copy()
                reference_basis = "initialized_symmetry_expanded_assembly"
            else:
                target_relative = np.asarray(
                    geometry.target_transform,
                    dtype=np.float64,
                )
                reference_basis = "declared_target_transform"
            translation_error = float(
                np.linalg.norm(
                    observed_relative[:3, 3] - target_relative[:3, 3]
                )
            )
            rotation_error = _rotation_error_deg(
                observed_relative[:3, :3],
                target_relative[:3, :3],
            )
            contact_count = int(
                (distances < geometry.contact_cutoff).sum()
            )
            contacts_satisfied = (
                contact_count >= geometry.minimum_heavy_atom_contacts
            )
            satisfied = (
                translation_error <= geometry.translation_tolerance
                and rotation_error <= geometry.rotation_tolerance_deg
                and contacts_satisfied
                and report["hard_clashes_below_2_0A"] == 0
            )
            report.update(
                {
                    "target_mode": geometry.mode,
                    "reference_basis": reference_basis,
                    "target_relative_transform": target_relative.tolist(),
                    "translation_error": translation_error,
                    "translation_tolerance": geometry.translation_tolerance,
                    "rotation_error_deg": rotation_error,
                    "rotation_tolerance_deg": (
                        geometry.rotation_tolerance_deg
                    ),
                    "declared_contact_cutoff": geometry.contact_cutoff,
                    "declared_contact_count": contact_count,
                    "minimum_heavy_atom_contacts": (
                        geometry.minimum_heavy_atom_contacts
                    ),
                    "contacts_satisfied": contacts_satisfied,
                    "satisfied": satisfied,
                }
            )
        else:
            checks: list[bool] = []
            if geometry.distance is not None:
                if geometry.distance.type != "com":
                    distance_error = None
                    distance_satisfied = False
                else:
                    distance_error = abs(
                        report["centroid_distance"]
                        - geometry.distance.target
                    )
                    distance_satisfied = (
                        distance_error <= geometry.distance.tolerance
                    )
                report.update(
                    {
                        "distance_type": geometry.distance.type,
                        "distance_target": geometry.distance.target,
                        "distance_tolerance": geometry.distance.tolerance,
                        "distance_error": distance_error,
                        "distance_satisfied": distance_satisfied,
                    }
                )
                if geometry.distance.type != "com":
                    report["distance_diagnostic"] = (
                        "Standalone geometric validation currently supports "
                        "only distance.type=com"
                    )
                checks.append(distance_satisfied)
            if geometry.normal_angle_deg is not None:
                normal_error = abs(
                    report["normal_angle_deg"]
                    - geometry.normal_angle_deg.target
                )
                normal_satisfied = (
                    normal_error <= geometry.normal_angle_deg.tolerance
                )
                report.update(
                    {
                        "normal_angle_target_deg": (
                            geometry.normal_angle_deg.target
                        ),
                        "normal_angle_tolerance_deg": (
                            geometry.normal_angle_deg.tolerance
                        ),
                        "normal_angle_error_deg": normal_error,
                        "normal_angle_satisfied": normal_satisfied,
                    }
                )
                checks.append(normal_satisfied)
            if geometry.twist_deg is not None:
                twist_deg = float(
                    np.rad2deg(
                        np.arctan2(
                            observed_relative[1, 0],
                            observed_relative[0, 0],
                        )
                    )
                )
                twist_satisfied = (
                    geometry.twist_deg.minimum
                    <= twist_deg
                    <= geometry.twist_deg.maximum
                )
                report.update(
                    {
                        "twist_deg": twist_deg,
                        "twist_minimum_deg": geometry.twist_deg.minimum,
                        "twist_maximum_deg": geometry.twist_deg.maximum,
                        "twist_satisfied": twist_satisfied,
                    }
                )
                checks.append(twist_satisfied)
            if geometry.contacts is not None:
                contact_count = int(
                    (distances < geometry.contacts.cutoff).sum()
                )
                contacts_satisfied = (
                    contact_count
                    >= geometry.contacts.min_heavy_atom_contacts
                )
                report.update(
                    {
                        "declared_contact_cutoff": geometry.contacts.cutoff,
                        "declared_contact_count": contact_count,
                        "minimum_heavy_atom_contacts": (
                            geometry.contacts.min_heavy_atom_contacts
                        ),
                        "contacts_satisfied": contacts_satisfied,
                    }
                )
                checks.append(contacts_satisfied)
            no_hard_clashes = report["hard_clashes_below_2_0A"] == 0
            report.update(
                {
                    "target_mode": geometry.mode,
                    "satisfied": all(checks) and no_hard_clashes,
                }
            )
        if (
            edge.required
            and edge.satisfaction_stage == "input"
            and not report["satisfied"]
        ):
            failed_required_edges.append(edge.id)
        reports.append(report)

    return {
        "all_required_satisfied": not failed_required_edges,
        "failed_required_edge_instances": failed_required_edges,
        "unsatisfied_output_target_instances": [
            report["edge_instance_id"]
            for report in reports
            if report["satisfaction_stage"] == "output"
            and not report["satisfied"]
        ],
        "edges": reports,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chain_id(index: int) -> str:
    """Return RFD3-safe A..Z, a..z, then extended mmCIF identifiers.

    AtomWorks' current contig grammar accepts one-letter chain identifiers.
    Mixed preexpanded cages can easily need more than 26 source fragment
    chains, so consume the lowercase alphabet before falling back to general
    mmCIF identifiers.  The adapter separately fails closed if an executable
    contig would reference an extended identifier.
    """

    if index < 0:
        raise ValueError("Chain index cannot be negative")
    single = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if index < len(single):
        return single[index]
    # Continue with AA after the 52 single-character identifiers.  Using the
    # original spreadsheet-style value 27 avoids colliding with A..Z.
    value = index - 25
    characters: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        characters.append(chr(ord("A") + remainder))
    return "".join(reversed(characters))


def _continuous_fragment_paths(scaffold_graph) -> tuple[tuple[str, ...], ...]:
    """Return every physical polymer path in deterministic N-to-C order.

    Fixed fragments joined by generated scaffold links are pieces of one
    eventual protein chain.  Giving every fragment its own mmCIF chain wastes
    AtomWorks' one-character contig namespace and made otherwise valid T
    cages fail once they contained more than 52 fragments.  This topology is
    already proven linear and acyclic by ``compile_scaffold_graph``.
    """

    incoming: dict[str, str] = {}
    outgoing: dict[str, str] = {}
    for link in scaffold_graph.links.values():
        if link.chain_break:
            continue
        outgoing[link.from_fragment_instance_id] = (
            link.to_fragment_instance_id
        )
        incoming[link.to_fragment_instance_id] = (
            link.from_fragment_instance_id
        )

    paths: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for start in scaffold_graph.nodes:
        if start in incoming or start in visited:
            continue
        path: list[str] = []
        current = start
        while current not in visited:
            visited.add(current)
            path.append(current)
            if current not in outgoing:
                break
            current = outgoing[current]
        paths.append(tuple(path))
    if visited != set(scaffold_graph.nodes):
        raise RuntimeError(
            "Validated scaffold graph could not be partitioned into "
            "continuous polymer paths"
        )
    return tuple(paths)


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
    polymer_paths = _continuous_fragment_paths(scaffold_graph)
    source_entity_ids = {
        fragment_id: str(index + 1)
        for index, fragment_id in enumerate(spec.fragments)
    }

    # The complete presymmetrized structure may legitimately contain more
    # than AtomWorks' 52 one-character chain identifiers. Only fragments in
    # the native copy-zero ASU contig need that compact namespace; other
    # physical copies can use extended mmCIF identifiers. Preserve the old
    # allocation for small assemblies and prioritize required ASU endpoints
    # only when the namespace would otherwise spill.
    fragment_order = list(instances.fragments)
    compact_polymer_paths = len(fragment_order) > 52
    if compact_polymer_paths:
        chain_id_by_fragment = {
            fragment_id: _chain_id(path_index)
            for path_index, path in enumerate(polymer_paths)
            for fragment_id in path
        }
        entity_id_by_fragment = {
            fragment_id: str(path_index + 1)
            for path_index, path in enumerate(polymer_paths)
            for fragment_id in path
        }
    else:
        # Preserve the historical small-assembly representation byte for
        # byte: one source fragment per input chain and source entity.
        chain_id_by_fragment = {
            fragment_id: _chain_id(index)
            for index, fragment_id in enumerate(fragment_order)
        }
        entity_id_by_fragment = {
            fragment.id: source_entity_ids[fragment.source_id]
            for fragment in instances.fragments.values()
        }

    first_label_seq_id_by_fragment: dict[str, int] = {}
    for path in polymer_paths:
        next_label_seq_id = 1
        for fragment_id in path:
            first_label_seq_id_by_fragment[fragment_id] = next_label_seq_id
            source_id = instances.fragments[fragment_id].source_id
            next_label_seq_id += len({
                atom.residue_id for atom in source_atoms[source_id]
            })

    compiled_atoms: list[_CompiledAtom] = []
    fragment_ranges: dict[str, dict[str, Any]] = {}
    global_residue_index = 0
    for fragment in instances.fragments.values():
        atoms = source_atoms[fragment.source_id]
        coordinates = np.asarray(
            [atom.coordinate for atom in atoms],
            dtype=np.float64,
        )
        transformed = apply_transform(coordinates, fragment.transform)
        chain_id = chain_id_by_fragment[fragment.id]
        entity_id = entity_id_by_fragment[fragment.id]
        first_label_seq_id = (
            first_label_seq_id_by_fragment[fragment.id]
            if compact_polymer_paths
            else 1
        )
        residue_labels: dict[tuple[str, int, str], int] = {}
        fragment_atom_start = len(compiled_atoms)
        fragment_residue_indices: list[int] = []
        for atom, coordinate in zip(atoms, transformed, strict=True):
            if atom.residue_id not in residue_labels:
                residue_labels[atom.residue_id] = (
                    first_label_seq_id + len(residue_labels)
                )
                fragment_residue_indices.append(global_residue_index)
                global_residue_index += 1
            compiled_atoms.append(
                _CompiledAtom(
                    atom_index=len(compiled_atoms),
                    chain_id=chain_id,
                    entity_id=entity_id,
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
            "entity_id": entity_id,
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
        "source_polymer_chain_count": len(polymer_paths),
        "source_polymer_paths": [list(path) for path in polymer_paths],
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
            spec,
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
        clashing_pairs = sorted(
            (
                pair
                for pair in clash_report["group_pairs"]
                if pair["hard_clash_count"]
            ),
            key=lambda pair: (
                -pair["hard_clash_count"],
                pair["minimum_atom_distance"],
            ),
        )
        pair_summary = "; ".join(
            f"{pair['left_group_instance_id']} vs "
            f"{pair['right_group_instance_id']}: "
            f"{pair['hard_clash_count']} pair(s), "
            f"min {pair['minimum_atom_distance']:.3f} A"
            for pair in clashing_pairs[:4]
        )
        remaining = len(clashing_pairs) - 4
        if remaining > 0:
            pair_summary += f"; plus {remaining} additional group pair(s)"
        raise ValueError(
            "Standalone compilation rejected severe inter-group clashes: "
            f"{clash_report['total_hard_clashes']} atom pairs are closer "
            f"than {clash_report['hard_cutoff']:.2f} A; minimum distance "
            f"is {clash_report['minimum_inter_group_distance']:.3f} A; "
            f"clashing groups: {pair_summary}"
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


def _interface_constraint_groups(
    atoms: list[_CompiledAtom],
    instances,
) -> list[dict[str, Any]]:
    """Compile each concrete interface into one cross-chain motif unit."""

    atom_indices_by_fragment: dict[str, list[int]] = {}
    for atom in atoms:
        atom_indices_by_fragment.setdefault(
            atom.fragment_instance_id,
            [],
        ).append(atom.atom_index)

    groups: list[dict[str, Any]] = []
    for edge in instances.interfaces.values():
        left_port = instances.ports[edge.left_port_instance_id]
        right_port = instances.ports[edge.right_port_instance_id]
        left_fragments = tuple(left_port.fragment_instance_ids)
        right_fragments = tuple(right_port.fragment_instance_ids)
        left_atoms = tuple(
            sorted(
                atom_index
                for fragment_id in left_fragments
                for atom_index in atom_indices_by_fragment.get(fragment_id, ())
            )
        )
        right_atoms = tuple(
            sorted(
                atom_index
                for fragment_id in right_fragments
                for atom_index in atom_indices_by_fragment.get(fragment_id, ())
            )
        )
        if not left_atoms or not right_atoms:
            raise ValueError(
                f"Interface constraint group {edge.id!r} must contain atoms "
                "on both sides"
            )
        overlap = sorted(set(left_atoms) & set(right_atoms))
        if overlap:
            raise ValueError(
                f"Interface constraint group {edge.id!r} assigns atoms to "
                f"both sides: {overlap[:8]}"
            )
        groups.append(
            {
                "group_id": edge.id,
                "source_interface_id": edge.source_id,
                "hyperedge_id": edge.hyperedge_id or edge.source_id,
                "orbit_id": edge.orbit_id,
                "transform_set_id": edge.transform_set_id,
                "action_copy_index": edge.action_copy_index,
                "action_transform_id": edge.action_transform_id,
                "physical_edge_index": edge.physical_edge_index,
                "equivalent_action_transform_ids": list(
                    edge.equivalent_action_transform_ids
                ),
                "edge_stabilizer_order": edge.edge_stabilizer_order,
                "left_orbit_id": edge.left_orbit_id,
                "right_orbit_id": edge.right_orbit_id,
                "left_transform_index": edge.left_transform_index,
                "right_transform_index": edge.right_transform_index,
                "source_copy_index": edge.source_copy_index,
                "target_copy_index": edge.target_copy_index,
                "left_port_instance_id": edge.left_port_instance_id,
                "right_port_instance_id": edge.right_port_instance_id,
                "left_fragment_instance_ids": list(left_fragments),
                "right_fragment_instance_ids": list(right_fragments),
                "left_atom_indices": list(left_atoms),
                "right_atom_indices": list(right_atoms),
                "atom_indices": sorted((*left_atoms, *right_atoms)),
            }
        )
    return groups


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
        "interface_constraint_groups": _interface_constraint_groups(
            atoms,
            instances,
        ),
        "fragment_ranges": compilation["fragment_ranges"],
        "source_polymer_chain_count": compilation[
            "source_polymer_chain_count"
        ],
        "source_polymer_paths": compilation["source_polymer_paths"],
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
            "scaffold_link_instances": len(
                compiled_scaffold_links(instances)
            ),
        },
        "interface_edges": [
            edge.model_dump(mode="json")
            for edge in instances.interfaces.values()
        ],
        "scaffold_links": [
            link.model_dump(mode="json")
            for link in compiled_scaffold_links(instances).values()
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
            "Scaffold segment coordinates are not generated in this artifact.",
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
