"""Translate standalone compiler diagnostics into objective metric names."""

from statistics import fmean
from typing import Any


def build_static_metric_map(
    *,
    clash_report: dict[str, Any],
    interface_report: dict[str, Any],
    linker_report: dict[str, Any],
    cavity_report: dict[str, Any],
) -> dict[str, float]:
    links = linker_report["links"]
    orbits = cavity_report["orbits"]
    metrics = {
        "clashes.total_hard_clashes": float(
            clash_report["total_hard_clashes"]
        ),
        "interfaces.failed_required_count": float(
            len(interface_report["failed_required_edge_instances"])
        ),
        "linkers.infeasible_count": float(
            len(linker_report["infeasible_link_instances"])
        ),
        "linkers.feasible_fraction": (
            sum(bool(link["within_maximum_contour"]) for link in links)
            / len(links)
            if links
            else 1.0
        ),
    }
    minimum_distance = clash_report["minimum_inter_group_distance"]
    if minimum_distance is not None:
        metrics["clashes.minimum_inter_group_distance"] = float(
            minimum_distance
        )
    endpoint_distances = [
        float(link["endpoint_distance"])
        for link in links
        if link.get("endpoint_distance") is not None
    ]
    if endpoint_distances:
        metrics["linkers.minimum_endpoint_distance"] = min(
            endpoint_distances
        )
        metrics["linkers.mean_endpoint_distance"] = fmean(
            endpoint_distances
        )
        metrics["linkers.maximum_endpoint_distance"] = max(
            endpoint_distances
        )
    required_residues = [
        float(link["minimum_required_residues_at_3_8A"])
        for link in links
        if link.get("minimum_required_residues_at_3_8A") is not None
    ]
    if required_residues:
        metrics["linkers.maximum_minimum_required_residues"] = max(
            required_residues
        )
    link_metric_definitions = {
        "scaffolds.maximum_terminal_tangent_to_chord_angle_deg": (
            "from_terminal_tangent_to_chord_angle_deg",
            "to_terminal_tangent_to_chord_angle_deg",
            max,
        ),
        "scaffolds.maximum_terminal_tangent_relative_angle_deg": (
            "terminal_tangent_relative_angle_deg",
            None,
            max,
        ),
        "scaffolds.maximum_terminal_plane_normal_relative_angle_deg": (
            "terminal_plane_normal_relative_angle_deg",
            None,
            max,
        ),
        "scaffolds.maximum_chord_out_of_plane_angle_deg": (
            "endpoint_chord_out_of_plane_angle_deg",
            None,
            max,
        ),
        "scaffolds.minimum_chord_axis_clearance": (
            "minimum_endpoint_chord_axis_clearance",
            None,
            min,
        ),
        "scaffolds.minimum_interior_chord_fixed_atom_clearance": (
            "minimum_interior_chord_fixed_atom_clearance",
            None,
            min,
        ),
    }
    for metric_name, (
        primary_key,
        secondary_key,
        reduction,
    ) in link_metric_definitions.items():
        values = [
            float(value)
            for link in links
            for value in (
                link.get(primary_key),
                link.get(secondary_key) if secondary_key is not None else None,
            )
            if value is not None
        ]
        if values:
            metrics[metric_name] = reduction(values)
    if orbits:
        metrics["cavities.minimum_central_void_radius"] = min(
            float(orbit["central_void_radius"]) for orbit in orbits
        )
        metrics["cavities.minimum_central_void_diameter"] = 2.0 * min(
            float(orbit["central_void_radius"]) for orbit in orbits
        )
        metrics["cavities.minimum_axis_clearance"] = min(
            float(orbit["minimum_axis_clearance"]) for orbit in orbits
        )
        metrics["cavities.minimum_axis_clearance_diameter"] = 2.0 * min(
            float(orbit["minimum_axis_clearance"]) for orbit in orbits
        )
        metrics["assemblies.outer_diameter"] = 2.0 * max(
            float(
                orbit.get(
                    "maximum_center_extent",
                    orbit.get(
                        "maximum_axis_extent",
                        orbit["central_void_radius"],
                    ),
                )
            )
            for orbit in orbits
        )
    return metrics
