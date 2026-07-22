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
    if orbits:
        metrics["cavities.minimum_central_void_radius"] = min(
            float(orbit["central_void_radius"]) for orbit in orbits
        )
        metrics["cavities.minimum_axis_clearance"] = min(
            float(orbit["minimum_axis_clearance"]) for orbit in orbits
        )
    return metrics
