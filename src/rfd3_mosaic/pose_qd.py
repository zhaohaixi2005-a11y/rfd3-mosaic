"""Select high-quality poses across diverse ring/cage morphology cells."""

import argparse
import json
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any

from rfd3_mosaic.pose_select import (
    _available_quaternion_groups,
    _quaternion,
    quaternion_angular_distance_degrees,
)
from rfd3_mosaic.pose_stratify import _bin_index, _parse_edges

DEFAULT_DESCRIPTORS = {
    "minimum_axis_clearance_fraction_of_sampled_radius": [
        0.0,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        10.0,
    ],
    "maximum_axial_to_radial_aspect_ratio": [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        10.0,
    ],
}


def _parse_descriptor(value: str) -> tuple[str, list[float]]:
    """Parse NAME=EDGE,EDGE,... from the command line."""

    name, separator, edge_text = value.partition("=")
    if not separator or not name.strip():
        raise ValueError(
            "Descriptor must use NAME=EDGE,EDGE,... syntax"
        )
    return name.strip(), _parse_edges(edge_text)


def _resolve_diversity_group(
    candidates: list[dict[str, Any]],
    group_id: str | None,
) -> str:
    groups = _available_quaternion_groups(candidates)
    if group_id is None:
        if len(groups) != 1:
            raise ValueError(
                "Cannot infer one orientation-diversity group; pass "
                f"--diversity-group from {sorted(groups)}"
            )
        return next(iter(groups))
    if group_id not in groups:
        raise ValueError(
            f"Diversity group {group_id!r} has no sampled quaternion"
        )
    return group_id


def select_quality_diverse_candidates(
    ranking: list[dict[str, Any]],
    *,
    descriptor_edges: dict[str, list[float]],
    per_cell: int = 1,
    max_selected: int | None = None,
    quality_pool_fraction: float = 1.0,
    minimum_orientation_separation_degrees: float = 0.0,
    group_id: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain ranked poses across morphology cells with SO(3) deduplication."""

    if not descriptor_edges:
        raise ValueError("At least one morphology descriptor is required")
    if per_cell < 1:
        raise ValueError("per_cell must be at least one")
    if max_selected is not None and max_selected < 1:
        raise ValueError("max_selected must be at least one")
    if not 0.0 < quality_pool_fraction <= 1.0:
        raise ValueError("quality_pool_fraction must be in (0, 1]")
    if not 0.0 <= minimum_orientation_separation_degrees <= 180.0:
        raise ValueError("Orientation separation must be between 0 and 180")
    for name, edges in descriptor_edges.items():
        if len(edges) < 2 or any(
            right <= left for left, right in zip(edges, edges[1:])
        ):
            raise ValueError(
                f"Descriptor {name!r} requires strictly increasing edges"
            )

    feasible_ranked = [
        (ensemble_rank, candidate)
        for ensemble_rank, candidate in enumerate(ranking, start=1)
        if candidate["accepted"]
    ]
    if not feasible_ranked:
        raise ValueError("Pose ensemble contains no accepted candidates")
    quality_pool_count = max(
        1, ceil(len(feasible_ranked) * quality_pool_fraction)
    )
    quality_pool = feasible_ranked[:quality_pool_count]
    feasible = [candidate for _, candidate in quality_pool]
    diversity_group = _resolve_diversity_group(feasible, group_id)
    descriptor_names = list(descriptor_edges)
    occupied: dict[tuple[int, ...], list[tuple[int, dict[str, Any]]]] = {}
    outside_count = 0
    missing_count = 0
    for ensemble_rank, candidate in quality_pool:
        values = [candidate.get(name) for name in descriptor_names]
        if any(value is None for value in values):
            missing_count += 1
            continue
        bins = tuple(
            _bin_index(float(value), descriptor_edges[name])
            for name, value in zip(descriptor_names, values, strict=True)
        )
        if any(index is None for index in bins):
            outside_count += 1
            continue
        occupied.setdefault(bins, []).append((ensemble_rank, candidate))

    selected: list[dict[str, Any]] = []
    selected_quaternions: list[list[float]] = []
    selected_seeds: set[int] = set()
    selected_per_cell: dict[tuple[int, ...], int] = {}
    for cell_slot in range(per_cell):
        for ensemble_rank, candidate in quality_pool:
            if candidate["pose_seed"] in selected_seeds:
                continue
            if any(candidate.get(name) is None for name in descriptor_names):
                continue
            cell = tuple(
                _bin_index(
                    float(candidate[name]),
                    descriptor_edges[name],
                )
                for name in descriptor_names
            )
            if any(index is None for index in cell):
                continue
            if selected_per_cell.get(cell, 0) != cell_slot:
                continue
            quaternion = _quaternion(candidate, diversity_group)
            separations = [
                quaternion_angular_distance_degrees(quaternion, previous)
                for previous in selected_quaternions
            ]
            minimum_separation = min(separations) if separations else None
            if (
                minimum_separation is not None
                and minimum_separation
                < minimum_orientation_separation_degrees
            ):
                continue
            descriptor_cell = {
                name: {
                    "value": float(candidate[name]),
                    "bin": index,
                    "interval": [
                        descriptor_edges[name][index],
                        descriptor_edges[name][index + 1],
                    ],
                }
                for name, index in zip(
                    descriptor_names, cell, strict=True
                )
            }
            selected.append(
                {
                    **candidate,
                    "ensemble_rank": ensemble_rank,
                    "quality_diversity_cell": descriptor_cell,
                    "minimum_orientation_separation_deg": (
                        minimum_separation
                    ),
                }
            )
            selected_quaternions.append(quaternion)
            selected_seeds.add(candidate["pose_seed"])
            selected_per_cell[cell] = cell_slot + 1
            if max_selected is not None and len(selected) == max_selected:
                break
        if max_selected is not None and len(selected) == max_selected:
            break

    selected.sort(key=lambda candidate: candidate["ensemble_rank"])
    coverage: list[dict[str, Any]] = []
    cell_dimensions = [
        range(len(descriptor_edges[name]) - 1)
        for name in descriptor_names
    ]
    for cell in product(*cell_dimensions):
        candidates = occupied.get(cell, [])
        coverage.append(
            {
                "cell": {
                    name: {
                        "bin": index,
                        "interval": [
                            descriptor_edges[name][index],
                            descriptor_edges[name][index + 1],
                        ],
                    }
                    for name, index in zip(
                        descriptor_names, cell, strict=True
                    )
                },
                "candidate_count": len(candidates),
                "selected_count": selected_per_cell.get(cell, 0),
            }
        )
    coverage.extend(
        [
            {
                "outside_configured_bins": outside_count,
                "candidate_count": outside_count,
                "selected_count": 0,
            },
            {
                "missing_descriptors": missing_count,
                "candidate_count": missing_count,
                "selected_count": 0,
            },
            {
                "excluded_by_quality_pool": (
                    len(feasible_ranked) - quality_pool_count
                ),
                "candidate_count": (
                    len(feasible_ranked) - quality_pool_count
                ),
                "selected_count": 0,
            },
        ]
    )
    return diversity_group, selected, coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the best accepted seed positions across ring/cage "
            "morphology cells while removing near-duplicate SO(3) poses."
        )
    )
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--descriptor",
        action="append",
        help=(
            "Repeatable NAME=EDGE,EDGE,... morphology grid definition. "
            "Defaults to axis clearance and axial/radial aspect ratio."
        ),
    )
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--max-selected", type=int, default=16)
    parser.add_argument(
        "--quality-pool-fraction",
        type=float,
        default=0.25,
        help=(
            "Top fraction of accepted ensemble-ranked candidates eligible "
            "for morphology selection."
        ),
    )
    parser.add_argument(
        "--min-orientation-separation-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument("--diversity-group")
    arguments = parser.parse_args()

    descriptor_edges = (
        dict(_parse_descriptor(value) for value in arguments.descriptor)
        if arguments.descriptor
        else DEFAULT_DESCRIPTORS
    )
    ensemble_path = arguments.ensemble.resolve()
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    group_id, shortlist, coverage = select_quality_diverse_candidates(
        ensemble["ranking"],
        descriptor_edges=descriptor_edges,
        per_cell=arguments.per_cell,
        max_selected=arguments.max_selected,
        quality_pool_fraction=arguments.quality_pool_fraction,
        minimum_orientation_separation_degrees=(
            arguments.min_orientation_separation_deg
        ),
        group_id=arguments.diversity_group,
    )
    output_path = arguments.output or (
        ensemble_path.parent / "pose_qd_shortlist.json"
    )
    occupied_count = sum(
        item.get("candidate_count", 0) > 0
        for item in coverage
        if "cell" in item
    )
    selected_cell_count = sum(
        item.get("selected_count", 0) > 0
        for item in coverage
        if "cell" in item
    )
    payload = {
        "schema_version": 1,
        "selector": "rfd3_mosaic.pose_qd",
        "source_ensemble": str(ensemble_path),
        "quality_source": "source ensemble static-priority ranking",
        "selection_claim": (
            "CPU geometric admissibility and coverage; not fold-quality "
            "prediction"
        ),
        "diversity_group": group_id,
        "descriptor_edges": descriptor_edges,
        "per_cell": arguments.per_cell,
        "max_selected": arguments.max_selected,
        "quality_pool_fraction": arguments.quality_pool_fraction,
        "minimum_orientation_separation_deg": (
            arguments.min_orientation_separation_deg
        ),
        "occupied_cell_count": occupied_count,
        "selected_cell_count": selected_cell_count,
        "selected_count": len(shortlist),
        "coverage": coverage,
        "shortlist": shortlist,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"occupied morphology cells: {occupied_count}")
    print(f"selected morphology cells: {selected_cell_count}")
    print(f"selected poses: {len(shortlist)}")
    for candidate in shortlist:
        separation = candidate["minimum_orientation_separation_deg"]
        separation_text = (
            "reference" if separation is None else f"{separation:.3f} deg"
        )
        print(
            f"rank={candidate['ensemble_rank']} "
            f"seed={candidate['pose_seed']} "
            f"penalty={candidate['objective_penalty']:.6g} "
            f"clearance={candidate['minimum_axis_clearance']:.3f} "
            f"clearance_fraction="
            f"{candidate['minimum_axis_clearance_fraction_of_sampled_radius']:.3f} "
            f"aspect="
            f"{candidate['maximum_axial_to_radial_aspect_ratio']:.3f} "
            f"min_SO3_separation={separation_text}"
        )
    print(f"shortlist: {output_path.resolve()}")


if __name__ == "__main__":
    main()
