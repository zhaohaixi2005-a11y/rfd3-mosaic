"""Build a radius-by-tilt stratified shortlist from a pose ensemble."""

import argparse
from bisect import bisect_right
import json
from pathlib import Path
from typing import Any, Iterable


def _parse_edges(value: str) -> list[float]:
    edges = [float(item.strip()) for item in value.split(",")]
    if len(edges) < 2:
        raise ValueError("At least two bin edges are required")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("Bin edges must be strictly increasing")
    return edges


def _bin_index(value: float, edges: list[float]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    index = bisect_right(edges, value) - 1
    return index if 0 <= index < len(edges) - 1 else None


def _sampled_groups(candidates: Iterable[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    for candidate in candidates:
        for group_id, sample in candidate.get(
            "initialization_samples", {}
        ).items():
            if (
                sample.get("sampled_radius") is not None
                and sample.get("principal_axis_tilt_deg") is not None
            ):
                groups.add(group_id)
    return groups


def stratify_candidates(
    ranking: list[dict[str, Any]],
    *,
    radius_edges: list[float],
    tilt_edges: list[float],
    per_cell: int = 1,
    group_id: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain the best scored accepted poses within every occupied cell."""

    if per_cell < 1:
        raise ValueError("per_cell must be at least one")
    feasible = [candidate for candidate in ranking if candidate["accepted"]]
    groups = _sampled_groups(feasible)
    if group_id is None:
        if len(groups) != 1:
            raise ValueError(
                "Cannot infer one stratification group; pass "
                f"--group from {sorted(groups)}"
            )
        group_id = next(iter(groups))
    elif group_id not in groups:
        raise ValueError(
            f"Group {group_id!r} lacks radius/principal-tilt metadata"
        )

    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    outside_count = 0
    for ensemble_rank, candidate in enumerate(ranking, start=1):
        if not candidate["accepted"]:
            continue
        sample = candidate["initialization_samples"][group_id]
        radius = float(sample["sampled_radius"])
        tilt = float(sample["principal_axis_tilt_deg"])
        radius_bin = _bin_index(radius, radius_edges)
        tilt_bin = _bin_index(tilt, tilt_edges)
        if radius_bin is None or tilt_bin is None:
            outside_count += 1
            continue
        cells.setdefault((radius_bin, tilt_bin), []).append(
            {
                **candidate,
                "ensemble_rank": ensemble_rank,
                "stratum": {
                    "radius_bin": radius_bin,
                    "radius_interval": [
                        radius_edges[radius_bin],
                        radius_edges[radius_bin + 1],
                    ],
                    "tilt_bin": tilt_bin,
                    "tilt_interval_deg": [
                        tilt_edges[tilt_bin],
                        tilt_edges[tilt_bin + 1],
                    ],
                },
            }
        )

    shortlist = [
        candidate
        for cell in sorted(cells)
        for candidate in cells[cell][:per_cell]
    ]
    shortlist.sort(key=lambda candidate: candidate["ensemble_rank"])
    coverage = []
    for radius_bin in range(len(radius_edges) - 1):
        for tilt_bin in range(len(tilt_edges) - 1):
            candidates = cells.get((radius_bin, tilt_bin), [])
            coverage.append(
                {
                    "radius_bin": radius_bin,
                    "radius_interval": [
                        radius_edges[radius_bin],
                        radius_edges[radius_bin + 1],
                    ],
                    "tilt_bin": tilt_bin,
                    "tilt_interval_deg": [
                        tilt_edges[tilt_bin],
                        tilt_edges[tilt_bin + 1],
                    ],
                    "candidate_count": len(candidates),
                    "selected_count": min(len(candidates), per_cell),
                }
            )
    coverage.append(
        {
            "outside_configured_bins": outside_count,
            "candidate_count": outside_count,
            "selected_count": 0,
        }
    )
    return group_id, shortlist, coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retain the best accepted pose from each radius-by-principal-tilt "
            "cell instead of allowing one compact orientation family to "
            "dominate a global ranking."
        )
    )
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group")
    parser.add_argument(
        "--radius-edges",
        default="20,22.5,25,27.5,30",
    )
    parser.add_argument(
        "--tilt-edges",
        default="0,22.5,45,67.5,90",
    )
    parser.add_argument("--per-cell", type=int, default=1)
    arguments = parser.parse_args()

    ensemble_path = arguments.ensemble.resolve()
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    radius_edges = _parse_edges(arguments.radius_edges)
    tilt_edges = _parse_edges(arguments.tilt_edges)
    group_id, shortlist, coverage = stratify_candidates(
        ensemble["ranking"],
        radius_edges=radius_edges,
        tilt_edges=tilt_edges,
        per_cell=arguments.per_cell,
        group_id=arguments.group,
    )
    output_path = arguments.output or (
        ensemble_path.parent / "pose_stratified_shortlist.json"
    )
    payload = {
        "schema_version": 1,
        "selector": "rfd3_mosaic.pose_stratify",
        "source_ensemble": str(ensemble_path),
        "group": group_id,
        "radius_edges": radius_edges,
        "tilt_edges_deg": tilt_edges,
        "per_cell": arguments.per_cell,
        "selected_count": len(shortlist),
        "coverage": coverage,
        "shortlist": shortlist,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    occupied = sum(
        item.get("candidate_count", 0) > 0
        for item in coverage
        if "radius_bin" in item
    )
    total = (len(radius_edges) - 1) * (len(tilt_edges) - 1)
    print(f"occupied strata: {occupied}/{total}")
    print(f"selected poses: {len(shortlist)}")
    for candidate in shortlist:
        sample = candidate["initialization_samples"][group_id]
        print(
            f"rank={candidate['ensemble_rank']} "
            f"seed={candidate['pose_seed']} "
            f"radius={sample['sampled_radius']:.3f} "
            f"tilt={sample['principal_axis_tilt_deg']:.3f} "
            f"penalty={candidate['objective_penalty']:.6g} "
            f"cell=r{candidate['stratum']['radius_bin']}-"
            f"t{candidate['stratum']['tilt_bin']}"
        )
    print(f"shortlist: {output_path.resolve()}")


if __name__ == "__main__":
    main()
