"""Select a geometry-ranked, orientation-diverse rigid-pose shortlist."""

import argparse
import json
from math import acos, degrees, sqrt
from pathlib import Path
from typing import Any, Iterable


def quaternion_angular_distance_degrees(
    left: Iterable[float],
    right: Iterable[float],
) -> float:
    """Return the shortest SO(3) angle between xyzw unit quaternions."""

    q_left = tuple(float(value) for value in left)
    q_right = tuple(float(value) for value in right)
    if len(q_left) != 4 or len(q_right) != 4:
        raise ValueError("Orientation quaternions must contain four values")
    left_norm = sqrt(sum(value * value for value in q_left))
    right_norm = sqrt(sum(value * value for value in q_right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Orientation quaternions must be nonzero")
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(q_left, q_right, strict=True)
    ) / (left_norm * right_norm)
    # q and -q encode the same rotation.
    return degrees(2.0 * acos(min(1.0, max(0.0, abs(dot)))))


def _available_quaternion_groups(
    candidates: Iterable[dict[str, Any]],
) -> set[str]:
    groups: set[str] = set()
    for candidate in candidates:
        for group_id, sample in candidate.get(
            "initialization_samples", {}
        ).items():
            if sample.get("quaternion_xyzw") is not None:
                groups.add(group_id)
    return groups


def _quaternion(
    candidate: dict[str, Any],
    group_id: str,
) -> list[float]:
    try:
        quaternion = candidate["initialization_samples"][group_id][
            "quaternion_xyzw"
        ]
    except KeyError as error:
        raise ValueError(
            f"Pose seed {candidate.get('pose_seed')} has no sampled "
            f"quaternion for group {group_id!r}"
        ) from error
    if quaternion is None:
        raise ValueError(
            f"Pose seed {candidate.get('pose_seed')} uses a fixed "
            f"orientation for group {group_id!r}"
        )
    return quaternion


def select_diverse_candidates(
    ranking: list[dict[str, Any]],
    *,
    count: int,
    minimum_separation_degrees: float,
    pool_size: int,
    group_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Greedily retain ranked poses separated by a minimum SO(3) angle."""

    if count < 1:
        raise ValueError("Shortlist count must be at least one")
    if pool_size < count:
        raise ValueError("Pool size cannot be smaller than shortlist count")
    if not 0.0 <= minimum_separation_degrees <= 180.0:
        raise ValueError("Orientation separation must be between 0 and 180")

    feasible = [candidate for candidate in ranking if candidate["accepted"]]
    pool = feasible[:pool_size]
    if not pool:
        raise ValueError("Pose ensemble contains no accepted candidates")
    groups = _available_quaternion_groups(pool)
    if group_id is None:
        if len(groups) != 1:
            raise ValueError(
                "Cannot infer one diversity group; pass --diversity-group "
                f"from {sorted(groups)}"
            )
        group_id = next(iter(groups))
    elif group_id not in groups:
        raise ValueError(
            f"Diversity group {group_id!r} has no sampled quaternion"
        )

    selected: list[dict[str, Any]] = []
    selected_quaternions: list[list[float]] = []
    for pool_rank, candidate in enumerate(pool, start=1):
        quaternion = _quaternion(candidate, group_id)
        separations = [
            quaternion_angular_distance_degrees(quaternion, previous)
            for previous in selected_quaternions
        ]
        minimum_separation = min(separations) if separations else None
        if (
            minimum_separation is not None
            and minimum_separation < minimum_separation_degrees
        ):
            continue
        selected.append(
            {
                **candidate,
                "ensemble_rank": pool_rank,
                "minimum_orientation_separation_deg": minimum_separation,
            }
        )
        selected_quaternions.append(quaternion)
        if len(selected) == count:
            break
    return group_id, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select accepted poses in score order while enforcing a minimum "
            "quaternion SO(3) separation."
        )
    )
    parser.add_argument("--ensemble", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument(
        "--min-orientation-separation-deg",
        type=float,
        default=30.0,
    )
    parser.add_argument("--diversity-group")
    arguments = parser.parse_args()

    ensemble_path = arguments.ensemble.resolve()
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    group_id, shortlist = select_diverse_candidates(
        ensemble["ranking"],
        count=arguments.count,
        minimum_separation_degrees=(
            arguments.min_orientation_separation_deg
        ),
        pool_size=arguments.pool_size,
        group_id=arguments.diversity_group,
    )
    output_path = arguments.output or (
        ensemble_path.parent / "pose_shortlist.json"
    )
    payload = {
        "schema_version": 1,
        "selector": "rfd3_mosaic.pose_select",
        "source_ensemble": str(ensemble_path),
        "diversity_group": group_id,
        "requested_count": arguments.count,
        "selected_count": len(shortlist),
        "pool_size": arguments.pool_size,
        "minimum_orientation_separation_deg": (
            arguments.min_orientation_separation_deg
        ),
        "shortlist": shortlist,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"selected poses: {len(shortlist)}/{arguments.count} "
        f"from top {arguments.pool_size} accepted candidates"
    )
    print(f"diversity group: {group_id}")
    for candidate in shortlist:
        separation = candidate["minimum_orientation_separation_deg"]
        separation_text = (
            "reference" if separation is None else f"{separation:.3f} deg"
        )
        print(
            f"rank={candidate['ensemble_rank']} "
            f"seed={candidate['pose_seed']} "
            f"penalty={candidate['objective_penalty']:.6g} "
            f"max_link_span="
            f"{candidate['maximum_linker_endpoint_distance']:.3f} "
            f"axis_clearance={candidate['minimum_axis_clearance']:.3f} "
            f"min_orientation_separation={separation_text}"
        )
    if len(shortlist) < arguments.count:
        print(
            "warning: the requested diversity threshold yielded fewer poses; "
            "increase --pool-size or lower the separation threshold"
        )
    print(f"shortlist: {output_path.resolve()}")


if __name__ == "__main__":
    main()
