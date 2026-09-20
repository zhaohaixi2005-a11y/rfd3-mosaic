"""Independent final-coordinate checks of generated spatial route ownership.

This checks a declared geometric region, not chain linking or knot topology.
Only continuously numbered generated runs with two fixed CA anchors are covered.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _distance_to_chord(
    points: np.ndarray, start: np.ndarray, stop: np.ndarray
) -> np.ndarray:
    direction = stop - start
    fraction = np.clip(
        (points - start) @ direction / max(float(direction @ direction), 1e-12),
        0.0,
        1.0,
    )
    return np.linalg.norm(points - (start + fraction[:, None] * direction), axis=-1)


def audit_generated_route_ownership(
    *,
    coordinates: np.ndarray,
    chain_ids: Sequence[str],
    residue_numbers: Sequence[int],
    fixed_mask: Sequence[bool],
    routing_clearance: float = 3.2,
    routing_anchor_taper_residues: float = 2.0,
    routing_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Check CA points and edge midpoints against every other chain's route.

    Coordinates must contain exactly one protein CA per residue, in chain and
    sequence order. Actual final anchor coordinates define the regions, so a
    rigid motif move does not cause comparison to the original input pose.
    """

    for name, value in (
        ("routing_clearance", routing_clearance),
        ("routing_anchor_taper_residues", routing_anchor_taper_residues),
        ("routing_tolerance", routing_tolerance),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if routing_anchor_taper_residues == 0.0:
        raise ValueError("routing_anchor_taper_residues must be positive")
    xyz = np.asarray(coordinates, dtype=float)
    chains = np.asarray(chain_ids, dtype=str)
    residues = np.asarray(residue_numbers, dtype=int)
    fixed = np.asarray(fixed_mask, dtype=bool)
    if xyz.shape != (len(chains), 3) or not (
        len(residues) == len(fixed) == len(chains)
    ):
        raise ValueError("Route audit requires matched CA coordinates and residue metadata")
    if not np.isfinite(xyz).all():
        raise ValueError("Route audit coordinates must be finite")
    identities = list(zip(chains.tolist(), residues.tolist()))
    if len(set(identities)) != len(identities):
        raise ValueError("Route audit requires unique chain/residue identities")

    runs: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []
    start = 0
    while start < len(xyz):
        if fixed[start]:
            start += 1
            continue
        stop = start
        while (
            stop + 1 < len(xyz)
            and not fixed[stop + 1]
            and chains[stop + 1] == chains[start]
            and residues[stop + 1] == residues[stop] + 1
        ):
            stop += 1
        left = (
            start > 0
            and fixed[start - 1]
            and chains[start - 1] == chains[start]
            and residues[start] == residues[start - 1] + 1
        )
        right = (
            stop + 1 < len(xyz)
            and fixed[stop + 1]
            and chains[stop + 1] == chains[start]
            and residues[stop + 1] == residues[stop] + 1
        )
        record = {
            "chain_id": str(chains[start]),
            "first_generated_residue": int(residues[start]),
            "last_generated_residue": int(residues[stop]),
            "generated_ca_count": stop - start + 1,
            "start_index": start,
            "stop_index": stop,
        }
        if left and right:
            runs.append(record)
        else:
            uncovered.append(
                {
                    key: value for key, value in record.items()
                    if key not in {"start_index", "stop_index"}
                } | {"fixed_anchor_count": int(left) + int(right)}
            )
        start = stop + 1

    ambiguous: list[dict[str, Any]] = []
    reference_margin_conflicts: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        own = xyz[[run["start_index"] - 1, run["stop_index"] + 1]]
        for other_index in range(i + 1, len(runs)):
            other_run = runs[other_index]
            if run["chain_id"] == other_run["chain_id"]:
                continue
            other = xyz[
                [other_run["start_index"] - 1, other_run["stop_index"] + 1]
            ]
            hausdorff = max(
                float(_distance_to_chord(own, other[0], other[1]).max()),
                float(_distance_to_chord(other, own[0], own[1]).max()),
            )
            for run_index, competitor_index in ((i, other_index), (other_index, i)):
                maximum_margin = routing_clearance * min(
                    1.0,
                    (runs[run_index]["generated_ca_count"] + 1)
                    / (2.0 * routing_anchor_taper_residues),
                )
                if hausdorff < maximum_margin - routing_tolerance:
                    reference_margin_conflicts.append({
                        "run_index": run_index,
                        "other_run_index": competitor_index,
                        "distance_difference_upper_bound_angstrom": hausdorff,
                        "required_maximum_margin_angstrom": maximum_margin,
                    })
            if (
                np.max(np.linalg.norm(own - other, axis=-1)) <= routing_tolerance
                or np.max(np.linalg.norm(own - other[::-1], axis=-1)) <= routing_tolerance
            ):
                ambiguous.append({"run_index": i, "other_run_index": other_index})

    checked_samples = 0
    checked_pairs = 0
    violated_samples = 0
    violated_pairs = 0
    maximum_excess = 0.0
    reports: list[dict[str, Any]] = []
    for i, run in enumerate(runs):
        start, stop = run["start_index"], run["stop_index"]
        competitors = [
            j for j, other in enumerate(runs)
            if other["chain_id"] != run["chain_id"]
        ]
        polyline = xyz[start - 1 : stop + 2]
        # Interleave junction/CA-edge midpoints and generated CAs. Fixed CAs
        # themselves are never penalized or counted as generated samples.
        samples = np.empty((2 * run["generated_ca_count"] + 1, 3), dtype=float)
        samples[::2] = 0.5 * (polyline[:-1] + polyline[1:])
        samples[1::2] = polyline[1:-1]
        positions = np.arange(1, len(samples) + 1, dtype=float) / 2.0
        margins = routing_clearance * np.minimum(
            1.0,
            np.minimum(
                positions / routing_anchor_taper_residues,
                (run["generated_ca_count"] + 1 - positions)
                / routing_anchor_taper_residues,
            ),
        )
        own_distances = _distance_to_chord(samples, polyline[0], polyline[-1])
        excesses = []
        for j in competitors:
            other = runs[j]
            other_distances = _distance_to_chord(
                samples,
                xyz[other["start_index"] - 1],
                xyz[other["stop_index"] + 1],
            )
            excesses.append(np.maximum(0.0, margins + own_distances - other_distances))
        matrix = (
            np.stack(excesses, axis=-1)
            if excesses else np.empty((len(samples), 0))
        )
        sample_max = matrix.max(axis=1) if competitors else np.zeros(len(samples))
        violating = sample_max > routing_tolerance
        report = {
            key: value for key, value in run.items()
            if key not in {"start_index", "stop_index"}
        }
        report.update({
            "run_index": i,
            "left_anchor_residue": int(residues[start - 1]),
            "right_anchor_residue": int(residues[stop + 1]),
            "competitor_count": len(competitors),
            "checked_sample_count": len(samples) if competitors else 0,
            "checked_pair_count": int(matrix.size),
            "violated_sample_count": int(violating.sum()),
            "maximum_excess_angstrom": float(sample_max.max()),
            "violating_samples": [
                {
                    "sequence_position_from_left_anchor": float(positions[k]),
                    "sample_kind": "ca" if k % 2 else "edge_midpoint",
                    "required_margin_angstrom": float(margins[k]),
                    "maximum_excess_angstrom": float(sample_max[k]),
                    "worst_competing_run_index": competitors[int(matrix[k].argmax())],
                }
                for k in np.flatnonzero(violating)
            ],
        })
        reports.append(report)
        checked_samples += report["checked_sample_count"]
        checked_pairs += int(matrix.size)
        violated_samples += int(violating.sum())
        violated_pairs += int((matrix > routing_tolerance).sum())
        maximum_excess = max(maximum_excess, report["maximum_excess_angstrom"])

    passed = (
        violated_samples == 0
        and not (routing_clearance > 0.0 and ambiguous)
        and not reference_margin_conflicts
    )
    return {
        "schema_version": 1,
        "measurement": "final_ca_and_edge_midpoint_route_region_ownership",
        "scope": "continuous_two_fixed_anchor_generated_runs",
        "topological_non_interlocking_guarantee": False,
        "applicable": checked_pairs > 0,
        "passed": bool(passed),
        "thresholds": {
            "routing_clearance": routing_clearance,
            "routing_anchor_taper_residues": routing_anchor_taper_residues,
            "routing_tolerance": routing_tolerance,
        },
        "formula": "excess(s,j)=max(0,margin(s)+distance(x_s,S_own)-distance(x_s,S_j)); margin(s)=clearance*min(1,s/taper,(n+1-s)/taper)",
        "bounded_generated_run_count": len(runs),
        "uncovered_generated_run_count": len(uncovered),
        "uncovered_generated_ca_count": sum(run["generated_ca_count"] for run in uncovered),
        "checked_sample_count": checked_samples,
        "checked_pair_count": checked_pairs,
        "violated_sample_count": violated_samples,
        "violated_pair_count": violated_pairs,
        "violated_sample_fraction": violated_samples / checked_samples if checked_samples else 0.0,
        "maximum_excess_angstrom": maximum_excess,
        "ambiguous_reference_pair_count": len(ambiguous),
        "ambiguous_reference_pairs": ambiguous,
        "reference_margin_conflict_count": len(reference_margin_conflicts),
        "reference_margin_conflicts": reference_margin_conflicts,
        "reference_feasibility_interpretation": "A conflict proves this chord-margin contract infeasible, not the protein pose physically impossible; absence of conflicts does not establish feasibility.",
        "runs": reports,
        "uncovered_runs": uncovered,
    }
