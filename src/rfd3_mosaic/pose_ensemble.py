"""Generate and rank rigid Interface-Seed pose candidates without a GPU."""

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from rfd3_mosaic.compile import load_interface_seed_config
from rfd3_mosaic.output import compile_standalone


def _latin_hypercube(
    sample_count: int,
    dimension_count: int,
    *,
    random_seed: int,
) -> np.ndarray:
    """Generate a reproducible randomized Latin hypercube on [0, 1)."""

    if sample_count < 1 or dimension_count < 1:
        raise ValueError("Latin hypercube dimensions must be positive")
    rng = np.random.default_rng(random_seed)
    samples = np.empty((sample_count, dimension_count), dtype=np.float64)
    for dimension in range(dimension_count):
        strata = rng.permutation(sample_count)
        samples[:, dimension] = (
            strata + rng.random(sample_count)
        ) / sample_count
    return samples


def _joint_sample_overrides(
    group_ids: list[str],
    *,
    sample_count: int,
    random_seed: int,
) -> list[dict[str, dict[str, Any]]]:
    """Jointly stratify radius, axial offset, and Haar SO(3) unit inputs."""

    dimensions_per_group = 5
    samples = _latin_hypercube(
        sample_count,
        dimensions_per_group * len(group_ids),
        random_seed=random_seed,
    )
    overrides: list[dict[str, dict[str, Any]]] = []
    for sample in samples:
        candidate: dict[str, dict[str, Any]] = {}
        for group_index, group_id in enumerate(group_ids):
            start = dimensions_per_group * group_index
            candidate[group_id] = {
                "radius_unit": float(sample[start]),
                "axial_offset_unit": float(sample[start + 1]),
                "so3_unit": [
                    float(sample[start + 2]),
                    float(sample[start + 3]),
                    float(sample[start + 4]),
                ],
            }
        overrides.append(candidate)
    return overrides


def _candidate_summary(
    manifest: dict[str, Any],
    *,
    pose_seed: int,
    directory: Path,
) -> dict[str, Any]:
    validation = manifest["validation"]
    clashes = validation["inter_group_clashes"]
    interfaces = validation["interfaces"]
    linkers = validation["scaffold_link_geometry"]
    objectives = validation["objectives"]
    hard_clashes = int(clashes["total_hard_clashes"])
    interface_ok = bool(interfaces["all_required_satisfied"])
    linker_ok = bool(linkers["all_continuous_links_within_maximum_contour"])
    required_failures = int(objectives["required_failure_count"])
    link_reports = linkers.get("links", [])
    endpoint_distances = [
        float(link["endpoint_distance"])
        for link in link_reports
        if link.get("endpoint_distance") is not None
    ]
    cavity_orbits = validation.get("symmetry_cavities", {}).get(
        "orbits", []
    )
    axis_clearances = [
        float(orbit["minimum_axis_clearance"])
        for orbit in cavity_orbits
        if orbit.get("minimum_axis_clearance") is not None
    ]
    accepted = (
        hard_clashes == 0
        and interface_ok
        and linker_ok
        and required_failures == 0
    )
    samples = manifest.get("initialization_samples", {})
    principal_tilts = [
        float(sample["principal_axis_tilt_deg"])
        for sample in samples.values()
        if sample.get("principal_axis_tilt_deg") is not None
    ]
    return {
        "pose_seed": pose_seed,
        "directory": str(directory.resolve()),
        "accepted": accepted,
        "hard_clashes": hard_clashes,
        "minimum_inter_group_distance": float(
            clashes["minimum_inter_group_distance"]
        ),
        "interface_ok": interface_ok,
        "linker_ok": linker_ok,
        "required_objective_failures": required_failures,
        "objective_penalty": float(
            objectives["total_weighted_penalty"]
        ),
        "mean_linker_endpoint_distance": (
            fmean(endpoint_distances) if endpoint_distances else None
        ),
        "maximum_linker_endpoint_distance": (
            max(endpoint_distances) if endpoint_distances else None
        ),
        "minimum_axis_clearance": (
            min(axis_clearances) if axis_clearances else None
        ),
        "maximum_principal_axis_tilt_deg": (
            max(principal_tilts) if principal_tilts else None
        ),
        "initialization_samples": samples,
    }


def _ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Rank feasibility first, configured objectives second, geometry last."""

    maximum_span = item.get("maximum_linker_endpoint_distance")
    mean_span = item.get("mean_linker_endpoint_distance")
    return (
        not item["accepted"],
        item["required_objective_failures"],
        item["hard_clashes"],
        item["objective_penalty"],
        float("inf") if maximum_span is None else maximum_span,
        float("inf") if mean_span is None else mean_span,
        -item["minimum_inter_group_distance"],
        item["pose_seed"],
    )


def compile_pose_ensemble(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    base_directory: str | Path = ".",
    sample_count: int = 32,
    seed_start: int = 0,
    sampling_strategy: str = "random",
) -> dict[str, Any]:
    """Compile, statically validate, and rank reproducible rigid poses."""

    if sample_count < 1:
        raise ValueError("sample_count must be at least one")
    config = Path(config_path).resolve()
    spec = load_interface_seed_config(config)
    group_ids = sorted(spec.initialization)
    if sampling_strategy == "latin_hypercube":
        overrides = _joint_sample_overrides(
            group_ids,
            sample_count=sample_count,
            random_seed=seed_start,
        )
    elif sampling_strategy == "random":
        overrides = [None] * sample_count
    else:
        raise ValueError(
            "sampling_strategy must be 'random' or 'latin_hypercube'"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for index, sample_override in enumerate(overrides):
        pose_seed = seed_start + index
        candidate_directory = output / (
            f"candidate_{index:04d}_seed_{pose_seed}"
        )
        artifacts = compile_standalone(
            config,
            candidate_directory,
            base_directory=base_directory,
            strict_validation=False,
            random_seed=pose_seed,
            sample_overrides=sample_override,
        )
        manifest = json.loads(
            artifacts.manifest_path.read_text(encoding="utf-8")
        )
        candidates.append(
            _candidate_summary(
                manifest,
                pose_seed=pose_seed,
                directory=candidate_directory,
            )
        )

    ranked = sorted(
        candidates,
        key=_ranking_key,
    )
    payload = {
        "schema_version": 1,
        "compiler": "rfd3_mosaic.pose_ensemble",
        "config": str(config),
        "sample_count": sample_count,
        "seed_start": seed_start,
        "sampling_strategy": sampling_strategy,
        "joint_sample_dimensions": [
            "radius",
            "axial_offset",
            "so3_u1",
            "so3_u2",
            "so3_u3",
        ],
        "accepted_count": sum(
            bool(candidate["accepted"]) for candidate in ranked
        ),
        "ranking": ranked,
    }
    manifest_path = output / "pose_ensemble.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["manifest_path"] = str(manifest_path.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample Haar-uniform rigid seed orientations and radial poses, "
            "then rank them with static geometry gates."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-directory", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("random", "latin_hypercube"),
        default="latin_hypercube",
    )
    arguments = parser.parse_args()
    report = compile_pose_ensemble(
        arguments.config,
        arguments.output_dir,
        base_directory=arguments.base_directory,
        sample_count=arguments.samples,
        seed_start=arguments.seed_start,
        sampling_strategy=arguments.sampling_strategy,
    )
    print(
        f"accepted poses: {report['accepted_count']}/"
        f"{report['sample_count']}"
    )
    for candidate in report["ranking"][:10]:
        samples = candidate["initialization_samples"]
        primary_sample = samples.get("primary_seed", {})
        radius = primary_sample.get("sampled_radius")
        maximum_span = candidate["maximum_linker_endpoint_distance"]
        axis_clearance = candidate["minimum_axis_clearance"]
        principal_tilt = candidate["maximum_principal_axis_tilt_deg"]
        radius_text = f"{radius:.3f}" if radius is not None else "NA"
        maximum_span_text = (
            f"{maximum_span:.3f}" if maximum_span is not None else "NA"
        )
        axis_clearance_text = (
            f"{axis_clearance:.3f}"
            if axis_clearance is not None
            else "NA"
        )
        principal_tilt_text = (
            f"{principal_tilt:.3f}"
            if principal_tilt is not None
            else "NA"
        )
        print(
            f"seed={candidate['pose_seed']} "
            f"accepted={candidate['accepted']} "
            f"clashes={candidate['hard_clashes']} "
            f"radius={radius_text} "
            f"max_link_span={maximum_span_text} "
            f"axis_clearance={axis_clearance_text} "
            f"principal_tilt={principal_tilt_text} "
            f"penalty={candidate['objective_penalty']:.6g}"
        )
    print(f"manifest: {report['manifest_path']}")


if __name__ == "__main__":
    main()
