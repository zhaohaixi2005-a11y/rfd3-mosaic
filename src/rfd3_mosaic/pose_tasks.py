"""Prepare distinct, reproducible task inputs without running diffusion.

The distance spectrum is invariant to overall rigid motion and copy labels.
It is a conservative descriptor, not an RMSD alignment or a complete shape
invariant: different shapes (including mirror images) can share a spectrum.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from rfd3_mosaic.output.standalone import compile_standalone
from rfd3_mosaic.pose_optimizer import _evaluation_from_manifest, _write_assembly
from rfd3_mosaic.sampling_plan import compile_sampling_plan, pose_plan_is_stochastic
from rfd3_mosaic.schema import UserDesignSpec, load_user_design
from rfd3_mosaic.structure import read_structure_atoms


def pose_distance_spectrum(coordinates: np.ndarray, groups: list[str]) -> np.ndarray:
    """Sorted distances between CA atoms in different rigid seed instances."""
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(groups) != len(xyz):
        raise ValueError("Pose coordinates and rigid-group identities disagree")
    if not np.isfinite(xyz).all():
        raise ValueError("Pose coordinates must be finite")
    if len(xyz) > 4096:
        raise ValueError("Pose spectrum supports at most 4096 seed CA atoms")
    labels = np.asarray(groups)
    distances = [
        np.linalg.norm(
            xyz[index + 1 :][labels[index + 1 :] != group] - xyz[index], axis=1
        )
        for index, group in enumerate(groups)
    ]
    spectrum = np.sort(np.concatenate(distances)) if distances else np.empty(0)
    if not spectrum.size:
        raise ValueError("Pose diversity requires at least two rigid seed instances")
    return spectrum


def spectrum_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or not left.size:
        raise ValueError("Cannot compare pose spectra with different atom counts")
    return float(np.sqrt(np.mean(np.square(left - right))))


def _rotation_xyz_degrees(rotation: Any) -> list[float]:
    """Invert the compiler's Rz @ Ry @ Rx convention, including gimbal lock."""
    matrix = np.asarray(rotation, dtype=float)
    y = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(y)) > 1e-8:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        z = 0.0
    return np.degrees([x, y, z]).tolist()


def _freeze_pose(design: UserDesignSpec, samples: dict[str, Any]) -> UserDesignSpec:
    payload = design.model_dump(mode="json", exclude_none=True)
    sampling = payload["sampling"]
    sampling.pop("replicates_per_pose", None)
    sampling.pop("initial_pose", None)
    sampling.pop("initial_poses", None)
    poses = {}
    for group, sample in samples.items():
        radius = sample["sampled_radius"]
        axial = sample["sampled_axial_offset"]
        poses[group] = {
            "radius": {"minimum": radius, "maximum": radius},
            "axial_offset": {"minimum": axial, "maximum": axial},
            "radial_direction": sample["radial_direction"],
            "orientation": {
                "method": "fixed",
                "rotation_deg": _rotation_xyz_degrees(sample["rotation_matrix"]),
            },
        }
    if design.sampling.initial_poses:
        # Compiler motion-group IDs differ from public coupling-group IDs.
        # Candidate construction assigns each declared group its own seed;
        # the compiler records that seed alongside the realized transform.
        remapped = {}
        for component, initial in design.sampling.initial_poses.items():
            matches = [
                group
                for group, sample in samples.items()
                if sample["random_seed"] == initial.seed
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Cannot freeze an ambiguous component-to-motion-group mapping"
                )
            remapped[component] = poses[matches[0]]
        if len(remapped) != len(poses):
            raise ValueError(
                "Every sampled rigid group must have a public pose declaration"
            )
        sampling["initial_poses"] = remapped
    elif len(poses) == 1:
        sampling["initial_pose"] = next(iter(poses.values()))
    else:
        raise ValueError("Multiple seed groups require an explicit component graph")
    return UserDesignSpec.model_validate(payload)


def _compile_pose(design: UserDesignSpec, directory: Path):
    directory.mkdir(parents=True)
    specification = directory / "assembly.yaml"
    _write_assembly(design, specification)
    outputs = compile_standalone(
        specification,
        directory / "compiled",
        base_directory=design.input.parent,
        strict_validation=False,
    )
    manifest = json.loads(outputs.manifest_path.read_text())
    mapping = json.loads(outputs.mapping_path.read_text())
    atoms = read_structure_atoms(
        outputs.structure_path, mmcif_identifier_namespace="label"
    )
    if len(atoms) != len(mapping["atom_mappings"]):
        raise ValueError("Compiled structure and atom mapping disagree")
    coordinates, groups = [], []
    for atom, record in zip(atoms, mapping["atom_mappings"], strict=True):
        if record["source"]["atom_name"] == "CA":
            coordinates.append(atom.coordinate)
            groups.append(record["instance"]["motion_group_instance_id"])
    return (
        manifest,
        np.asarray(coordinates),
        groups,
        np.asarray([atom.coordinate for atom in atoms]),
    )


def prepare_pose_tasks(
    config: Path,
    output_directory: Path,
    *,
    count: int = 4,
    candidates: int = 32,
    seed: int = 0,
    minimum_separation: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rank CPU-feasible candidates, deduplicate, and freeze task-level poses.

    The input must explicitly permit pose sampling. No hidden radius prior,
    task-name randomness, GPU inference, or runtime mobility changes are added.
    """
    if not 1 <= count <= candidates <= 4096 or seed < 0:
        raise ValueError("Require 1 <= count <= candidates <= 4096 and seed >= 0")
    if not math.isfinite(minimum_separation) or minimum_separation <= 0:
        raise ValueError("minimum_separation must be a positive finite angstrom value")
    design = load_user_design(config)
    if not pose_plan_is_stochastic(compile_sampling_plan(design)):
        raise ValueError(
            "prepare-poses needs an explicit variable sampling.initial_pose or "
            "sampling.initial_poses. A fixed input cannot produce distinct poses."
        )
    output = output_directory.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pose task directory: {output}")
    emit = progress or (lambda message: None)
    records, feasible = [], []
    with tempfile.TemporaryDirectory(prefix="mosaic-pose-tasks-") as temporary:
        root = Path(temporary)
        for index in range(candidates):
            candidate_seed = seed + index
            emit(f"pose candidate {index + 1}/{candidates}")
            payload = design.model_dump(mode="json", exclude_none=True)
            sampling = payload["sampling"]
            if sampling.get("initial_pose"):
                sampling["initial_pose"]["seed"] = candidate_seed
            else:
                for group, pose in sampling["initial_poses"].items():
                    digest = hashlib.sha256(
                        f"{candidate_seed}:{group}".encode()
                    ).digest()
                    pose["seed"] = int.from_bytes(digest[:4], "big")
            candidate = UserDesignSpec.model_validate(payload)
            record: dict[str, Any] = {
                "candidate_seed": candidate_seed,
                "accepted": False,
            }
            records.append(record)
            try:
                manifest, xyz, groups, all_xyz = _compile_pose(
                    candidate, root / str(index)
                )
                evaluation = _evaluation_from_manifest(manifest)
                record["geometry"] = evaluation.to_dict()
                record["geometry"]["score"] = [
                    value if math.isfinite(value) else None
                    for value in evaluation.score
                ]
                if not evaluation.feasible:
                    record["reason"] = "hard_geometry_failure"
                    continue
                frozen = _freeze_pose(candidate, manifest["initialization_samples"])
                replay, replay_xyz, replay_groups, replay_all_xyz = _compile_pose(
                    frozen, root / f"replay-{index}"
                )
                if (
                    groups != replay_groups
                    or xyz.shape != replay_xyz.shape
                    or all_xyz.shape != replay_all_xyz.shape
                    or not np.allclose(all_xyz, replay_all_xyz, atol=1e-5, rtol=0)
                    or not np.allclose(
                        xyz,
                        replay_xyz,
                        atol=1e-5,
                        rtol=0,
                    )
                ):
                    raise ValueError(
                        "Frozen pose does not reproduce candidate coordinates"
                    )
                if not _evaluation_from_manifest(replay).feasible:
                    raise ValueError("Frozen pose failed geometry replay")
                signature = pose_distance_spectrum(replay_xyz, replay_groups)
                record["feasible"] = True
                record["initialization_samples"] = manifest["initialization_samples"]
                feasible.append(
                    (evaluation.score, candidate_seed, frozen, signature, record)
                )
            except (ValueError, NotImplementedError) as error:
                record["reason"] = str(error)
        selected = []
        for _, candidate_seed, frozen, signature, record in sorted(
            feasible, key=lambda row: (row[0], row[1])
        ):
            separation = min(
                (spectrum_distance(signature, row[2]) for row in selected), default=None
            )
            record["minimum_selected_separation_angstrom"] = separation
            if separation is not None and separation < minimum_separation:
                record["reason"] = "duplicate_pose_geometry"
                continue
            if len(selected) >= count:
                record["reason"] = "requested_count_reached"
                continue
            record["accepted"] = True
            selected.append((candidate_seed, frozen, signature, record))

        # Only publish the task directory after candidate evaluation completes.
        output.mkdir(parents=True)
        inputs = output / "inputs"
        inputs.mkdir()
        frozen_input = inputs / design.input.name
        shutil.copy2(design.input, frozen_input)
        tasks = []
        for index, (candidate_seed, frozen, _, record) in enumerate(selected):
            task_id = f"pose-{index:03d}-s{candidate_seed}"
            payload = frozen.model_dump(mode="json", exclude_none=True)
            payload["name"] = task_id
            payload["input"] = str(frozen_input)
            payload["output"] = {
                "root": str(
                    design.output.root
                    if design.output is not None
                    else config.resolve().parent / "runs" / "rfd3-mosaic"
                ),
                "campaign": task_id,
            }
            task_path = output / f"{task_id}.yaml"
            UserDesignSpec.model_validate(payload)
            task_path.write_text(yaml.safe_dump(payload, sort_keys=False))
            tasks.append(
                {
                    "task": str(task_path),
                    "candidate_seed": candidate_seed,
                    "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                    "designs": frozen.sampling.designs,
                    "minimum_selected_separation_angstrom": record[
                        "minimum_selected_separation_angstrom"
                    ],
                }
            )
        report = {
            "schema_version": 1,
            "pose_scope": "task",
            "source": str(config.resolve()),
            "source_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "frozen_input": str(frozen_input),
            "input_sha256": hashlib.sha256(frozen_input.read_bytes()).hexdigest(),
            "requested_tasks": count,
            "selected_tasks": len(tasks),
            "candidate_count": candidates,
            "seed": seed,
            "minimum_separation_angstrom": minimum_separation,
            "distance_metric": "RMS difference of sorted inter-seed CA pair distances",
            "limitations": "Conservative pose descriptor, not aligned RMSD or folding success; distinct shapes may share a spectrum. Runtime mobility is unchanged and may reduce final pose diversity.",
            "tasks": tasks,
            "candidates": records,
        }
        (output / "pose_tasks.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report
