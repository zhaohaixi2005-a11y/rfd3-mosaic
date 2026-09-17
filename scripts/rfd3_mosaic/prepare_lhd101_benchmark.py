#!/usr/bin/env python3
"""Freeze a LHD101-only benchmark; submission is a separate site operation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from rfd3_mosaic.pose_optimizer import _evaluation_from_manifest
from rfd3_mosaic.pose_tasks import _compile_pose, prepare_pose_tasks
from rfd3_mosaic.schema import load_user_design


def pose(low, high, *, direction=(1.0, 0.0, 0.0), axial=0):
    return {
        "radius": {"minimum": low, "maximum": high},
        "axial_offset": {"minimum": -axial, "maximum": axial},
        "radial_direction": list(direction),
        "orientation": {"method": "uniform_so3"},
        "seed": 0,
    }


def base(input_path, symmetry, designs):
    return {
        "schema_version": 1,
        "name": "lhd101-benchmark-template",
        "input": str(input_path),
        "symmetry": symmetry,
        "preferences": {"component_motion": "locked"},
        "sampling": {
            "designs": designs,
            "seed": 9170000,
            "timesteps": 50,
            "preset": "exact_mosaic",
            "low_memory_mode": True,
            "execution_backend": "explicit_all_copy",
            "scaffold_packing": "off",
        },
        "output": {"root": "runs/rfd3-mosaic", "campaign": "lhd101-benchmark"},
    }


def fixed(selector, group):
    return {
        "kind": "fixed_xyz",
        "selector": selector,
        "atoms": "all",
        "coupling_group": group,
    }


def families(input_path, designs):
    """Each tuple is one geometry family with explicitly paired motion arms."""
    result = []
    for symmetry, site in [("C2", "lrz"), ("C3", "lmu"), ("C4", "lrz"), ("C6", "lmu")]:
        payload = base(input_path, symmetry, designs)
        payload["task"] = "preserve_supplied_geometry"
        payload["guidance"] = {"intra_chain_weight": 1.0}
        payload["constraints"] = [
            fixed("A165-194", "interface_seed"),
            fixed("B211-241", "interface_seed"),
        ]
        payload["generation"] = [
            {
                "kind": "between",
                "from_selector": "B211-241",
                "to_selector": "A165-194",
                "orbit_offset": "nearest_adjacent",
                "length": 100,
            }
        ]
        payload["sampling"]["initial_pose"] = pose(20, 35)
        modes = ("locked", "guided", "free") if symmetry == "C3" else ("locked", "free")
        result.append(
            (
                f"{symmetry.lower()}-interface",
                payload,
                modes,
                2 if symmetry == "C3" else 1,
                site,
                "complete A/B interface",
            )
        )

    payload = base(input_path, "C3", designs)
    payload["task"] = "create_symmetric_interface"
    payload["sampling"].pop("scaffold_packing")
    payload["constraints"] = [fixed("A165-194", "motif")]
    payload["generation"] = [
        {"kind": "terminal", "anchor": "A165-194", "terminus": end, "length": 55}
        for end in ("n", "c")
    ]
    payload["sampling"]["initial_pose"] = pose(20, 30)
    result.append(
        (
            "c3-new-packing",
            payload,
            ("locked", "guided"),
            1,
            "lmu",
            "LHD101 A165-194 only; new interface task",
        )
    )

    payload = base(input_path, "C3", designs)
    payload["task"] = "preserve_supplied_geometry"
    payload["constraints"] = [
        fixed("A165-194", "interface_seed"),
        fixed("B211-241", "interface_seed"),
    ]
    payload["generation"] = [
        {"kind": "terminal", "anchor": anchor, "terminus": end, "length": 35}
        for anchor in ("A165-194", "B211-241")
        for end in ("n", "c")
    ]
    payload["sampling"].update(
        initial_pose=pose(20, 32), scaffold_packing="symmetric_generated"
    )
    result.append(
        (
            "c3-complex-extension",
            payload,
            ("guided",),
            1,
            "lrz",
            "complete A/B interface; four terminal extensions",
        )
    )

    payload = base(input_path, "C3", designs)
    payload["components"] = {
        name: {"selectors": [selector]}
        for name, selector in [
            ("alpha", "A165-174"),
            ("beta", "A178-184"),
            ("gamma", "A188-194"),
        ]
    }
    payload["connections"] = [
        {"id": "alpha_beta", "from": "alpha.C", "to": "beta.N", "length": 30},
        {"id": "beta_gamma", "from": "beta.C", "to": "gamma.N", "length": 30},
    ]
    payload["sampling"]["initial_poses"] = {
        name: pose(25 + 15 * i, 30 + 15 * i, axial=5)
        for i, name in enumerate(payload["components"])
    }
    result.append(
        (
            "c3-component-graph",
            payload,
            ("free",),
            1,
            "lmu",
            "three independent LHD101 A-chain fragments; not the full A/B interface",
        )
    )

    for symmetry, radii, site in [
        ("D2", (25, 45), "lmu"),
        ("D3", (30, 50), "lrz"),
        ("D4", (35, 55), "lrz"),
        ("T", (60, 90), "lrz"),
        ("O", (95, 125), "lrz"),
    ]:
        payload = base(input_path, symmetry, designs)
        payload["components"] = {
            "alpha": {"selectors": ["A165-174"]},
            "beta": {"selectors": ["A185-194"]},
        }
        payload["connections"] = [
            {"id": "alpha_beta", "from": "alpha.C", "to": "beta.N", "length": 60}
        ]
        payload["sampling"]["initial_poses"] = {
            name: pose(
                radius,
                radius + 10,
                direction=(0.74285724, 0.52000009, 0.42161954),
                axial=5,
            )
            for name, radius in zip(("alpha", "beta"), radii)
        }
        result.append(
            (
                f"{symmetry.lower()}-multi-orbit",
                payload,
                ("free",),
                1,
                site,
                "two independent LHD101 fragments; symmetry/mobility/continuity, not a connected interface-seeded cage",
            )
        )

    payload = base(input_path, "I", designs)
    payload["task"] = "preserve_supplied_geometry"
    payload["constraints"] = [fixed("A165-194", "motif")]
    payload["generation"] = [
        {"kind": "terminal", "anchor": "A165-194", "terminus": "c", "length": 20}
    ]
    payload["sampling"]["initial_pose"] = pose(
        175, 185, direction=(0.8017837, 0.5345225, 0.2672612)
    )
    # Match the existing I-continuity canary orientation; random orientation
    # at this direction can overlap neighbouring copies of the long motif.
    payload["sampling"]["initial_pose"]["orientation"] = {
        "method": "fixed",
        "rotation_deg": [17.0, 29.0, 11.0],
    }
    result.append(
        (
            "i-continuity",
            payload,
            ("locked",),
            1,
            "lrz",
            "LHD101 A165-194 only; 60-copy static continuity, not cage packing",
        )
    )
    return result


def apply_motion(payload, mode):
    payload["preferences"]["component_motion"] = mode
    payload["fixed_arrangement"] = (
        "locked"
        if mode == "locked" or payload.get("components")
        else "optimize_components"
    )
    if payload.get("components") and mode != "locked":
        for component in payload["components"].values():
            component["pose"] = {
                "mode": "bounded_mobile",
                "subspace": "bounded_se3",
                "proposal": "scaffold_objectives",
                "max_translation": 3.0,
                "max_rotation_deg": 10.0,
                "start_fraction": 0.05,
                "end_fraction": 0.75,
                "response": 0.2,
                "max_step_translation": 0.25,
                "max_step_rotation_deg": 1.0,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--designs", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.designs < 50:
        raise ValueError("This benchmark requires at least 50 designs per task")
    project = Path(__file__).resolve().parents[2]
    input_path = project / "examples/rfd3_mosaic/lhd101_c3/inputs/7mwr_interface.pdb"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "designs_per_task": args.designs,
        "timesteps": 50,
        "records": [],
        "excluded": [
            {
                "case": "c4-c2-quotient",
                "reason": "No validated intrinsic C2 LHD101 component supplied; do not impose a false stabilizer",
            }
        ],
        "scope": "Single-source LHD101 benchmark, not cross-seed generalization. Complete A/B geometry is preserved only in declared interface/complex tasks. Motion pairs share compiled coordinates; each YAML shares one pose across all designs.",
    }
    manifest_path = output / "benchmark_manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if (
            previous["input_sha256"] != manifest["input_sha256"]
            or previous["designs_per_task"] != args.designs
        ):
            raise ValueError("Resume input or design count changed")
        manifest = previous
    for index, (name, payload, modes, count, site, scope) in enumerate(
        families(input_path, args.designs)
    ):
        source = output / f"{name}-template.yaml"
        source.write_text(yaml.safe_dump(payload, sort_keys=False))
        print(f"Preparing {name}", flush=True)
        report_path = output / "poses" / name / "pose_tasks.json"
        if args.resume and report_path.exists():
            report = json.loads(report_path.read_text())
            if (
                report["source_sha256"]
                != hashlib.sha256(source.read_bytes()).hexdigest()
            ):
                raise ValueError(f"Resume template changed: {name}")
        else:
            report = prepare_pose_tasks(
                source,
                output / "poses" / name,
                count=count,
                candidates=args.candidates,
                seed=91700 + index * 100,
                progress=lambda message: print(message, flush=True),
            )
        if not report["tasks"]:
            manifest["excluded"].append(
                {"case": name, "reason": "No feasible candidate; see pose report"}
            )
        for pose_index, task in enumerate(report["tasks"]):
            frozen = yaml.safe_load(Path(task["task"]).read_text())
            baseline = None
            for mode in modes:
                config = deepcopy(frozen)
                task_name = f"lhd-{name}-p{pose_index}-{mode}"
                if any(record["task"] == task_name for record in manifest["records"]):
                    continue
                config["name"] = task_name
                config["sampling"]["seed"] = (
                    9170000 + index * 10000 + pose_index * args.designs
                )
                apply_motion(config, mode)
                path = output / f"{task_name}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                design = load_user_design(path)
                geometry, _, _, coordinates = _compile_pose(
                    design, output / "checks" / task_name
                )
                if not _evaluation_from_manifest(geometry).feasible:
                    raise ValueError(f"Frozen task is infeasible: {task_name}")
                if baseline is not None and not np.allclose(
                    baseline, coordinates, atol=1e-5, rtol=0
                ):
                    raise ValueError(
                        f"Motion pair initial coordinates disagree: {task_name}"
                    )
                baseline = coordinates
                record = {
                    "task": task_name,
                    "config": str(path),
                    "site": site,
                    "family": name,
                    "motion": mode,
                    "pose_index": pose_index,
                    "pose_candidate_seed": task["candidate_seed"],
                    "diffusion_seed": config["sampling"]["seed"],
                    "designs": args.designs,
                    "scope": scope,
                    "coordinate_sha256": hashlib.sha256(
                        coordinates.tobytes()
                    ).hexdigest(),
                }
                manifest["records"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {len(manifest['records'])} tasks / {sum(item['designs'] for item in manifest['records'])} designs",
        flush=True,
    )


if __name__ == "__main__":
    main()
