#!/usr/bin/env python3
"""Replay every frozen benchmark task with disjoint seeds and bounded jobs.

Prepare never submits. Submit records scheduler receipts before issuing the
next array; it refuses to repeat an uncertain submission. The first design
of EVERY task forms the canary array. Remaining designs depend on successful
execution of that site's canaries, not on scientific quality thresholds.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess


def partition_designs(count: int, chunk_size: int) -> list[tuple[int, int]]:
    if count < 1 or chunk_size < 1:
        raise ValueError("Design count and chunk size must be positive")
    return [(0, 1)] + [(start, min(count, start + chunk_size))
                       for start in range(1, count, chunk_size)]


def group_by_runtime(rows, seconds_per_design, budget_seconds=28800):
    """First-fit decreasing with 25% timing margin and 3 min per shard startup.

    Groups only change scheduler occupancy; every frozen shard remains an
    independent worker with its original seed range and audit directory.
    Measured times are estimates, not a guarantee against a slower GPU.
    """
    import math
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Runtime budget must be finite and positive")
    timed = []
    for row in rows:
        rate = float(seconds_per_design[row["task"]])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Runtime estimates must be finite and positive")
        cost = 180 + 1.25 * row["designs"] * rate
        if cost > budget_seconds:
            raise ValueError(f"One shard exceeds runtime budget: {row['script']}")
        timed.append((cost, row))
    bins = []
    for cost, row in sorted(timed, key=lambda item: -item[0]):
        for group in bins:
            if group["estimated_seconds"] + cost <= budget_seconds:
                group["rows"].append(row)
                group["estimated_seconds"] += cost
                break
        else:
            bins.append({"rows": [row], "estimated_seconds": cost})
    return bins


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(args) -> None:
    import yaml
    from rfd3_mosaic.cli import main as cli_main
    from rfd3_mosaic.pose_tasks import _compile_pose
    from rfd3_mosaic.pose_optimizer import _evaluation_from_manifest
    from rfd3_mosaic.schema import load_user_design
    from rfd3_mosaic.run_layout import dated_experiment_root

    source = json.loads(args.manifest.read_text())
    records = source["records"]
    if len({r["task"] for r in records}) != len(records):
        raise ValueError("Duplicate benchmark tasks")
    if not any(r["site"] == args.site for r in records):
        raise ValueError("No tasks for this site")
    if sha(args.input) != source["input_sha256"]:
        raise ValueError("Input seed hash differs from baseline")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != args.revision:
        raise ValueError("Source revision mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Replay requires a clean committed checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    profile = yaml.safe_load(args.profile.read_text())
    checkpoint = Path(profile["checkpoint"])
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from baseline")
    sizes = json.loads(args.shard_sizes.read_text()) if args.shard_sizes else {}
    unknown = set(sizes) - {r["task"] for r in records}
    if unknown:
        raise ValueError(f"Unknown shard-size tasks: {unknown}")
    receipt = {
        "schema_version": 1, "site": args.site, "revision": revision,
        "baseline_manifest_sha256": sha(args.manifest),
        "input_sha256": source["input_sha256"], "checkpoint_sha256": checkpoint_sha,
        "profile": profile, "campaign": args.campaign, "records": [],
        "all_task_names": [r["task"] for r in records],
    }
    manifest_path = args.output / "replay_manifest.json"
    for item in records:
        if item["site"] != args.site:
            continue
        payload = yaml.safe_load((args.config_root / Path(item["config"]).name).read_text())
        if payload["sampling"]["designs"] != item["designs"] or item["designs"] < 50:
            raise ValueError("Baseline task must retain at least 50 designs")
        if payload["sampling"]["seed"] != item["diffusion_seed"]:
            raise ValueError("Baseline diffusion seed mismatch")
        payload["input"] = str(args.input.resolve())
        payload["output"] = {"root": str(args.run_root.resolve()), "campaign": args.campaign}
        payload["resources"] = {"profile": args.site, **{
            k: profile["slurm"][k] for k in ("cpus", "memory", "walltime")
        }}
        parent_config = args.output / (item["task"] + ".yaml")
        parent_config.write_text(yaml.safe_dump(payload, sort_keys=False))
        manifest, _, _, coordinates = _compile_pose(
            load_user_design(parent_config), args.output / "geometry" / item["task"]
        )
        coordinate_sha = hashlib.sha256(coordinates.tobytes()).hexdigest()
        if coordinate_sha != item["coordinate_sha256"]:
            raise ValueError(f"Pose changed: {item['task']}")
        if not _evaluation_from_manifest(manifest).feasible:
            raise ValueError(f"Frozen pose no longer passes preflight: {item['task']}")
        size = int(sizes.get(item["task"], args.shard_size))
        for start, stop in partition_designs(item["designs"], size):
            payload["name"] = f"{item['task']}-d{start:03d}-{stop-1:03d}"
            payload["sampling"]["seed"] = item["diffusion_seed"] + start
            payload["sampling"]["designs"] = stop - start
            config = args.output / (payload["name"] + ".yaml")
            config.write_text(yaml.safe_dump(payload, sort_keys=False))
            job_dir = args.output / "jobs" / payload["name"]
            log = io.StringIO()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                cli_main(["render", str(config), "--profile", str(args.profile),
                          "--defer-runtime-preflight", "--output-dir", str(job_dir)])
            (job_dir / "render.log").write_text(log.getvalue())
            resolved = yaml.safe_load((job_dir / "resolved_config.yaml").read_text())
            receipt["records"].append({
                "task": item["task"], "family": item["family"], "motion": item["motion"],
                "pose_index": item["pose_index"], "coordinate_sha256": coordinate_sha,
                "global_start": start, "global_stop": stop, "designs": stop-start,
                "seed": item["diffusion_seed"] + start, "stage": "canary" if start == 0 else "bulk",
                "config": str(config), "config_sha256": sha(config),
                "script": str(job_dir / "generated_job.sbatch"),
                "run_root": str(dated_experiment_root(
                    resolved["output"]["root"], run_day=resolved["output"]["run_date"],
                    experiment=resolved["name"],
                )),
            })
            save(manifest_path, receipt)
        print(f"Prepared {item['task']}: {item['designs']} designs; original pose verified", flush=True)
    receipt["prepared"] = True
    save(manifest_path, receipt)


def dispatch(path: Path, index: int) -> None:
    scripts = json.loads(path.read_text())
    if not 0 <= index < len(scripts):
        raise ValueError("Array index outside manifest")
    selected = scripts[index]
    if isinstance(selected, str):
        os.execv("/bin/bash", ["/bin/bash", selected])
    failed = False
    for script in selected:
        # One shard failing must not erase unrelated benchmarks in this group.
        result = subprocess.run(["/bin/bash", script])
        failed = failed or result.returncode != 0
    raise SystemExit(1 if failed else 0)


def submit(args) -> None:
    receipt = json.loads(args.manifest.read_text())
    if not receipt.get("prepared"):
        raise ValueError("Not prepared")
    previous = receipt.get("submissions", [])
    dependency = None
    stages = ("canary", "bulk")
    if previous:
        # Retry only a scheduler-confirmed QOS rejection, never an uncertain
        # network result or an accepted job. Keep the rejected receipt intact.
        rejected = previous[-1]
        canaries = [s for s in previous if s["stage"] == "canary" and s["status"] == "submitted"]
        safe_retry = (
            args.retry_qos_rejection and len(canaries) == 1
            and rejected["stage"] == "bulk" and rejected.get("returncode", 0) != 0
            and not rejected.get("stdout", "").strip()
            and "QOSMaxSubmitJobPerUserLimit" in rejected.get("stderr", "")
            and not any(s["stage"] == "bulk" and s["status"] == "submitted" for s in previous)
        )
        if not safe_retry:
            raise ValueError("Already or uncertainly submitted; inspect receipts before retrying")
        dependency = canaries[0]["job_id"]
        stages = ("bulk",)
    if args.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    receipt.setdefault("submissions", [])
    output = args.manifest.parent
    timings = json.loads(args.runtime_estimates.read_text()) if args.runtime_estimates else None
    for stage in stages:
        rows = [r for r in receipt["records"] if r["stage"] == stage]
        if not rows:
            continue
        jobs = output / (stage + "_scripts.json")
        groups = (
            group_by_runtime(rows, timings, args.runtime_budget_seconds)
            if stage == "bulk" and timings is not None
            else [{"rows": [r], "estimated_seconds": None} for r in rows]
        )
        save(jobs, [[r["script"] for r in g["rows"]] for g in groups])
        slurm = receipt["profile"]["slurm"]
        lines = ["#!/bin/bash -l", "#SBATCH --job-name=mosaic-replay-" + stage]
        for flag, key in (("partition", "partition"), ("gres", "gres"), ("cpus-per-task", "cpus"),
                          ("mem", "memory"), ("time", "walltime"), ("account", "account"), ("qos", "qos")):
            if slurm.get(key):
                lines.append(f"#SBATCH --{flag}={slurm[key]}")
        lines += [f"#SBATCH --array=0-{len(groups)-1}%{args.concurrency}",
                  f"#SBATCH --output={output}/{stage}-%A_%a.out",
                  f"#SBATCH --error={output}/{stage}-%A_%a.err"]
        if dependency:
            lines.append("#SBATCH --dependency=afterok:" + dependency)
        lines += ["set -euo pipefail", "exec python3 " + shlex.quote(str(Path(__file__).resolve()))
                  + " dispatch " + shlex.quote(str(jobs)) + ' "$SLURM_ARRAY_TASK_ID"']
        script = output / (stage + ".sbatch")
        script.write_text("\n".join(lines) + "\n")
        submission = {
            "stage": stage, "status": "submission_started", "script": str(script),
            "submission_driver_sha256": sha(Path(__file__)),
            "array_tasks": len(groups), "shards": len(rows),
            "runtime_budget_seconds": args.runtime_budget_seconds if timings else None,
        }
        receipt["submissions"].append(submission)
        save(args.manifest, receipt)
        result = subprocess.run(["sbatch", "--parsable", str(script)], text=True, capture_output=True)
        submission.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        if result.returncode:
            submission["status"] = "failed_or_uncertain"
            save(args.manifest, receipt)
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(";")[0]
        if not job.isdecimal():
            save(args.manifest, receipt)
            raise RuntimeError("Unrecognized scheduler receipt; do not blindly resubmit")
        submission.update(status="submitted", job_id=job)
        for i, group in enumerate(groups):
            for position, row in enumerate(group["rows"]):
                row["job_id"] = f"{job}_{i}"
                row["group_position"] = position
                row["group_estimated_seconds"] = group["estimated_seconds"]
                row["expected_run_directory"] = str(Path(row["run_root"]) / row["job_id"])
                save(Path(row["script"]).parent / "submission.json", {
                    "job_id": row["job_id"], "executor": "slurm", "script": row["script"],
                    "run_root": row["run_root"], "expected_run_directory": row["expected_run_directory"],
                })
        save(args.manifest, receipt)
        dependency = job
        print(f"Submitted {stage}: array {job}, shards {len(rows)}, designs {sum(r['designs'] for r in rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("manifest", "config-root", "input", "profile", "output", "run-root"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("site", "revision", "checkpoint-sha256", "campaign"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--shard-size", type=int, default=5)
    prep.add_argument("--shard-sizes", type=Path, help="Optional task-to-size JSON, for measured slow tasks")
    sub = commands.add_parser("submit")
    sub.add_argument("manifest", type=Path)
    sub.add_argument("--concurrency", type=int, default=4)
    sub.add_argument("--runtime-estimates", type=Path, help="Observed seconds per design by task; combine shards within time budget")
    sub.add_argument("--runtime-budget-seconds", type=float, default=28800)
    sub.add_argument("--retry-qos-rejection", action="store_true")
    dis = commands.add_parser("dispatch")
    dis.add_argument("manifest", type=Path)
    dis.add_argument("index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "submit":
        submit(args)
    else:
        dispatch(args.manifest, args.index)


if __name__ == "__main__":
    main()
