"""Run discovery, status aggregation and self-contained HTML reporting."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.run_index import read_run_record, valid_run_id
from rfd3_mosaic.run_layout import dated_run_directory

SCHEMA_VERSION = 1
AUDIT_GLOBS = (
    "rfd3_prevalidation.json",
    "*_audit.json",
)


@dataclass(frozen=True)
class RunReference:
    """Resolved connection between a scheduler job and its run artifacts."""

    job_id: str | None
    run_directory: Path | None
    submission_directory: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping in {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _run_directory_from_submission(directory: Path, job_id: str) -> Path:
    receipt_path = directory / "submission.json"
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        expected = receipt.get("expected_run_directory")
        if expected:
            return Path(str(expected)).expanduser().resolve()
    resolved = _load_yaml(directory / "resolved_config.yaml")
    output = resolved.get("output") or {}
    root = Path(str(output["root"])).expanduser()
    run_date = output.get("run_date")
    if run_date:
        return dated_run_directory(
            root,
            run_day=str(run_date),
            experiment=str(resolved["name"]),
            job_id=job_id,
        )
    return (root / str(output["campaign"]) / str(resolved["name"]) / job_id).resolve()


def _reference_from_submission(directory: Path) -> RunReference:
    receipt = _load_json(directory / "submission.json")
    job_id = str(receipt.get("job_id") or "").strip()
    if not job_id:
        raise ValueError(f"Submission receipt has no job_id: {directory}")
    run_directory = _run_directory_from_submission(directory, job_id)
    return RunReference(
        job_id=job_id,
        run_directory=run_directory if run_directory.is_dir() else None,
        submission_directory=directory.resolve(),
    )


def _candidate_root(root: str | Path | None) -> Path:
    value = root or os.environ.get("RFD3_MOSAIC_RUN_ROOT")
    if value is None:
        raise ValueError("A run ID requires --root or RFD3_MOSAIC_RUN_ROOT")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {path}")
    return path


def resolve_run_reference(
    target: str | Path,
    *,
    root: str | Path | None = None,
) -> RunReference:
    """Resolve a run directory, submission receipt, or executor run ID."""

    raw = str(target)
    path = Path(raw).expanduser()
    if path.exists():
        path = path.resolve()
        if path.is_file():
            if path.name != "submission.json":
                raise ValueError(
                    "A file run reference must be a submission.json receipt"
                )
            return _reference_from_submission(path.parent)
        if (path / "submission.json").is_file():
            return _reference_from_submission(path)
        job_id = path.name if valid_run_id(path.name) else None
        if (path / "experiment_summary.json").is_file() or job_id:
            return RunReference(job_id=job_id, run_directory=path)
        raise ValueError(
            "Directory is neither a run nor submission directory: " f"{path}"
        )

    if not valid_run_id(raw):
        raise FileNotFoundError(f"Run reference does not exist: {path}")
    search_root = _candidate_root(root)
    indexed = read_run_record(search_root, raw)
    if indexed is not None:
        indexed_run = Path(str(indexed["run_directory"])).resolve()
        submission_value = indexed.get("submission_directory")
        indexed_submission = (
            Path(str(submission_value)).resolve() if submission_value else None
        )
        return RunReference(
            job_id=raw,
            run_directory=indexed_run if indexed_run.is_dir() else None,
            submission_directory=indexed_submission,
        )
    run_candidates = sorted(
        candidate.resolve()
        for candidate in search_root.rglob(raw)
        if candidate.is_dir()
        and candidate.name == raw
        and "software" not in candidate.parts
    )
    if len(run_candidates) > 1:
        raise ValueError(
            f"JobID {raw} matched multiple run directories: "
            + ", ".join(str(path) for path in run_candidates)
        )
    if run_candidates:
        return RunReference(job_id=raw, run_directory=run_candidates[0])

    submission_candidates = []
    for receipt_path in search_root.rglob("submission.json"):
        try:
            receipt = _load_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(receipt.get("job_id")) == raw:
            submission_candidates.append(receipt_path.parent.resolve())
    if len(submission_candidates) > 1:
        raise ValueError(
            f"JobID {raw} matched multiple submission receipts: "
            + ", ".join(str(path) for path in submission_candidates)
        )
    if submission_candidates:
        return _reference_from_submission(submission_candidates[0])
    raise FileNotFoundError(
        f"No run directory or submission receipt for JobID {raw} under "
        f"{search_root}"
    )


def _run_scheduler(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def query_scheduler(job_id: str | None) -> dict[str, Any] | None:
    """Read Slurm without making scheduler state a reporting dependency."""

    if not job_id or job_id.startswith("local-"):
        return None
    output = _run_scheduler(
        [
            "sacct",
            "-j",
            job_id,
            "--starttime",
            "1970-01-01",
            "--format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,NodeList",
            "-n",
            "-P",
        ]
    )
    if output:
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) >= 7 and fields[0] == job_id:
                return {
                    "source": "sacct",
                    "job_id": fields[0],
                    "job_name": fields[1],
                    "partition": fields[2],
                    "state": fields[3].split()[0],
                    "exit_code": fields[4],
                    "elapsed": fields[5],
                    "node_list": fields[6],
                }
    output = _run_scheduler(
        [
            "squeue",
            "-h",
            "-j",
            job_id,
            "-o",
            "%i|%j|%P|%T|%M|%N|%R",
        ]
    )
    if output:
        fields = output.splitlines()[0].split("|")
        if len(fields) >= 7:
            return {
                "source": "squeue",
                "job_id": fields[0],
                "job_name": fields[1],
                "partition": fields[2],
                "state": fields[3].upper(),
                "exit_code": None,
                "elapsed": fields[4],
                "node_list": fields[5],
                "reason": fields[6],
            }
    return None


def _audit_paths(run_directory: Path, worker: dict[str, Any]) -> list[Path]:
    paths: dict[Path, None] = {}
    for value in worker.get("reports") or []:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = run_directory / path
        elif not path.is_file():
            # Worker summaries intentionally record absolute provenance paths.
            # Resolve a unique same-named artifact after a run has been copied
            # or physically reorganized without mutating its frozen summary.
            matches = [
                candidate
                for candidate in run_directory.rglob(path.name)
                if candidate.is_file() and "software" not in candidate.parts
            ]
            if len(matches) == 1:
                path = matches[0]
        paths[path.resolve()] = None
    for pattern in AUDIT_GLOBS:
        for path in run_directory.rglob(pattern):
            if "software" not in path.parts:
                paths[path.resolve()] = None
    return sorted(paths)


def _audit_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "exists": path.is_file(),
        "passed": False,
    }
    if not path.is_file():
        record["error"] = "declared report is missing"
        return record
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record["error"] = str(error)
        return record
    passed = payload.get("passed")
    if passed is None:
        passed = payload.get("status") == "passed"
    record.update(
        {
            "passed": bool(passed),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "assembly_shape_contract": payload.get("assembly_shape_contract"),
        }
    )
    return record


def _logical_state(
    worker: dict[str, Any],
    scheduler: dict[str, Any] | None,
) -> str:
    worker_state = str(worker.get("status") or "").lower()
    if worker_state in {"completed", "failed"}:
        return worker_state
    scheduler_state = str((scheduler or {}).get("state") or "").upper()
    if scheduler_state in {"PENDING", "CONFIGURING", "SUSPENDED"}:
        return "queued"
    if scheduler_state in {"RUNNING", "COMPLETING"}:
        return "running"
    if scheduler_state == "COMPLETED":
        return "completed_without_worker_summary"
    if scheduler_state:
        return "failed"
    if worker_state == "running":
        return "running"
    return "submitted" if scheduler is None else "unknown"


def _run_design_context(run_directory: Path | None) -> dict[str, Any] | None:
    """Recover the human-facing design identity from frozen run artifacts.

    A run name is not a sufficient provenance explanation.  In particular,
    engineering canaries may use repository reference inputs that are not the
    user's current biological target.  Prefer the frozen compiled input and
    provenance copied with the run so reports remain useful off-cluster.
    """

    if run_directory is None or not run_directory.is_dir():
        return None
    context: dict[str, Any] = {}
    resolved_path = run_directory / "resolved_config.yaml"
    if resolved_path.is_file():
        try:
            resolved = _load_yaml(resolved_path)
        except (OSError, ValueError, yaml.YAMLError):
            resolved = {}
        topology = resolved.get("topology") or {}
        sampling = resolved.get("sampling") or {}
        resources = resolved.get("resources") or {}
        context.update(
            {
                "topology_kind": topology.get("kind"),
                "authoring_config": topology.get("config"),
                "timesteps": sampling.get("timesteps"),
                "requested_designs": sampling.get("designs"),
                "seed": sampling.get("seed"),
                "profile": resources.get("profile_name"),
            }
        )

    compiled_path = run_directory / "input" / "rfd3_input.json"
    if compiled_path.is_file():
        try:
            payload = _load_json(compiled_path)
            example = next(iter(payload.values())) if len(payload) == 1 else {}
        except (OSError, ValueError, json.JSONDecodeError):
            example = {}
        if isinstance(example, dict):
            source = example.get("input")
            if source is not None:
                source_path = Path(str(source)).expanduser()
                if not source_path.is_absolute():
                    source_path = (compiled_path.parent / source_path).resolve()
                # This is the materialized structure consumed by RFD3, not
                # necessarily the structure supplied by the user.  Keep the
                # distinction explicit so resolver runs never obscure where
                # their biological seed came from.
                context["compiled_input"] = str(source_path)
            symmetry = example.get("symmetry") or {}
            extra = example.get("extra") or {}
            preferences = extra.get("resolved_design_preferences")
            if isinstance(preferences, dict):
                context["resolved_preferences"] = preferences
            assembly_shape = extra.get("assembly_shape")
            if isinstance(assembly_shape, dict):
                context["assembly_shape"] = assembly_shape
            if isinstance(symmetry, dict):
                context["symmetry"] = symmetry.get("id")
            relations = extra.get("assembly_interface_relations") or []
            stages = {
                str(relation.get("satisfaction_stage"))
                for relation in relations
                if isinstance(relation, dict)
            }
            if "output" in stages:
                task = "create_symmetric_interface"
            elif "input" in stages:
                task = "preserve_supplied_geometry"
            else:
                task = "fixed_geometry_scaffolding"
            context.update(
                {
                    "task": task,
                    "symmetry_multiplicity": extra.get("symmetry_multiplicity"),
                    "fixed_component_count": len(
                        extra.get("motif_constraint_orbits") or []
                    ),
                }
            )
            interface_instances: dict[str, int] = {}
            for relation in relations:
                if not isinstance(relation, dict):
                    continue
                interface_id = relation.get("source_interface_id")
                if interface_id is None:
                    continue
                label = str(interface_id)
                interface_instances[label] = interface_instances.get(label, 0) + 1
            if interface_instances:
                context["interface_instances"] = interface_instances

    manifest_path = run_directory / "input" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
        inputs = manifest.get("inputs") or []
        source_inputs = [
            {
                "path": str(item["path"]),
                "sha256": item.get("sha256"),
            }
            for item in inputs
            if isinstance(item, dict) and item.get("path")
        ]
        if source_inputs:
            context["source_inputs"] = source_inputs
            # Retain the singular field for API consumers when there is one
            # source while still supporting future multi-file assemblies.
            if len(source_inputs) == 1:
                context["structure_input"] = source_inputs[0]["path"]

    specification_path = run_directory / "input" / "assembly_specification.yaml"
    if specification_path.is_file():
        try:
            specification = _load_yaml(specification_path)
        except (OSError, ValueError, yaml.YAMLError):
            specification = {}
        assembly = specification.get("assembly") or {}
        fragments = assembly.get("fragments") or {}
        ports = assembly.get("ports") or {}
        interfaces = assembly.get("interfaces") or {}
        interface_selections: dict[str, list[str]] = {}
        for interface_id, interface in interfaces.items():
            if not isinstance(interface, dict):
                continue
            selections: list[str] = []
            for side in ("left_port", "right_port"):
                port = ports.get(interface.get(side)) or {}
                for fragment_id in port.get("fragments") or []:
                    fragment = fragments.get(fragment_id) or {}
                    selection = fragment.get("selection")
                    if selection is not None and str(selection) not in selections:
                        selections.append(str(selection))
            if selections:
                interface_selections[str(interface_id)] = selections
        if interface_selections:
            context["interface_selections"] = interface_selections

    provenance_path = run_directory / "provenance.json"
    if provenance_path.is_file():
        try:
            provenance = _load_json(provenance_path)
        except (OSError, ValueError, json.JSONDecodeError):
            provenance = {}
        public_source = provenance.get("public_user_design_source") or {}
        if isinstance(public_source, dict) and public_source.get("path"):
            context["resolved_design_source"] = public_source["path"]

    return {key: value for key, value in context.items() if value is not None}


def collect_run_status(
    reference: RunReference,
    *,
    include_scheduler: bool = True,
) -> dict[str, Any]:
    """Aggregate scheduler, worker, audit and artifact state fail-closed."""

    run_directory = reference.run_directory
    worker: dict[str, Any] = {}
    if run_directory is not None:
        summary_path = run_directory / "experiment_summary.json"
        if summary_path.is_file():
            worker = _load_json(summary_path)
    scheduler = query_scheduler(reference.job_id) if include_scheduler else None
    audits = (
        [_audit_record(path) for path in _audit_paths(run_directory, worker)]
        if run_directory is not None
        else []
    )
    state = _logical_state(worker, scheduler)
    declared_reports = list(worker.get("reports") or [])
    passed: bool | None
    if state in {"queued", "running", "submitted", "unknown"}:
        passed = None
    elif state == "completed":
        passed = bool(declared_reports) and all(audit["passed"] for audit in audits)
    else:
        passed = False

    structures: list[str] = []
    logs: list[str] = []
    if run_directory is not None and run_directory.is_dir():
        structures = [
            str(path.resolve())
            for pattern in ("*.cif", "*.cif.gz", "*.pdb", "*.pdb.gz")
            for path in sorted(run_directory.glob(pattern))
        ]
        logs = [
            str(path.resolve())
            for pattern in (
                "slurm-*.out",
                "slurm-*.err",
                "run-*.out",
                "run-*.err",
            )
            for path in sorted(run_directory.glob(pattern))
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": reference.job_id,
        "state": state,
        "passed": passed,
        "experiment": worker.get("experiment") or (scheduler or {}).get("job_name"),
        "run_directory": str(run_directory) if run_directory else None,
        "submission_directory": (
            str(reference.submission_directory)
            if reference.submission_directory
            else None
        ),
        "scheduler": scheduler,
        "design": _run_design_context(run_directory),
        "worker": worker or None,
        "audits": audits,
        "artifacts": {
            "structures": structures,
            "logs": logs,
            "mobility_trajectory": (
                worker.get("mobility_trajectory") if worker else None
            ),
        },
    }


def format_status_text(status: dict[str, Any]) -> str:
    """Render a concise terminal view from the canonical status payload."""

    worker = status.get("worker") or {}
    produced = worker.get("produced_designs")
    accepted = worker.get("accepted_designs")
    rejected = worker.get("rejected_designs")
    verdict = (
        "PASSED"
        if status["passed"] is True
        else "FAILED"
        if status["passed"] is False
        else "PENDING"
    )
    if status["state"] == "completed" and isinstance(produced, int):
        if accepted == produced:
            verdict = "PASSED"
        elif isinstance(accepted, int) and accepted > 0:
            verdict = "PARTIAL"
        else:
            verdict = "REJECTED"
    lines = [
        "RFD3-Mosaic run status",
        f"job:        {status['job_id'] or 'unknown'}",
        f"experiment: {status['experiment'] or 'unknown'}",
        f"state:      {status['state']}",
        f"verdict:    {verdict}",
        f"run:        {status['run_directory'] or 'not created yet'}",
    ]
    scheduler = status.get("scheduler") or {}
    if scheduler:
        lines.append(
            "scheduler:  "
            f"{scheduler.get('state', 'unknown')} "
            f"exit={scheduler.get('exit_code') or '-'} "
            f"elapsed={scheduler.get('elapsed') or '-'} "
            f"node={scheduler.get('node_list') or '-'}"
        )
    if isinstance(produced, int):
        lines.append(
            "designs:    "
            f"produced={produced} accepted={accepted} rejected={rejected}"
        )
    design = status.get("design") or {}
    if design:
        lines.extend(
            [
                f"task:       {design.get('task', 'unknown')}",
                f"input:      {design.get('structure_input', 'unknown')}",
                "symmetry:   "
                f"{design.get('symmetry', 'unknown')} "
                f"x{design.get('symmetry_multiplicity', 'unknown')}",
                "components: "
                f"{design.get('fixed_component_count', 'unknown')} fixed; "
                f"t={design.get('timesteps', 'unknown')} "
                f"seed={design.get('seed', 'unknown')} "
                f"designs={design.get('requested_designs', 1)}",
            ]
        )
        compiled_input = design.get("compiled_input")
        if compiled_input and compiled_input != design.get("structure_input"):
            lines.append(
                "RFD3 input:  "
                f"{compiled_input} "
                "(compiled pre-diffusion structure; not a generated design)"
            )
        preferences = design.get("resolved_preferences") or {}
        if preferences:
            lines.append(
                "preferences: "
                f"packing={preferences.get('packing', 'unknown')} "
                f"area={preferences.get('interface_area', 'unknown')} "
                f"cavity={preferences.get('cavity', 'unknown')} "
                f"diversity={preferences.get('diversity', 'unknown')} "
                f"motion={preferences.get('component_motion', 'unknown')}"
            )
        assembly_shape = design.get("assembly_shape") or {}
        if assembly_shape:
            requested = []
            for label, key in (
                ("outer", "diameter_angstrom"),
                ("cavity", "cavity_diameter_angstrom"),
            ):
                bounds = assembly_shape.get(key)
                if isinstance(bounds, dict):
                    requested.append(
                        f"{label}={bounds.get('minimum')}.."
                        f"{bounds.get('maximum')} A"
                    )
            if requested:
                lines.append("shape:       " + ", ".join(requested))
        source_inputs = design.get("source_inputs") or []
        if len(source_inputs) > 1:
            for index, item in enumerate(source_inputs, start=1):
                lines.append(f"source {index}:   {item.get('path', 'unknown')}")
        interface_instances = design.get("interface_instances") or {}
        if interface_instances:
            rendered = ", ".join(
                f"{interface_id} x{count}"
                for interface_id, count in sorted(interface_instances.items())
            )
            lines.append(f"interfaces: {rendered}")
        for interface_id, selections in sorted(
            (design.get("interface_selections") or {}).items()
        ):
            lines.append(
                f"  - {interface_id}: {' + '.join(str(x) for x in selections)}"
            )
    lines.append("audits:")
    if not status["audits"]:
        lines.append("  - none available")
    for audit in status["audits"]:
        label = "PASS" if audit["passed"] else "FAIL"
        lines.append(f"  - {label:<4} {audit['name']}")
        if audit["name"] == "graph_interface_guidance_audit.json":
            metrics = (audit.get("summary") or {}).get("final_packing_metrics") or {}
            if metrics:
                lines.append(
                    "         packing "
                    f"energy={metrics.get('energy', 'NA')} "
                    f"orientation={metrics.get('orientation', 'NA')} "
                    f"shape={metrics.get('shape', 'NA')} "
                    f"min_edge={metrics.get('minimum_edge_distance', 'NA')}"
                )
        if audit["name"] == "scaffold_core_guidance_audit.json":
            summary = audit.get("summary") or {}
            metrics = summary.get("final_metrics") or {}
            if metrics:
                lines.append(
                    "         monomer core "
                    f"normalized_rg={metrics.get('mean_normalized_rg', 'NA')} "
                    "tertiary_support="
                    f"{metrics.get('mean_tertiary_support_fraction', 'NA')} "
                    "long_range_deficit="
                    f"{metrics.get('long_range_contacts', 'NA')} "
                    "target_met="
                    f"{summary.get('scientific_quality_satisfied', 'NA')}"
                )
                if (
                    summary.get("scientific_quality_satisfied") is False
                    and not summary.get("quality_required", False)
                ):
                    lines.append(
                        "         advisory: scientific monomer-core targets "
                        "were not met; PASSED covers required geometry and "
                        "safety contracts only"
                    )
        if audit["name"] == "component_mobility_audit.json":
            summary = audit.get("summary") or {}
            components = summary.get("components") or []
            if components:
                maximum_translation_fraction = max(
                    float(item.get("translation_fraction_of_bound", 0.0))
                    for item in components
                )
                maximum_rotation_fraction = max(
                    float(item.get("rotation_fraction_of_bound", 0.0))
                    for item in components
                )
                lines.append(
                    "         mobility "
                    f"applied={summary.get('applied_proposal_count', 'NA')} "
                    f"moved={summary.get('nonzero_motion_observed', 'NA')} "
                    f"translation_bound_used={maximum_translation_fraction} "
                    f"rotation_bound_used={maximum_rotation_fraction}"
                )
        if audit["name"] == "assembly_interface_relation_audit.json":
            metrics = audit.get("summary") or {}
            if metrics.get("output_packing_edge_count", 0):
                lines.append(
                    "         output packing "
                    f"residue_pairs="
                    f"{metrics.get('minimum_reciprocal_contact_residue_pairs', 'NA')} "
                    f"density="
                    f"{metrics.get('minimum_reciprocal_contact_density', 'NA')} "
                    f"depth_sd="
                    f"{metrics.get('maximum_contact_depth_standard_deviation', 'NA')} "
                    f"void_proxy="
                    f"{metrics.get('maximum_local_contact_void_fraction_proxy', 'NA')}"
                )
        if audit["name"] == "scaffold_validity_audit.json":
            shape_contract = audit.get("assembly_shape_contract") or {}
            if shape_contract.get("declared"):
                for check in shape_contract.get("checks") or []:
                    lines.append(
                        "         shape "
                        f"{check.get('field')}="
                        f"{check.get('observed', 'NA')} A "
                        f"requested={check.get('requested_minimum')}.."
                        f"{check.get('requested_maximum')} A"
                    )
    structures = status["artifacts"]["structures"]
    lines.append(
        "raw outputs: "
        f"{len(structures)} "
        "(one per produced design; rejected designs are retained for audit)"
    )
    for path in structures:
        lines.append(f"  - {path}")
    if worker.get("error"):
        lines.append(
            f"failure:     {worker.get('error_type', 'Error')}: " f"{worker['error']}"
        )
    return "\n".join(lines)


def render_html_report(status: dict[str, Any]) -> str:
    """Create a dependency-free report that can be copied with the run."""

    worker = status.get("worker") or {}
    produced = worker.get("produced_designs")
    accepted = worker.get("accepted_designs")
    verdict = (
        "passed"
        if status["passed"] is True
        else "failed"
        if status["passed"] is False
        else "pending"
    )
    verdict_label = verdict.upper()
    if status["state"] == "completed" and isinstance(produced, int):
        if accepted == produced:
            verdict = "passed"
            verdict_label = "PASSED"
        elif isinstance(accepted, int) and accepted > 0:
            verdict = "pending"
            verdict_label = "PARTIAL"
        else:
            verdict = "failed"
            verdict_label = "REJECTED"
    audit_rows = []
    for audit in status["audits"]:
        summary = audit.get("summary")
        summary_text = (
            json.dumps(summary, sort_keys=True, indent=2) if summary is not None else ""
        )
        audit_rows.append(
            "<tr>"
            f"<td>{escape(audit['name'])}</td>"
            f"<td><span class='badge {'pass' if audit['passed'] else 'fail'}'>"
            f"{'PASS' if audit['passed'] else 'FAIL'}</span></td>"
            f"<td><details><summary>metrics</summary><pre>"
            f"{escape(summary_text)}</pre></details></td>"
            "</tr>"
        )
    structure_items = (
        "".join(
            f"<li><code>{escape(path)}</code></li>"
            for path in status["artifacts"]["structures"]
        )
        or "<li>None available</li>"
    )
    scheduler = status.get("scheduler") or {}
    design = status.get("design") or {}
    design_text = (
        json.dumps(design, sort_keys=True, indent=2)
        if design
        else "No frozen design provenance was available."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RFD3-Mosaic report — {escape(str(status['job_id'] or 'run'))}</title>
<style>
:root{{--bg:#0b1020;--card:#151d31;--text:#e8edf7;--muted:#9ba8bd;
--accent:#7dd3fc;--pass:#34d399;--fail:#fb7185;--pending:#fbbf24}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#080c18,#111a2e);
color:var(--text);font:15px/1.55 system-ui,sans-serif}} main{{max-width:1100px;margin:auto;padding:36px}}
h1{{font-size:34px;margin:0 0 6px}} h2{{margin-top:30px}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:24px 0}}
.card{{background:var(--card);border:1px solid #283653;border-radius:14px;padding:18px;box-shadow:0 12px 35px #0005}}
.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}
.value{{font-size:20px;font-weight:650;margin-top:5px;overflow-wrap:anywhere}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;font-weight:700;color:#08111c}}
.pass{{background:var(--pass)}} .fail{{background:var(--fail)}} .pending{{background:var(--pending)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:14px;overflow:hidden}}
th,td{{padding:13px;text-align:left;border-bottom:1px solid #283653;vertical-align:top}}
th{{color:var(--muted)}} code,pre{{font-family:ui-monospace,monospace;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;max-width:700px}} a{{color:var(--accent)}}
</style></head><body><main>
<div class="muted">RFD3-MOSAIC / RUN REPORT</div>
<h1>{escape(str(status['experiment'] or 'Unnamed experiment'))}</h1>
<div><span class="badge {verdict}">{verdict_label}</span></div>
<section class="grid">
<div class="card"><div class="label">Job ID</div><div class="value">{escape(str(status['job_id'] or 'unknown'))}</div></div>
<div class="card"><div class="label">Logical state</div><div class="value">{escape(status['state'])}</div></div>
<div class="card"><div class="label">Scheduler</div><div class="value">{escape(str(scheduler.get('state') or 'unavailable'))}</div></div>
<div class="card"><div class="label">Runtime</div><div class="value">{escape(str(scheduler.get('elapsed') or 'unknown'))}</div></div>
<div class="card"><div class="label">Design yield</div><div class="value">{escape(str(accepted if isinstance(produced, int) else 'unknown'))}/{escape(str(produced if isinstance(produced, int) else 'unknown'))}</div></div>
</section>
<h2>Required audits</h2>
<table><thead><tr><th>Report</th><th>Verdict</th><th>Summary</th></tr></thead>
<tbody>{''.join(audit_rows) or '<tr><td colspan="3">No audits available.</td></tr>'}</tbody></table>
<h2>Design provenance</h2>
<div class="card"><pre>{escape(design_text)}</pre></div>
<h2>Raw generated outputs</h2>
<p class="muted">One file per produced design. Audit-rejected outputs are
retained for diagnosis and are not accepted structures.</p>
<ul>{structure_items}</ul>
<h2>Execution details</h2>
<div class="card"><div class="label">Run directory</div><code>{escape(str(status['run_directory'] or 'not created'))}</code>
<div class="label" style="margin-top:14px">Failure</div><div>{escape(str(worker.get('error') or 'none'))}</div></div>
<p class="muted">Generated {escape(status['generated_at'])}; schema {status['schema_version']}.</p>
</main></body></html>"""


def write_report(
    status: dict[str, Any],
    output: str | Path | None = None,
) -> Path:
    """Write the canonical status JSON and a human-facing HTML report."""

    if output is None:
        run_directory = status.get("run_directory")
        if run_directory is None:
            raise ValueError(
                "A pending submission without a run directory requires --output"
            )
        output_path = Path(run_directory) / "mosaic_report.html"
    else:
        output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(status), encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
