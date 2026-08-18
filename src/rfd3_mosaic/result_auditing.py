"""Shared result-audit lifecycle for live and already completed RFD3 runs."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from rfd3_mosaic.assembly_compiler import (
    CompiledAudit,
    compile_audit_requirements,
)
from rfd3_mosaic.assembly_frontends import AuditRequirement
from rfd3_mosaic.rfd3_audit_gate import failed_audit_paths
from rfd3_mosaic.rfd3_mobility_audit import write_mobility_trajectory

CommandRunner = Callable[[list[str]], None]


@dataclass(frozen=True)
class ResultAuditOutcome:
    """Artifacts written by one topology-neutral post-inference audit pass."""

    reports: tuple[Path, ...]
    mobility_trajectory: Path | None


def run_command(command: list[str]) -> None:
    """Run one audit command with the same visible contract as the worker."""

    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def find_result_jsons(run_directory: str | Path) -> tuple[Path, ...]:
    """Return every independently sampled result metadata file in order."""

    root = Path(run_directory)
    candidates = tuple(sorted(root.glob("*model_0.json")))
    if not candidates:
        raise RuntimeError(
            "Expected at least one model_0 metadata JSON, observed "
            f"{[str(path) for path in candidates]}"
        )
    return candidates


def find_result_json(run_directory: str | Path) -> Path:
    """Require one result for compatibility with one-design callers."""

    candidates = find_result_jsons(run_directory)
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one model_0 metadata JSON, observed "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def find_compiled_input(run_directory: str | Path) -> Path:
    """Locate one frozen compiler input across current and legacy layouts."""

    root = Path(run_directory)
    candidates = [
        path
        for path in (
            root / "input" / "rfd3_input.json",
            root / "adapter" / "rfd3_input.json",
            root / "rfd3_input.json",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one frozen rfd3_input.json, observed "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0].resolve()


def symmetry_multiplicity(rfd3_input: str | Path) -> int:
    payload = json.loads(Path(rfd3_input).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Compiled RFD3 input must contain exactly one example")
    example = next(iter(payload.values()))
    return int(example["extra"]["symmetry_multiplicity"])


def run_result_audits(
    *,
    run_directory: str | Path,
    rfd3_input: str | Path,
    result_json: str | Path,
    semantic_audits: tuple[CompiledAudit, ...],
    output_directory: str | Path | None = None,
    python: str = sys.executable,
    command_runner: CommandRunner = run_command,
) -> ResultAuditOutcome:
    """Write every compiler-selected audit plus the universal scaffold audit."""

    root = Path(run_directory).resolve()
    input_path = Path(rfd3_input).resolve()
    if root != input_path and root not in input_path.parents:
        raise ValueError(
            "Post-hoc auditing requires the compiled input frozen inside "
            f"the run directory: {input_path}"
        )
    result_path = Path(result_json).resolve()
    report_root = (
        Path(output_directory).resolve()
        if output_directory is not None
        else root
    )
    if report_root != root and root not in report_root.parents:
        raise ValueError(
            "Result audit output must remain inside the frozen run directory: "
            f"{report_root}"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    trajectory = report_root / "mobility_trajectory.json"
    has_trajectory = write_mobility_trajectory(
        result_json=result_path,
        output=trajectory,
    )
    reports: list[Path] = []
    for audit in semantic_audits:
        report = report_root / audit.report_name
        command_runner(
            audit.command(
                python=python,
                result_json=result_path,
                output_report=report,
            )
        )
        if not report.is_file():
            raise RuntimeError(
                f"Audit command did not write its required report: {report}"
            )
        reports.append(report)

    scaffold_report = report_root / "scaffold_validity_audit.json"
    command_runner(
        [
            python,
            "-m",
            "rfd3_mosaic.rfd3_scaffold_audit",
            "--result-json",
            str(result_path),
            "--rfd3-input",
            str(input_path),
            "--output",
            str(scaffold_report),
            "--expected-symmetry-multiplicity",
            str(symmetry_multiplicity(input_path)),
            "--report-only",
        ]
    )
    if not scaffold_report.is_file():
        raise RuntimeError(
            "Scaffold audit command did not write its required report: "
            f"{scaffold_report}"
        )
    reports.append(scaffold_report)
    return ResultAuditOutcome(
        reports=tuple(reports),
        mobility_trajectory=trajectory if has_trajectory else None,
    )


def gate_result_audits(
    reports: tuple[Path, ...],
    *,
    python: str = sys.executable,
    command_runner: CommandRunner = run_command,
) -> None:
    """Apply the canonical fail-closed audit gate to a complete report set."""

    # Keep these compatibility keyword arguments while the worker and older
    # callers transition from the subprocess gate.  The actual decision uses
    # the same function as ``python -m rfd3_mosaic.rfd3_audit_gate``.
    del python, command_runner
    if not reports:
        raise ValueError("A result audit pass produced no required reports")
    failed = failed_audit_paths(list(reports))
    if failed:
        raise RuntimeError(
            "Required result audits failed: "
            + ", ".join(path.name for path in failed)
        )
    print(
        "Required result audits: PASSED ("
        + ", ".join(path.name for path in reports)
        + ")",
        flush=True,
    )


def _single_compiled_example(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Compiled RFD3 input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled RFD3 example must be a JSON object")
    return example


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _interface_seed_specification(
    *,
    compiled_directory: Path,
    config: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> Path:
    """Find the frozen legacy interface-seed specification, fail closed."""

    candidates = [
        compiled_directory / "assembly_specification.yaml",
    ]
    topology = config.get("topology") or {}
    for value in (
        topology.get("config") if isinstance(topology, Mapping) else None,
        extra.get("interface_seed_config"),
    ):
        if value:
            candidates.append(Path(str(value)).expanduser())
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = _load_yaml_mapping(candidate)
        if "interface_seed" in payload:
            return candidate.resolve()
    raise FileNotFoundError(
        "The frozen legacy interface-seed specification is unavailable; "
        "post-hoc auditing refuses to guess it"
    )


def infer_existing_run_audits(
    *,
    run_directory: str | Path,
    rfd3_input: str | Path,
    resolved_config: Mapping[str, Any],
) -> tuple[CompiledAudit, ...]:
    """Recover the audit plan from immutable run artifacts.

    The rules intentionally mirror ``lower_experiment_topology``.  No audit is
    inferred from whichever reports happen to exist, because an earlier crash
    may have occurred before writing the complete required set.
    """

    input_path = Path(rfd3_input).resolve()
    example = _single_compiled_example(input_path)
    extra = example.get("extra") or {}
    if not isinstance(extra, dict):
        raise ValueError("Compiled RFD3 extra metadata must be a mapping")
    topology = resolved_config.get("topology") or {}
    if not isinstance(topology, Mapping):
        raise ValueError("Resolved topology must be a mapping")
    kind = topology.get("kind")

    if kind == "interface_seed":
        requirements = [AuditRequirement.INTERFACE_GEOMETRY]
    elif kind in {"central_motif", "user_design"}:
        requirements = [AuditRequirement.EXACT_CONSTRAINT_ORBIT]
    else:
        raise ValueError(
            f"Unsupported or missing frozen topology kind {kind!r}"
        )

    if kind == "user_design":
        relations = extra.get("assembly_interface_relations") or []
        if not isinstance(relations, list):
            raise ValueError(
                "assembly_interface_relations must be a frozen list"
            )
        if any(not isinstance(relation, dict) for relation in relations):
            raise ValueError(
                "Every frozen assembly interface relation must be a mapping"
            )
        if relations:
            requirements.append(
                AuditRequirement.ASSEMBLY_INTERFACE_RELATIONS
            )
        if any(
            bool(relation.get("required", True))
            and relation.get("satisfaction_stage") == "output"
            and (relation.get("target_geometry") or {}).get("mode")
            == "geometric_constraints"
            for relation in relations
        ):
            requirements.append(AuditRequirement.GRAPH_INTERFACE_GUIDANCE)
        orbits = extra.get("motif_constraint_orbits") or []
        if not isinstance(orbits, list):
            raise ValueError("motif_constraint_orbits must be a frozen list")
        if any(not isinstance(orbit, dict) for orbit in orbits):
            raise ValueError(
                "Every frozen motif constraint orbit must be a mapping"
            )
        if any(
            orbit.get("mobility_mode") == "orbit_rigid"
            for orbit in orbits
        ):
            requirements.append(
                AuditRequirement.BOUNDED_COMPONENT_MOBILITY
            )

    mapping = input_path.parent / "mapping.json"
    specification: Path | None = None
    if kind == "interface_seed":
        if not mapping.is_file():
            raise FileNotFoundError(
                f"Frozen adapter mapping is missing: {mapping}"
            )
        specification = _interface_seed_specification(
            compiled_directory=input_path.parent,
            config=resolved_config,
            extra=extra,
        )
    return compile_audit_requirements(
        tuple(requirements),
        compiled_input=input_path,
        adapter_mapping=mapping if kind == "interface_seed" else None,
        specification=specification,
        base_directory=resolved_config.get("project_directory"),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ResultAuditOutcome",
    "find_compiled_input",
    "find_result_json",
    "find_result_jsons",
    "gate_result_audits",
    "infer_existing_run_audits",
    "run_result_audits",
    "symmetry_multiplicity",
    "utc_now",
]
