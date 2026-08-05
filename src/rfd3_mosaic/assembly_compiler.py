"""Compile every topology frontend through one native Assembly IR path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rfd3_mosaic.assembly_frontends import (
    AuditRequirement,
    lower_experiment_topology,
)
from rfd3_mosaic.output import compile_assembly_rfd3_input


@dataclass(frozen=True)
class CompiledAudit:
    """One topology-neutral semantic audit emitted by the compiler.

    ``experiment_worker`` deliberately knows nothing about central motifs or
    interface seeds.  It only supplies the result JSON and output report to
    this frozen command description.  Keeping audit selection at compile time
    prevents topology-specific branches from leaking into execution.
    """

    module: str
    report_name: str
    input_arguments: tuple[tuple[str, str], ...]

    def command(
        self,
        *,
        python: str,
        result_json: str | Path,
        output_report: str | Path,
    ) -> list[str]:
        command = [python, "-m", self.module]
        for flag, value in self.input_arguments:
            command.extend([flag, value])
        command.extend(
            [
                "--result-json",
                str(result_json),
                "--output",
                str(output_report),
                "--report-only",
            ]
        )
        return command


@dataclass(frozen=True)
class CompiledAssembly:
    """Topology-neutral artifacts consumed by RFD3 execution."""

    input_path: Path
    example_id: str
    semantic_audits: tuple[CompiledAudit, ...]


def compile_experiment_assembly(
    topology: Mapping[str, Any],
    output_directory: str | Path,
    *,
    project_directory: str | Path,
    experiment_name: str,
) -> CompiledAssembly:
    """Lower a user frontend, then compile one AssemblySpecification.

    Frontend compatibility is deliberately separated from compilation.  The
    native compiler below is invoked exactly once regardless of whether the
    user described a central motif or a cross-subunit interface seed.
    """

    output = Path(output_directory)
    project = Path(project_directory)
    request = lower_experiment_topology(
        topology,
        output,
        project_directory=project,
        experiment_name=experiment_name,
    )
    artifacts = compile_assembly_rfd3_input(
        request.specification_path,
        output,
        base_directory=project,
        example_id=request.example_id,
        pose_seed=request.pose_seed,
        pose_candidate_manifest=request.pose_candidate_manifest,
        linker_length=request.linker_length,
        extra_metadata=request.audit_metadata,
    )

    audits: list[CompiledAudit] = []
    for requirement in request.audit_requirements:
        if requirement == AuditRequirement.EXACT_CONSTRAINT_ORBIT:
            audits.append(
                CompiledAudit(
                    module="rfd3_mosaic.rfd3_constraint_orbit_audit",
                    report_name="constraint_orbit_audit.json",
                    input_arguments=(
                        ("--compiled-input", str(artifacts.input_path)),
                    ),
                )
            )
        elif requirement == AuditRequirement.INTERFACE_GEOMETRY:
            audits.append(
                CompiledAudit(
                    module="rfd3_mosaic.rfd3_seed_audit",
                    report_name="seed_integrity_audit.json",
                    input_arguments=(
                        ("--adapter-mapping", str(artifacts.mapping_path)),
                        ("--config", str(request.specification_path)),
                        ("--base-directory", str(project)),
                    ),
                )
            )
        elif requirement == AuditRequirement.BOUNDED_COMPONENT_MOBILITY:
            audits.append(
                CompiledAudit(
                    module="rfd3_mosaic.rfd3_mobility_audit",
                    report_name="component_mobility_audit.json",
                    input_arguments=(
                        ("--compiled-input", str(artifacts.input_path)),
                    ),
                )
            )
        else:
            raise ValueError(
                f"Unsupported audit requirement {requirement!r}"
            )
    return CompiledAssembly(
        input_path=artifacts.input_path,
        example_id=request.example_id,
        semantic_audits=tuple(audits),
    )


__all__ = [
    "CompiledAudit",
    "CompiledAssembly",
    "compile_experiment_assembly",
]
