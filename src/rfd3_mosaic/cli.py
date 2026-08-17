"""Unified command-line interface for RFD3-Mosaic experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import yaml

from rfd3_mosaic.assembly_compiler import compile_experiment_assembly
from rfd3_mosaic.capabilities import (
    capability_manifest,
    required_capabilities_for_design,
)
from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.design_compiler import (
    bind_constraint_plan,
    lower_user_design,
)
from rfd3_mosaic.design_preferences import compile_design_preferences
from rfd3_mosaic.experiment import (
    build_execution_plan,
    render_submission,
    resolve_experiment,
)
from rfd3_mosaic.execution import executor_for_id
from rfd3_mosaic.graph_search import search_graph_design
from rfd3_mosaic.posthoc_audit import audit_existing_run
from rfd3_mosaic.rfd3_prevalidate import prevalidate_rfd3_input
from rfd3_mosaic.run_index import (
    list_run_records,
    rebuild_run_index,
    record_submission,
)
from rfd3_mosaic.run_reporting import (
    collect_run_status,
    format_status_text,
    resolve_run_reference,
    write_report,
)
from rfd3_mosaic.seed_library import materialize_seed_library
from rfd3_mosaic.schema import (
    BetweenGeneration,
    SimpleCageIntentSpec,
    UserDesignSpec,
    UserDesignTask,
    load_simple_cage_intent,
    load_user_design,
)
from rfd3_mosaic.sampling_plan import compile_sampling_plan
from rfd3_mosaic.simple_architecture import analyze_simple_architectures
from rfd3_mosaic.simple_resolver import (
    enumerate_simple_design_candidates,
    resolve_simple_intent,
)
from rfd3_mosaic.structure_inspection import (
    inspect_declared_interface_relation,
    inspect_structure_interfaces,
    write_structure_inspection,
)
from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)


def _add_quick_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", default="p100")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=("exact_mosaic", "official_rfd3"),
        default="exact_mosaic",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render, but do not call sbatch.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfd3-mosaic")
    commands = parser.add_subparsers(dest="command", required=True)

    capabilities = commands.add_parser(
        "capabilities",
        help="Show implemented features and their validation maturity.",
    )
    capabilities.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    inspect = commands.add_parser(
        "inspect",
        help=(
            "Analyze an input PDB/mmCIF, detect candidate interface seeds "
            "and write a short ordinary-user cage intent."
        ),
    )
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--output-dir", required=True, type=Path)
    inspect.add_argument("--name")
    inspect.add_argument(
        "--architecture",
        choices=("auto", "ring", "cage"),
        default="auto",
    )
    inspect.add_argument(
        "--composition",
        choices=("auto", "homomer", "heteromer"),
        default="auto",
    )
    inspect.add_argument(
        "--symmetry",
        dest="inspect_symmetries",
        action="append",
        help="Allowed symmetry; repeat it, or omit for automatic discovery.",
    )
    inspect.add_argument("--contact-cutoff", type=float, default=4.5)
    inspect.add_argument("--minimum-atom-contacts", type=int, default=4)
    inspect.add_argument(
        "--minimum-contact-residues-per-side",
        type=int,
        default=2,
    )
    inspect.add_argument("--length-min", type=int, default=40)
    inspect.add_argument("--length-max", type=int, default=100)
    inspect.add_argument("--subunits-min", type=int)
    inspect.add_argument("--subunits-max", type=int)
    inspect.add_argument("--diameter-min", type=float)
    inspect.add_argument("--diameter-max", type=float)
    inspect.add_argument("--cavity-diameter-min", type=float)
    inspect.add_argument("--cavity-diameter-max", type=float)
    inspect.add_argument("--profile", default="p100")

    resolve = commands.add_parser(
        "resolve",
        help=(
            "Enumerate, rank and freeze an ordinary cage intent as normal "
            "replayable public design YAML candidates."
        ),
    )
    resolve.add_argument("config", type=Path)
    resolve.add_argument("--output-dir", required=True, type=Path)
    resolve.add_argument(
        "--symmetry",
        dest="resolve_symmetries",
        action="append",
        help="Candidate Cn symmetry; repeat, or use the intent choices.",
    )
    resolve.add_argument(
        "--pose-samples",
        type=int,
        default=None,
        help=(
            "Explicit number of global seed-pose starts. When omitted, "
            "the ordinary diversity preference selects 4/8/16 starts."
        ),
    )
    resolve.add_argument(
        "--optimize-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run deterministic CPU rigid-component pose optimization for "
            "multi-seed intents before freezing candidates (default: on)."
        ),
    )
    resolve.add_argument(
        "--pose-optimize-top",
        type=int,
        default=4,
        help="Number of initial topology candidates optimized on CPU.",
    )
    resolve.add_argument(
        "--pose-optimization-levels",
        type=int,
        default=3,
        help="Number of coarse-to-fine deterministic pattern-search levels.",
    )
    resolve.add_argument(
        "--pose-maximum-translation",
        type=float,
        default=12.0,
        help="Maximum per-component displacement from its input pose (A).",
    )
    resolve.add_argument(
        "--pose-maximum-rotation-deg",
        type=float,
        default=25.0,
        help="Maximum per-axis rotation from the input pose (degrees).",
    )
    resolve.add_argument("--seed-start", type=int, default=0)
    resolve.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="RFD3 steps frozen into every resolved candidate (2-200).",
    )
    resolve.add_argument("--top", type=int, default=20)
    resolve.add_argument("--max-candidates", type=int, default=4096)
    resolve.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    validate = commands.add_parser(
        "validate",
        help=(
            "Validate, lower and preflight an experiment without writing "
            "persistent files."
        ),
    )
    validate.add_argument("config", type=Path)
    validate.add_argument("--profile")

    plan = commands.add_parser(
        "plan",
        help="Resolve and display the execution plan without writing files.",
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--profile")
    plan.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    search = commands.add_parser(
        "search",
        help=(
            "Enumerate assembly-graph symmetry neighbours and initial poses, "
            "then rank complete static candidates before RFD3."
        ),
    )
    search.add_argument("config", type=Path)
    search.add_argument("--output-dir", required=True, type=Path)
    search.add_argument(
        "--symmetry",
        dest="search_symmetries",
        action="append",
        help=(
            "Candidate symmetry ID; repeat to search several groups. "
            "Defaults to the symmetry declared in the design."
        ),
    )
    search.add_argument(
        "--interface",
        dest="interfaces",
        action="append",
        help="Interface ID to search; repeat it, or omit to search all edges.",
    )
    search.add_argument("--pose-samples", type=int, default=1)
    search.add_argument("--seed-start", type=int, default=0)
    search.add_argument("--top", type=int, default=20)
    search.add_argument("--max-candidates", type=int, default=4096)
    search.add_argument("--include-identity", action="store_true")
    search.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    render = commands.add_parser(
        "render",
        help="Freeze the config and render a short Slurm job.",
    )
    render.add_argument("config", type=Path)
    render.add_argument("--profile")
    render.add_argument("--output-dir", type=Path)

    for command, help_text in (
        (
            "run",
            "Validate, freeze and run one design through the configured executor.",
        ),
        (
            "submit",
            "Compatibility alias for 'run'.",
        ),
    ):
        submit = commands.add_parser(command, help=help_text)
        submit.add_argument("config", type=Path)
        submit.add_argument("--profile")
        submit.add_argument("--output-dir", type=Path)
        submit.add_argument(
            "--dry-run",
            action="store_true",
            help="Render but do not call sbatch.",
        )

    status = commands.add_parser(
        "status",
        help="Find one run and summarize scheduler, worker and audit state.",
    )
    status.add_argument("target", help="Run directory, receipt, or Slurm JobID.")
    status.add_argument(
        "--root",
        type=Path,
        help="Search root for a numeric JobID (or set RFD3_MOSAIC_RUN_ROOT).",
    )
    status.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )
    status.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Read artifacts only; do not query sacct/squeue.",
    )

    report = commands.add_parser(
        "report",
        help="Generate a self-contained HTML and JSON report for one run.",
    )
    report.add_argument("target", help="Run directory, receipt, or Slurm JobID.")
    report.add_argument("--root", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--no-scheduler", action="store_true")

    audit = commands.add_parser(
        "audit",
        help=(
            "Rerun all applicable result audits for an existing output "
            "without rerunning RFD3."
        ),
    )
    audit.add_argument("target", help="Run directory, receipt, or Slurm JobID.")
    audit.add_argument(
        "--root",
        type=Path,
        help="Search root for a numeric JobID (or set RFD3_MOSAIC_RUN_ROOT).",
    )
    audit.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Refresh the report without querying sacct/squeue.",
    )

    runs = commands.add_parser(
        "runs",
        help="List jobs recorded in one persistent RFD3-Mosaic run index.",
    )
    runs.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Top-level output root containing .rfd3-mosaic/jobs.",
    )
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument(
        "--rebuild",
        action="store_true",
        help="Import historical experiment_summary.json files before listing.",
    )
    runs.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )

    central = commands.add_parser(
        "central",
        help="Generate around one fixed central motif and submit.",
    )
    central.add_argument("--input", required=True, type=Path)
    central.add_argument("--motif", required=True)
    central.add_argument("--n-length", type=int, default=35)
    central.add_argument("--c-length", type=int, default=35)
    central.add_argument("--campaign", default="central-motif")
    _add_quick_runtime_arguments(central)

    interface = commands.add_parser(
        "interface",
        help="Generate between a fixed interface seed and submit.",
    )
    interface.add_argument("--config", required=True, type=Path)
    pose = interface.add_mutually_exclusive_group(required=True)
    pose.add_argument("--manifest", type=Path)
    pose.add_argument("--pose-seed", type=int)
    interface.add_argument("--length", type=int)
    interface.add_argument("--campaign", default="interface-seed")
    _add_quick_runtime_arguments(interface)
    return parser


def _safe_default_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "rfd3-mosaic")[:64]


def _write_quick_experiment(arguments: argparse.Namespace) -> Path:
    output_root = arguments.output.expanduser().resolve()
    campaign = _safe_default_name(arguments.campaign)
    if campaign != arguments.campaign:
        raise ValueError(
            "--campaign must contain only letters, numbers, '.', '_' or '-'"
        )
    if arguments.command == "central":
        default_name = (
            f"central-n{arguments.n_length}-c{arguments.c_length}"
            f"-s{arguments.seed}"
        )
        topology = {
            "kind": "central_motif",
            "template_input": str(arguments.input.expanduser().resolve()),
            "fixed_selector": arguments.motif,
            "n_terminal_length": arguments.n_length,
            "c_terminal_length": arguments.c_length,
        }
    else:
        default_name = f"interface-t{arguments.steps}-s{arguments.seed}"
        topology = {
            "kind": "interface_seed",
            "config": str(arguments.config.expanduser().resolve()),
            "pose_candidate_manifest": (
                str(arguments.manifest.expanduser().resolve())
                if arguments.manifest is not None
                else None
            ),
            "pose_seed": arguments.pose_seed,
            "linker_length": arguments.length,
        }
    name = _safe_default_name(arguments.name or default_name)
    payload = {
        "schema_version": 1,
        "name": name,
        "topology": topology,
        "sampling": {
            "preset": arguments.preset,
            "timesteps": arguments.steps,
            "seed": arguments.seed,
        },
        "resources": {"profile": arguments.profile},
        "output": {
            "root": str(output_root),
            "campaign": campaign,
        },
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    request_directory = (
        output_root
        / campaign
        / "_requests"
        / name
        / timestamp
    )
    request_directory.mkdir(parents=True, exist_ok=False)
    path = request_directory / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_public_experiment(
    design: UserDesignSpec,
    source_path: Path,
) -> Path:
    """Materialize the internal experiment envelope for one public design."""

    # Prove that the public declaration lowers and that its complete expanded
    # assembly satisfies the strict static geometry gates before creating a
    # request directory or consuming a scheduler slot.
    _preflight_public_design_geometry(design)
    if design.output is None:
        raise ValueError(
            "Public design render/submit requires output.root"
        )
    resources = design.resources.model_dump(
        mode="json",
        exclude_none=True,
    )
    # Initial-pose sampling is compiled from the public design into the
    # AssemblySpecification.  The internal experiment envelope contains only
    # diffusion-loop settings.
    sampling = design.sampling.model_dump(
        mode="json",
        exclude={"initial_pose", "initial_poses"},
    )
    payload = {
        "schema_version": 1,
        "name": design.name,
        "topology": {
            "kind": "user_design",
            "config": str(source_path.expanduser().resolve()),
            "example_id": design.name,
        },
        "sampling": sampling,
        "resources": resources,
        "output": design.output.model_dump(mode="json"),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    request_directory = (
        design.output.root
        / design.output.campaign
        / "_requests"
        / design.name
        / timestamp
    )
    request_directory.mkdir(parents=True, exist_ok=False)
    path = request_directory / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class _PublicGeometryPreflight:
    atom_count: int
    residue_count: int
    chain_count: int
    runtime_features_validated: bool


def _preflight_public_design_geometry(
    design: UserDesignSpec,
) -> _PublicGeometryPreflight:
    """Strictly compile one public design without leaving persistent files.

    Schema and selector validation alone cannot expose clashes introduced by
    symmetry expansion or initial-pose sampling.  The standalone compiler is
    the canonical implementation of those geometric checks, so public
    validate/render/run/submit commands all use it before scheduling RFD3.
    """

    with tempfile.TemporaryDirectory(
        prefix="rfd3-mosaic-preflight-",
    ) as temporary_directory:
        root = Path(temporary_directory)
        design_path = root / "public_user_design.yaml"
        design_path.write_text(
            yaml.safe_dump(
                design.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        assembly = compile_experiment_assembly(
            {
                "kind": "user_design",
                "config": str(design_path),
                "example_id": design.name,
            },
            root / "compiled",
            project_directory=design.input.parent,
            experiment_name=design.name,
        )
        report = prevalidate_rfd3_input(
            assembly.input_path,
            example_id=assembly.example_id,
            report_path=root / "rfd3_prevalidation.json",
        )
        return _PublicGeometryPreflight(
            atom_count=int(report["atom_count"]),
            residue_count=sum(
                int(value)
                for value in report["residues_per_chain"].values()
            ),
            chain_count=int(report["chain_count"]),
            runtime_features_validated=True,
        )


def _print_execution_plan(plan: dict, *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    design = plan["design"]
    sampling = plan["sampling"]
    execution = plan["execution"]
    software = plan["software"]
    print("RFD3-Mosaic execution plan")
    print(f"name:       {plan['name']}")
    print(f"topology:   {design['topology']}")
    print(f"timesteps:  {sampling['timesteps']}")
    print(f"seed:       {sampling['seed']}")
    print(f"preset:     {sampling['preset']}")
    print(f"backend:    {sampling['execution_backend']}")
    print(f"profile:    {execution['profile']}")
    print(f"executor:   {execution['executor']}")
    print(f"partitions: {execution['slurm']['partition']}")
    print(f"run root:   {plan['output']['run_root']}")
    print("effective constraints:")
    for constraint in design["effective_constraints"]:
        print(
            "  - "
            + constraint["operator"]
            + f" [{constraint['orbit_scope']}]"
            + f" selector={constraint['selector']}"
        )
    print(f"Mosaic commit: {software['commit']}")
    print(f"Foundry base:  {software['foundry_base_commit']}")
    print(f"tracked dirty: {software['tracked_dirty']}")


def _load_public_design_if_present(path: Path) -> UserDesignSpec | None:
    """Recognize the new public schema without misreading legacy configs."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if "topology" in payload:
        return None
    if {"input", "symmetry"}.issubset(payload):
        return load_user_design(source)
    return None


def _load_simple_intent_if_present(
    path: Path,
) -> SimpleCageIntentSpec | None:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") != "simple_cage_intent":
        return None
    return load_simple_cage_intent(source)


def _simple_intent_analysis(
    intent: SimpleCageIntentSpec,
) -> dict[str, object]:
    seed_sources = {
        (seed.source or intent.input).expanduser().resolve()
        for seed in intent.interface_seeds.values()
    }
    independent_seed_layout = (
        (
            intent.seed_layout == "solve"
            and len(intent.interface_seeds) > 1
        )
        or (
            intent.seed_layout == "auto"
            and any(
                seed.source is not None
                for seed in intent.interface_seeds.values()
            )
            and len(seed_sources) > 1
        )
    )
    if independent_seed_layout:
        with tempfile.TemporaryDirectory(
            prefix="rfd3-mosaic-seed-library-plan-"
        ) as raw:
            materialized = materialize_seed_library(intent, Path(raw))
            # The canonicalized seed library is already in arbitrary local
            # frames.  Prevent recursive materialization while analyzing the
            # ordinary architecture contract.
            analysis_intent = materialized.intent.model_copy(
                update={"seed_layout": "preserve_input"}
            )
            payload = _simple_intent_analysis(analysis_intent)
        payload["input"] = "independent interface-seed library"
        payload["seed_layout"] = {
            "requested": intent.seed_layout,
            "relative_pose": "solve",
        }
        payload["seed_library"] = {
            "mode": materialized.manifest["mode"],
            "seed_count": materialized.manifest["seed_count"],
            "sources": {
                seed_id: record["source"]
                for seed_id, record in materialized.manifest["seeds"].items()
            },
            "file_frames_are_assembly_coordinates": False,
            "requested_seed_layout": intent.seed_layout,
        }
        payload["architecture_resolution"] = (
            "global_seed_pose_and_topology_resolution_ready"
        )
        payload["ordinary_resolver"]["supported_contract"] = (
            "user-supplied preserve_exact seed interfaces; physical "
            "multiplicity filtering; canonical local frames; global finite-"
            "group pose starts; joint bounded SE(3) refinement; strict replay"
        )
        return payload
    inspection = inspect_structure_interfaces(
        intent.input,
        contact_cutoff=intent.inspection.contact_cutoff,
        minimum_atom_contacts=intent.inspection.minimum_atom_contacts,
        minimum_contact_residues_per_side=(
            intent.inspection.minimum_contact_residues_per_side
        ),
    )
    declared_evidence = []
    for seed_id, seed in intent.interface_seeds.items():
        evidence = inspect_declared_interface_relation(
            intent.input,
            interface_id=seed_id,
            participants=seed.participants,
            selectors=seed.selectors,
            contact_cutoff=intent.inspection.contact_cutoff,
            minimum_atom_contacts=(
                intent.inspection.minimum_atom_contacts
            ),
            minimum_contact_residues_per_side=(
                intent.inspection.minimum_contact_residues_per_side
            ),
        )
        if not evidence.contact_graph_connected:
            raise ValueError(
                f"Declared interface seed {seed_id!r} does not form a "
                "connected participant contact graph under its frozen "
                "input-contact thresholds"
            )
        declared_evidence.append(
            {
                "id": seed_id,
                "participants": list(evidence.participant_chains),
                "active_contact_pairs": [
                    list(pair) for pair in evidence.active_contact_pairs
                ],
                "contact_graph_connected": True,
            }
        )
    symmetry = (
        "auto"
        if intent.goal.symmetry == "auto"
        else list(intent.goal.symmetry)
    )
    hypotheses = analyze_simple_architectures(intent)
    accepted_hypotheses = [item for item in hypotheses if item.accepted]
    component_incidence_hypotheses: list[dict[str, object]] = []
    if len(intent.interface_seeds) == 1:
        interface_id, interface_seed = next(
            iter(intent.interface_seeds.items())
        )
        if len(interface_seed.participants) == 2:
            left_participant, right_participant = (
                interface_seed.participants
            )
            for hypothesis in accepted_hypotheses:
                if hypothesis.symmetry not in {"T", "O", "I"}:
                    continue
                physical_count = hypothesis.interface_physical_instances[
                    interface_id
                ]
                try:
                    incidence_plans = (
                        enumerate_binary_interface_incidence_plans(
                            symmetry=hypothesis.symmetry,
                            interface_id=interface_id,
                            left_participant=left_participant,
                            right_participant=right_participant,
                            physical_interface_count=physical_count,
                        )
                    )
                except (NotImplementedError, ValueError):
                    continue
                seen: set[tuple[int, int, int, int]] = set()
                for plan in incidence_plans:
                    signature = (
                        plan.left.valency,
                        plan.right.valency,
                        plan.left.physical_component_count,
                        plan.right.physical_component_count,
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    component_incidence_hypotheses.append({
                        "symmetry": plan.symmetry,
                        "interface_id": plan.interface_id,
                        "physical_interface_count": (
                            plan.physical_interface_count
                        ),
                        "left_participant": plan.left.participant,
                        "left_valency": plan.left.valency,
                        "left_component_count": (
                            plan.left.physical_component_count
                        ),
                        "right_participant": plan.right.participant,
                        "right_valency": plan.right.valency,
                        "right_component_count": (
                            plan.right.physical_component_count
                        ),
                        "executable": False,
                    })
    unresolved_variables = sorted(
        {
            variable
            for item in accepted_hypotheses
            for variable in item.unresolved
        }
    )
    declared_incidence: dict[str, list[str]] = {}
    for seed_id, seed in intent.interface_seeds.items():
        for participant in seed.participants:
            declared_incidence.setdefault(participant, []).append(seed_id)
    generation_length = intent.generation.length
    generation_length_payload = (
        generation_length
        if isinstance(generation_length, int)
        else generation_length.model_dump(mode="json")
    )
    try:
        executable_hypotheses = enumerate_simple_design_candidates(intent)
        resolver_ready = True
        resolver_candidate_count = len(executable_hypotheses)
        resolver_blocker = None
    except (NotImplementedError, TypeError, ValueError) as error:
        resolver_ready = False
        resolver_candidate_count = 0
        resolver_blocker = str(error)
    return {
        "schema_version": 1,
        "kind": intent.kind,
        "authoring_mode": "ordinary",
        "resolution_stage": "intent",
        "executable": False,
        "name": intent.name,
        "input": str(intent.input),
        "seed_layout": {
            "requested": intent.seed_layout,
            "relative_pose": "preserve_input",
        },
        "goal": {
            "architecture": intent.goal.architecture,
            "composition": intent.goal.composition,
            "symmetry": symmetry,
            "subunits": (
                intent.goal.subunits.model_dump(mode="json")
                if intent.goal.subunits is not None
                else None
            ),
            "diameter_angstrom": (
                intent.goal.diameter_angstrom.model_dump(mode="json")
                if intent.goal.diameter_angstrom is not None
                else None
            ),
            "cavity_diameter_angstrom": (
                intent.goal.cavity_diameter_angstrom.model_dump(mode="json")
                if intent.goal.cavity_diameter_angstrom is not None
                else None
            ),
        },
        "generation": {
            "length": generation_length_payload,
        },
        "resources": intent.resources.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "output": (
            intent.output.model_dump(mode="json", exclude_none=True)
            if intent.output is not None
            else None
        ),
        "interface_seeds": [
            {
                "id": seed_id,
                "participants": list(seed.participants),
                "selectors": dict(seed.selectors),
                "source": str(seed.source or intent.input),
                "use": seed.use.description,
                "geometry": seed.geometry,
            }
            for seed_id, seed in intent.interface_seeds.items()
        ],
        "detected_chain_count": len(inspection.chains),
        "detected_interface_candidate_count": len(
            inspection.interface_candidates
        ),
        "component_interface_sets": [
            {
                "chain_id": item.chain_id,
                "detected_port_count": item.detected_port_count,
                "interface_ids": list(item.interface_ids),
            }
            for item in inspection.component_interface_sets
        ],
        "declared_component_interface_sets": [
            {
                "participant": participant,
                "declared_port_count": len(interface_ids),
                "interface_ids": sorted(interface_ids),
            }
            for participant, interface_ids in sorted(
                declared_incidence.items()
            )
        ],
        "inspection": intent.inspection.model_dump(mode="json"),
        "declared_interface_evidence": declared_evidence,
        "architecture_hypotheses": [
            item.to_dict() for item in hypotheses
        ],
        "component_incidence_hypotheses": (
            component_incidence_hypotheses
        ),
        "accepted_architecture_hypothesis_count": len(
            accepted_hypotheses
        ),
        "blocking_unresolved_variables": (
            [] if resolver_ready else unresolved_variables
        ),
        "architecture_resolution": (
            "ordinary_resolver_ready"
            if resolver_ready
            else "candidate_symmetries_ready_connectivity_unresolved"
            if accepted_hypotheses
            else "requires_stabilizer_or_multi_component_search"
        ),
        "ordinary_resolver": {
            "ready": resolver_ready,
            "candidate_count": resolver_candidate_count,
            "supported_contract": (
                "several user-supplied preserve_exact seeds; bounded "
                "arbitrary-valency ordered unit paths; full-orbit finite-"
                "group relations; optional global relative-pose solving"
                if len(intent.interface_seeds) > 1
                else "one binary preserve_exact seed; full-orbit Cn ring"
            ),
            "blocker": resolver_blocker,
        },
    }


def _print_simple_intent_plan(
    intent: SimpleCageIntentSpec,
    *,
    output_format: str,
) -> None:
    payload = _simple_intent_analysis(intent)
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("RFD3-Mosaic ordinary-user cage intent")
    print(f"name:         {payload['name']}")
    print(f"input:        {payload['input']}")
    goal = payload["goal"]
    print(f"architecture: {goal['architecture']}")
    print(f"composition:  {goal['composition']}")
    print(f"symmetry:     {goal['symmetry']}")
    seed_layout = payload.get("seed_layout", {})
    if seed_layout:
        print(
            "seed layout:  "
            f"{seed_layout.get('requested', 'auto')} -> "
            f"{seed_layout.get('relative_pose', 'preserve_input')}"
        )
    length = payload["generation"]["length"]
    if isinstance(length, int):
        length_text = str(length)
    else:
        length_text = f"{length['minimum']}..{length['maximum']}"
    print(f"generation:   {length_text} residues per resolved connection")
    print("interface seeds:")
    for seed in payload["interface_seeds"]:
        print(
            f"  - {seed['id']}: "
            f"participants={','.join(seed['participants'])} "
            f"use={seed['use']} geometry={seed['geometry']}"
        )
    print(
        "input analysis: "
        f"{payload['detected_chain_count']} chains, "
        f"{payload['detected_interface_candidate_count']} "
        "candidate interfaces"
    )
    print(
        "architecture resolution: "
        f"{payload['architecture_resolution']}"
    )
    resolver = payload["ordinary_resolver"]
    if resolver["ready"]:
        print(
            "ordinary resolve: ready; "
            f"{resolver['candidate_count']} deterministic candidate(s)"
        )
        print("next: rfd3-mosaic resolve <intent> --output-dir <dir>")
    else:
        print(f"ordinary resolve: blocked: {resolver['blocker']}")
    print("observed component/interface incidence:")
    for component in payload["component_interface_sets"]:
        interfaces = ",".join(component["interface_ids"]) or "none"
        print(
            f"  - chain {component['chain_id']}: "
            f"ports={component['detected_port_count']} "
            f"interfaces={interfaces}"
        )
    print("declared seed-participant incidence:")
    for component in payload["declared_component_interface_sets"]:
        interfaces = ",".join(component["interface_ids"])
        print(
            f"  - participant {component['participant']}: "
            f"ports={component['declared_port_count']} "
            f"interfaces={interfaces}"
        )
    print("execution stage: intent only (not yet a frozen assembly)")
    if payload["blocking_unresolved_variables"]:
        print("Mosaic must still solve:")
        for variable in payload["blocking_unresolved_variables"]:
            print(f"  - {variable}")
    print("finite-group orbit-action candidates:")
    for hypothesis in payload["architecture_hypotheses"]:
        status = "compatible" if hypothesis["accepted"] else "rejected"
        print(
            f"  - {hypothesis['symmetry']}: {status}, "
            f"group_actions={hypothesis['group_action_count']}"
        )
    if payload["component_incidence_hypotheses"]:
        print("repeated supplied-interface component incidences:")
        for hypothesis in payload["component_incidence_hypotheses"]:
            print(
                f"  - {hypothesis['symmetry']}: "
                f"{hypothesis['left_participant']} "
                f"C{hypothesis['left_valency']} x"
                f"{hypothesis['left_component_count']} -- "
                f"{hypothesis['right_participant']} "
                f"C{hypothesis['right_valency']} x"
                f"{hypothesis['right_component_count']}; "
                f"interfaces={hypothesis['physical_interface_count']}"
            )


def _symmetry_copy_count(symmetry_id: str) -> int:
    """Return the finite number of proper symmetry copies."""

    if symmetry_id.startswith("C"):
        return int(symmetry_id[1:])
    if symmetry_id.startswith("D"):
        return 2 * int(symmetry_id[1:])
    return {"T": 12, "O": 24, "I": 60}[symmetry_id]


def _public_rigid_component_plan(
    design: UserDesignSpec,
    constraints,
    bound,
) -> list[dict[str, object]]:
    """Summarize rigid components before symmetry expansion."""

    bound_by_id = {item.plan.id: item for item in bound.operators}
    symmetry_id = (
        design.symmetry
        if isinstance(design.symmetry, str)
        else design.symmetry.id
    )
    copy_count = (
        len(design.finite_orbit_action.coset_representative_ids)
        if design.finite_orbit_action is not None
        else _symmetry_copy_count(symmetry_id)
    )
    components: dict[str, dict[str, object]] = {}
    component_atoms: dict[str, set[object]] = {}
    for operator in constraints.operators:
        if operator.operator != "fixed_xyz":
            continue
        component_id = operator.coupling_group or operator.id
        component = components.setdefault(
            component_id,
            {
                "component_id": component_id,
                "selectors": [],
                "declaration_ids": [],
                "pose": operator.parameters["pose"],
                "symmetry_copy_count": copy_count,
            },
        )
        component["selectors"].append(operator.selector)
        component["declaration_ids"].append(operator.id)
        component_atoms.setdefault(component_id, set()).update(
            bound_by_id[operator.id].atom_ids
        )
    result = []
    for component_id, component in components.items():
        atoms_per_copy = len(component_atoms[component_id])
        selector_count = len(component["selectors"])
        result.append(
            {
                **component,
                "selected_regions_per_copy": selector_count,
                "selected_atoms_per_copy": atoms_per_copy,
                "expanded_selected_atom_count": atoms_per_copy * copy_count,
                "interpretation": (
                    f"one rigid component containing {selector_count} "
                    f"selected region(s), expanded into {copy_count} "
                    "symmetry-related copies"
                ),
            }
        )
    return result


def _print_public_design_plan(
    design: UserDesignSpec,
    *,
    output_format: str,
) -> None:
    constraints = compile_constraint_plan(design)
    sampling = compile_sampling_plan(design)
    capabilities = required_capabilities_for_design(design)
    bound = bind_constraint_plan(design, constraints)
    try:
        lowered = lower_user_design(design)
    except (NotImplementedError, TypeError, ValueError) as error:
        lowering = {"status": "blocked", "reason": str(error)}
    else:
        lowering = {
            "status": "ready",
            "assembly": lowered.specification.model_dump(mode="json"),
        }
    inferred_interfaces = []
    automatic_initializations = []
    interface_usage = []
    if lowering["status"] == "ready":
        interface_usage = [
            {
                "interface_id": item.interface_id,
                "requested": item.requested,
                "physical_instance_count": item.physical_instance_count,
                "satisfied": item.satisfied,
            }
            for item in lowered.interface_usage
        ]
        inferred_interfaces = [
            {
                "id": interface_id,
                "mode": "design_generated_interface",
                "quality": "auto",
                "copy_relation": interface.copy_relation.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
            for interface_id, interface in lowered.specification.interfaces.items()
            if interface_id.startswith("auto_generated_interface_")
        ]
        if (
            sampling.initial_pose is None
            and not sampling.component_initial_poses
        ):
            automatic_initializations = [
                {
                    "group": group_id,
                    "radius": initialization.placement.radius.mean,
                    "axial_offset": (
                        initialization.placement.axial_offset.mean
                    ),
                    "radial_direction": (
                        initialization.placement.radial_direction
                    ),
                }
                for group_id, initialization in (
                    lowered.specification.initialization.items()
                )
            ]
    symmetry_id = (
        design.symmetry
        if isinstance(design.symmetry, str)
        else design.symmetry.id
    )
    rigid_components = _public_rigid_component_plan(
        design,
        constraints,
        bound,
    )
    payload = {
        "schema_version": 1,
        "user_mode": design.user_mode,
        "task": design.task.value if design.task is not None else None,
        "resolved_preferences": compile_design_preferences(
            design
        ).model_dump(mode="json"),
        "name": design.name,
        "input": str(design.input),
        "symmetry": symmetry_id,
        "generation": [
            item.model_dump(mode="json") for item in design.generation
        ],
        "components": {
            component_id: component.model_dump(mode="json")
            for component_id, component in design.components.items()
        },
        "ports": {
            port_id: port.model_dump(mode="json")
            for port_id, port in design.ports.items()
        },
        "interfaces": [
            item.model_dump(mode="json") for item in design.interfaces
        ],
        "interface_usage": interface_usage,
        "inferred_interfaces": inferred_interfaces,
        "automatic_initializations": automatic_initializations,
        "connections": [
            item.model_dump(mode="json", by_alias=True)
            for item in design.connections
        ],
        "constraint_plan": constraints.model_dump(mode="json"),
        "sampling_plan": sampling.model_dump(mode="json"),
        "required_capabilities": [
            {
                **item.model_dump(mode="json"),
                "maturity": item.maturity.label,
            }
            for item in capabilities
        ],
        "resolved_atom_counts": {
            operator.plan.id: len(operator.atom_ids)
            for operator in bound.operators
        },
        "rigid_components": rigid_components,
        "assembly_lowering": lowering,
    }
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("RFD3-Mosaic public design plan")
    print(
        "user mode:  "
        + (
            "simple (automatic assembly planning)"
            if design.user_mode == "simple"
            else "expert (explicit assembly graph)"
        )
    )
    if design.task is None:
        print("task:       custom/legacy declarations (no task preset)")
    elif design.task.value == "preserve_supplied_geometry":
        print(
            "task:       preserve supplied geometry "
            "(complete interface seed remains joint-rigid; generate only "
            "declared scaffold)"
        )
    else:
        arrangement = design.fixed_arrangement.value
        print(
            "task:       create symmetric interface "
            "(generated packing guidance; fixed arrangement="
            f"{arrangement})"
        )
    print(f"name:       {design.name}")
    print(f"input:      {design.input}")
    print(f"symmetry:   {symmetry_id}")
    resolved_preferences = compile_design_preferences(design)
    if design.task == UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY:
        print(
            "packing guidance: inactive for supplied interfaces "
            "(preserve_input is checked at the input stage)"
        )
    else:
        print(
            "preferences: "
            f"packing={resolved_preferences.packing.value} "
            f"area={resolved_preferences.interface_area.value} "
            f"cavity={resolved_preferences.cavity.value} "
            f"diversity={resolved_preferences.diversity.value} "
            f"motion={resolved_preferences.component_motion.value}"
        )
    print(
        "generation: "
        f"{len(design.generation) + len(design.connections)} region(s)"
    )
    if inferred_interfaces:
        if (
            design.task is not None
            and design.task.value == "create_symmetric_interface"
        ):
            if design.fixed_arrangement.value == "locked":
                print(
                    "inferred design mode: complete fixed arrangement + "
                    "generated-only packing guidance"
                )
            else:
                print(
                    "inferred design mode: exact rigid components + bounded "
                    "joint pose/packing optimization"
                )
        else:
            print(
                "inferred design mode: fixed central motif -> generated "
                "interface"
            )
        for interface in inferred_interfaces:
            relation = interface["copy_relation"]
            neighbour = relation.get(
                "transform",
                f"orbit_offset={relation.get('orbit_offset')}",
            )
            print(
                f"  - {interface['id']}: neighbour={neighbour} "
                "quality=auto (coverage + contiguous packing patch)"
            )
    elif design.generation and all(
        isinstance(item, BetweenGeneration) for item in design.generation
    ):
        print(
            "inferred design mode: supplied fixed geometry -> generated "
            "linker (no new interface target)"
        )
    if design.components:
        print(
            f"assembly graph: {len(design.components)} component(s), "
            f"{len(design.ports)} port(s), "
            f"{len(design.interfaces)} interface(s), "
            f"{len(design.connections)} connection(s)"
        )
        if design.ports:
            print("interface ports:")
            for port_id, port in design.ports.items():
                print(
                    f"  - {port_id}: component={port.component} "
                    f"selectors={','.join(port.selectors)}"
                )
        if design.interfaces:
            print("interface edges:")
            for interface in design.interfaces:
                relation = interface.copy_relation
                neighbour = (
                    relation.transform
                    if relation.transform is not None
                    else f"orbit_offset={relation.orbit_offset}"
                )
                hyperedge = (
                    f"hyperedge={interface.hyperedge_id}; "
                    if interface.hyperedge_id is not None
                    else ""
                )
                participants = (
                    f"{interface.between[0]} -> {interface.between[1]}"
                    if len(interface.between) == 2
                    else "[" + " + ".join(interface.between) + "]"
                )
                print(
                    f"  - {interface.id}: {hyperedge}"
                    f"{participants}@{neighbour} "
                    f"relation={interface.relation.mode} "
                    f"use={interface.use.description} "
                    f"required={interface.required}"
                )
        if interface_usage:
            print("resolved interface multiplicity:")
            for usage in interface_usage:
                print(
                    f"  - {usage['interface_id']}: "
                    f"requested={usage['requested']} "
                    "physical_instances="
                    f"{usage['physical_instance_count']}"
                )
    if (
        sampling.initial_pose is None
        and not sampling.component_initial_poses
    ):
        if automatic_initializations:
            print("initial pose: automatically planned from motif geometry")
            for pose in automatic_initializations:
                print(
                    f"  - component={pose['group']} "
                    f"radius={pose['radius']:g} "
                    f"axial={pose['axial_offset']:g}"
                )
        else:
            print("initial pose: input coordinates (already usable)")
    elif sampling.initial_pose is not None:
        pose = sampling.initial_pose
        print(
            "initial pose: "
            f"radius=[{pose.radius_minimum:g}, {pose.radius_maximum:g}] "
            f"axial=[{pose.axial_minimum:g}, {pose.axial_maximum:g}] "
            f"orientation={pose.orientation_method} seed={pose.seed}"
        )
    else:
        print("initial poses:")
        for pose in sampling.component_initial_poses:
            print(
                f"  - component={pose.group_id} "
                f"radius=[{pose.radius_minimum:g}, "
                f"{pose.radius_maximum:g}] "
                f"axial=[{pose.axial_minimum:g}, "
                f"{pose.axial_maximum:g}] "
                f"orientation={pose.orientation_method} seed={pose.seed}"
            )
    print("constraints:")
    if not constraints.operators:
        print("  - none (unconstrained degrees of freedom use normal diffusion)")
    for operator in constraints.operators:
        atom_count = next(
            len(item.atom_ids)
            for item in bound.operators
            if item.plan.id == operator.id
        )
        component = (
            " component="
            f"{operator.coupling_group or operator.id}"
            " pose="
            f"{operator.parameters['pose']['mode']}"
            if operator.operator == "fixed_xyz"
            else ""
        )
        print(
            f"  - {operator.operator} [{operator.stage.value}] "
            f"selector={operator.selector} "
            f"dofs={','.join(operator.controlled_dofs)} atoms={atom_count}"
            f"{component}"
        )
    print("rigid components:")
    if not rigid_components:
        print("  - none")
    for component in rigid_components:
        pose = component["pose"]
        print(
            f"  - {component['component_id']}: "
            f"{component['selected_regions_per_copy']} selected region(s) "
            f"and {component['selected_atoms_per_copy']} atom(s) per copy "
            f"x {component['symmetry_copy_count']} symmetry copies; "
            f"pose={pose['mode']}"
        )
        print(
            "    selectors: "
            + ", ".join(str(value) for value in component["selectors"])
        )
    print("required capabilities:")
    if not capabilities:
        print("  - none beyond the selected base RFD3 execution path")
    for item in capabilities:
        print(f"  - {item.id}: {item.maturity.label}")
    print(f"assembly lowering: {lowering['status']}")
    if lowering["status"] == "blocked":
        print(f"  reason: {lowering['reason']}")


def _print_capabilities(*, output_format: str) -> None:
    manifest = capability_manifest()
    if output_format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    print("RFD3-Mosaic capability matrix")
    print(
        "maturity: planned < schema_only < cpu_validated < gpu_canary "
        "< engineering < stable < scientifically_validated"
    )
    for item in manifest["capabilities"]:
        visibility = "public" if item["public_interface"] else "internal"
        print(
            f"{item['id']:<30} {item['maturity']:<24} {visibility}"
        )
        print(f"  {item['summary']}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "capabilities":
        _print_capabilities(output_format=arguments.format)
        return
    if arguments.command == "inspect":
        try:
            inspection = inspect_structure_interfaces(
                arguments.input,
                contact_cutoff=arguments.contact_cutoff,
                minimum_atom_contacts=arguments.minimum_atom_contacts,
                minimum_contact_residues_per_side=(
                    arguments.minimum_contact_residues_per_side
                ),
            )
            if not inspection.interface_candidates:
                raise ValueError(
                    "No candidate interfaces passed the inspection gates; "
                    "check the input assembly or relax the explicit inspect "
                    "thresholds"
                )
            name = _safe_default_name(
                arguments.name or f"{arguments.input.stem}-cage"
            )
            report_path, intent_path = write_structure_inspection(
                inspection,
                arguments.output_dir,
                intent_name=name,
                architecture=arguments.architecture,
                composition=arguments.composition,
                symmetries=tuple(arguments.inspect_symmetries or ()),
                generated_length_minimum=arguments.length_min,
                generated_length_maximum=arguments.length_max,
                subunit_minimum=arguments.subunits_min,
                subunit_maximum=arguments.subunits_max,
                diameter_minimum=arguments.diameter_min,
                diameter_maximum=arguments.diameter_max,
                cavity_diameter_minimum=arguments.cavity_diameter_min,
                cavity_diameter_maximum=arguments.cavity_diameter_max,
                profile=arguments.profile,
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print("RFD3-Mosaic structure inspection: PASSED")
        print(f"chains:               {len(inspection.chains)}")
        print(
            "candidate interfaces: "
            f"{len(inspection.interface_candidates)}"
        )
        for candidate in inspection.interface_candidates:
            print(
                f"  - {candidate.interface_id}: "
                f"{candidate.left_chain}<->{candidate.right_chain} "
                f"contacts={candidate.heavy_atom_contact_count} "
                "contact_residues="
                f"{candidate.left_contact_residue_count}/"
                f"{candidate.right_contact_residue_count}"
            )
        print("detected component faces:")
        for component in inspection.component_interface_sets:
            interfaces = ", ".join(component.interface_ids) or "none"
            print(
                f"  - chain {component.chain_id}: "
                f"{component.detected_port_count} port(s) [{interfaces}]"
            )
        print(f"inspection report:    {report_path}")
        print(f"editable simple YAML: {intent_path}")
        return
    if arguments.command == "runs":
        try:
            if arguments.limit < 1:
                raise ValueError("--limit must be a positive integer")
            rebuild = (
                rebuild_run_index(arguments.root)
                if arguments.rebuild
                else None
            )
            records = list_run_records(arguments.root)[: arguments.limit]
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        if arguments.format == "json":
            payload = (
                {"schema_version": 1, "rebuild": rebuild, "runs": records}
                if rebuild is not None
                else records
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        print("RFD3-Mosaic indexed runs")
        print(f"root: {arguments.root.expanduser().resolve()}")
        if rebuild is not None:
            print(
                "rebuild: "
                f"indexed={rebuild['indexed']} "
                f"skipped={rebuild['skipped']} "
                f"failed={rebuild['failed']}"
            )
            for failure in rebuild["failures"]:
                print(f"  WARNING {failure['path']}: {failure['error']}")
        if not records:
            print("  no indexed submissions")
            return
        print("JOB ID       STATE       EXPERIMENT")
        for record in records:
            print(
                f"{str(record['job_id']):<12} "
                f"{str(record.get('state') or 'unknown'):<11} "
                f"{record.get('experiment') or 'unknown'}"
            )
        return
    if arguments.command == "audit":
        try:
            reference = resolve_run_reference(
                arguments.target,
                root=arguments.root,
            )
            if reference.run_directory is None:
                raise ValueError(
                    "The referenced job has no run directory to audit"
                )
            result = audit_existing_run(reference.run_directory)
            status = collect_run_status(
                reference,
                include_scheduler=not arguments.no_scheduler,
            )
            report_path = write_report(status)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))
        print("RFD3-Mosaic post-hoc audit")
        print(f"run:       {result.run_directory}")
        print("inference: reused existing output (not rerun)")
        audit_states = {
            item["name"]: item for item in status.get("audits") or []
        }
        for report in result.reports:
            record = audit_states.get(report.name) or {}
            label = (
                "PASS"
                if record.get("passed") is True
                else "FAIL"
                if report.is_file()
                else "MISSING"
            )
            print(f"- {label} {report.name}")
        print(f"verdict:   {'PASSED' if result.passed else 'FAILED'}")
        if result.error:
            print(f"failure:   {result.error}")
        print(f"HTML report: {report_path}")
        print(f"JSON report: {report_path.with_suffix('.json')}")
        if not result.passed:
            raise SystemExit(1)
        return
    if arguments.command in {"status", "report"}:
        try:
            reference = resolve_run_reference(
                arguments.target,
                root=arguments.root,
            )
            status = collect_run_status(
                reference,
                include_scheduler=not arguments.no_scheduler,
            )
            if arguments.command == "status":
                if arguments.format == "json":
                    print(json.dumps(status, indent=2, sort_keys=True))
                else:
                    print(format_status_text(status))
                return
            report_path = write_report(status, arguments.output)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(f"HTML report: {report_path}")
        print(f"JSON report: {report_path.with_suffix('.json')}")
        return
    quick_command = arguments.command in {"central", "interface"}
    simple_intent: SimpleCageIntentSpec | None = None
    public_design: UserDesignSpec | None = None
    if not quick_command:
        try:
            simple_intent = _load_simple_intent_if_present(arguments.config)
            if simple_intent is None:
                public_design = _load_public_design_if_present(
                    arguments.config
                )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if simple_intent is not None:
        if arguments.command == "resolve":
            try:
                result = resolve_simple_intent(
                    simple_intent,
                    arguments.output_dir,
                    source_path=arguments.config,
                    symmetry_ids=arguments.resolve_symmetries,
                    pose_samples=arguments.pose_samples,
                    seed_start=arguments.seed_start,
                    timesteps=arguments.timesteps,
                    top_count=arguments.top,
                    max_candidates=arguments.max_candidates,
                    optimize_poses=arguments.optimize_poses,
                    pose_optimize_top=arguments.pose_optimize_top,
                    pose_optimization_levels=(
                        arguments.pose_optimization_levels
                    ),
                    pose_maximum_translation=(
                        arguments.pose_maximum_translation
                    ),
                    pose_maximum_rotation_deg=(
                        arguments.pose_maximum_rotation_deg
                    ),
                    progress=lambda message: print(
                        f"[resolve] {message}",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            except (
                NotImplementedError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                parser.error(str(error))
            if arguments.format == "json":
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            print("RFD3-Mosaic ordinary intent resolution")
            print(f"resolver:    {result['resolver']}")
            print(f"candidates:  {result['candidate_count']}")
            print(f"accepted:    {result['accepted_count']}")
            print(f"selected:    {result['selected_count']}")
            print(
                "automatic CPU recommendation: "
                + (
                    str(result["recommended_design"])
                    if result.get("recommended_design")
                    else "none (no strictly replayable candidate)"
                )
            )
            pose_optimization = result.get(
                "continuous_pose_optimization",
                {},
            )
            print(
                "CPU pose optimization: "
                + (
                    "enabled"
                    if pose_optimization.get("enabled")
                    else "disabled"
                )
            )
            for candidate in result["ranking"][: arguments.top]:
                state = (
                    "READY"
                    if candidate.get("resolved_design")
                    else "REJECTED"
                )
                print(
                    f"  - rank={candidate.get('rank', '-')} "
                    f"{state} symmetry={candidate['symmetry']} "
                    f"topology={candidate.get('topology_id')} "
                    "physical_units="
                    f"{candidate.get('physical_polymer_unit_count', '-')}"
                )
                if candidate.get("resolved_design"):
                    print(f"    design: {candidate['resolved_design']}")
                    restoration = candidate.get(
                        "feasibility_restoration"
                    ) or {}
                    bindings = restoration.get(
                        "linker_length_bindings"
                    ) or []
                    if restoration.get("changed") and bindings:
                        rendered_bindings = ", ".join(
                            f"{binding['source_link_id']}="
                            f"{binding['selected_length']}"
                            + (
                                f" (tie={binding['tie_group']})"
                                if binding.get("tie_group")
                                else ""
                            )
                            for binding in bindings
                        )
                        print(
                            "    restored linker lengths: "
                            + rendered_bindings
                        )
                elif candidate.get("replay_error"):
                    print(
                        "    replay error: "
                        f"{candidate['replay_error']}"
                    )
                elif candidate.get("error"):
                    print(f"    error: {candidate['error']}")
            print(f"manifest:    {result['manifest_path']}")
            return
        if arguments.command == "plan":
            try:
                _print_simple_intent_plan(
                    simple_intent,
                    output_format=arguments.format,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(str(error))
            return
        if arguments.command == "validate":
            try:
                payload = _simple_intent_analysis(simple_intent)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(str(error))
            print("Simple cage intent validation: PASSED")
            print(f"name:       {payload['name']}")
            print(
                "interfaces: "
                f"{len(payload['interface_seeds'])}"
            )
            print(
                "compatible finite-group orbit actions: "
                f"{payload['accepted_architecture_hypothesis_count']}"
            )
            print(
                "next stage: "
                f"{payload['architecture_resolution']}"
            )
            if payload["ordinary_resolver"]["ready"]:
                print(
                    "resolvable: yes (run resolve, then choose a strictly "
                    "replayed YAML)"
                )
            else:
                print("executable: no (architecture not frozen)")
            return
        parser.error(
            "Simple cage intent is valid but not yet frozen into one "
            "AssemblySpecification; run resolve and choose one replayed "
            "selected YAML before render/run"
        )
    if public_design is not None:
        if arguments.command == "resolve":
            parser.error(
                "resolve expects kind: simple_cage_intent; this file is "
                "already a standard executable public design"
            )
        if arguments.command == "search":
            try:
                result = search_graph_design(
                    public_design,
                    arguments.output_dir,
                    source_path=arguments.config,
                    symmetry_ids=arguments.search_symmetries,
                    interface_ids=arguments.interfaces,
                    include_identity=arguments.include_identity,
                    pose_samples=arguments.pose_samples,
                    seed_start=arguments.seed_start,
                    top_count=arguments.top,
                    max_combinations=arguments.max_candidates,
                )
            except (OSError, TypeError, ValueError) as error:
                parser.error(str(error))
            if arguments.format == "json":
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            print("RFD3-Mosaic graph search")
            print(
                "symmetries: "
                + ", ".join(result["searched_symmetries"])
            )
            print(f"candidates: {result['candidate_count']}")
            print(f"accepted:   {result['accepted_count']}")
            print(
                "need diffusion interface formation: "
                f"{result['diffusion_interface_formation_count']}"
            )
            print(f"failed:     {result['failed_compilation_count']}")
            print(f"replay failures: {result['replay_failure_count']}")
            print(f"selected:   {result['selected_count']}")
            print("top candidates:")
            for candidate in result["ranking"][: arguments.top]:
                transforms = ", ".join(
                    f"{key}={value}"
                    for key, value in candidate[
                        "neighbour_transforms"
                    ].items()
                )
                symmetry = candidate["symmetry"]
                if candidate.get("error") is not None:
                    print(
                        f"  - {candidate['candidate_id']} REJECTED "
                        f"symmetry={symmetry} {transforms} "
                        f"error={candidate['error']}"
                    )
                    continue
                interface_target = (
                    "needs_diffusion"
                    if candidate.get(
                        "requires_diffusion_interface_formation"
                    )
                    else "initialized"
                )
                print(
                    f"  - rank={candidate.get('rank', '-')} "
                    f"accepted={candidate['accepted']} "
                    f"symmetry={symmetry} {transforms} "
                    f"pose_sample={candidate['pose_sample_index']} "
                    f"clashes={candidate['hard_clashes']} "
                    "interface_target="
                    f"{interface_target} "
                    "contacts="
                    f"{candidate['interface_contact_count_below_4_5A']} "
                    "max_link="
                    f"{candidate['maximum_linker_endpoint_distance']}"
                )
                if candidate.get("resolved_design"):
                    print(f"    design: {candidate['resolved_design']}")
            print(f"manifest:   {result['manifest_path']}")
            return
        if arguments.command == "validate":
            try:
                plan = compile_constraint_plan(public_design)
                bind_constraint_plan(public_design, plan)
                preflight = _preflight_public_design_geometry(public_design)
            except (
                NotImplementedError,
                OSError,
                TypeError,
                ValueError,
            ) as error:
                parser.error(str(error))
            print("User design validation: PASSED")
            print(f"name:        {public_design.name}")
            symmetry_id = (
                public_design.symmetry
                if isinstance(public_design.symmetry, str)
                else public_design.symmetry.id
            )
            print(f"symmetry:    {symmetry_id}")
            print(f"constraints: {len(plan.operators)}")
            print(
                "geometry:    PASSED "
                f"({preflight.atom_count} atoms, "
                f"{preflight.residue_count} residues, "
                f"{preflight.chain_count} chains)"
            )
            print("RFD3 input:  PASSED (runtime features are finite)")
            return
        if arguments.command == "plan":
            try:
                _print_public_design_plan(
                    public_design,
                    output_format=arguments.format,
                )
            except (TypeError, ValueError) as error:
                parser.error(str(error))
            return
        try:
            config_path = _write_public_experiment(
                public_design,
                arguments.config,
            )
        except (NotImplementedError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        public_design = None
    else:
        if arguments.command == "search":
            parser.error(
                "search requires a public components/ports/interfaces design"
            )
        config_path = (
            _write_quick_experiment(arguments)
            if quick_command
            else arguments.config
        )
    try:
        experiment = resolve_experiment(
            config_path,
            profile_override=arguments.profile,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if arguments.command == "validate":
        print("Experiment validation: PASSED")
        print(f"name:     {experiment.name}")
        print(f"topology: {experiment.payload['topology']['kind']}")
        print(f"profile:  {experiment.payload['resources']['profile_name']}")
        print(f"run root: {experiment.run_root}")
        return

    if arguments.command == "plan":
        _print_execution_plan(
            build_execution_plan(experiment),
            output_format=arguments.format,
        )
        return

    if quick_command:
        print(f"generated experiment: {config_path}")

    try:
        script = render_submission(
            experiment,
            output_directory=getattr(arguments, "output_dir", None),
        )
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"resolved config: {script.parent / 'resolved_config.yaml'}")
    print(f"Slurm script:    {script}")
    if arguments.command == "render" or getattr(arguments, "dry_run", False):
        print("Submission: skipped")
        return

    try:
        executor = executor_for_id(experiment.payload["resources"]["executor"])
        submitted = executor.submit(script)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    output = submitted.message
    job_id = submitted.job_id
    receipt = {
        "job_id": job_id,
        "executor": submitted.executor,
        "sbatch_output": output,
        "script": str(script),
        "experiment": experiment.name,
        "run_root": str(experiment.run_root),
        "expected_run_directory": str(experiment.run_root / job_id),
    }
    receipt_path = script.parent / "submission.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        index_path = record_submission(
            root=experiment.payload["output"]["root"],
            job_id=job_id,
            experiment=experiment.name,
            campaign=experiment.payload["output"]["campaign"],
            run_directory=experiment.run_root / job_id,
            submission_directory=script.parent,
            executor=submitted.executor,
        )
    except (OSError, ValueError) as error:
        receipt["index_error"] = str(error)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"WARNING: run index was not updated: {error}")
        index_path = None
    print(output)
    print(f"submission receipt: {receipt_path}")
    if index_path is not None:
        print(f"run index:          {index_path}")


if __name__ == "__main__":
    main()
