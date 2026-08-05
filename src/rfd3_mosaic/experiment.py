"""Validated user-facing experiment configuration and Slurm rendering."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any

import yaml

from rfd3_mosaic.provenance.software import (
    collect_repository_provenance,
    file_identity,
    load_compatibility_manifest,
)
from rfd3_mosaic.provenance.source_snapshot import create_source_snapshot
from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.schema import load_user_design


SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TOPOLOGY_KINDS = {"interface_seed", "central_motif", "user_design"}
SAMPLER_PRESETS: dict[str, dict[str, Any]] = {
    "exact_mosaic": {
        "kind": "symmetry",
        "allow_realignment": False,
        "fixed_motif_finalization_mode": "motif_precedence",
        "preserve_fixed_motif_during_symmetry": True,
        "require_motif_constraint_groups": True,
        "symmetry_state_mode": "orbit_average",
        "symmetry_noise_mode": "coupled",
    },
    "official_rfd3": {
        "kind": "symmetry",
        "allow_realignment": False,
        "fixed_motif_finalization_mode": "official_reinsert_then_project",
        "preserve_fixed_motif_during_symmetry": False,
        "require_motif_constraint_groups": False,
        "symmetry_state_mode": "legacy_asu",
        "symmetry_noise_mode": "independent",
    },
}


def _repository_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / ".project-root").is_file():
            return candidate.resolve()
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if (candidate / ".project-root").is_file():
            return candidate
    raise RuntimeError("Cannot locate the rfd3-mosaic repository root")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")


def _safe_name(value: Any, label: str) -> str:
    result = str(value or "")
    if not SAFE_NAME.fullmatch(result) or result in {".", ".."}:
        raise ValueError(f"{label} must be one safe name, observed {result!r}")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _single_line(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be one non-empty line")
    return value


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    result = _single_line(value, label).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError(f"{label} must be one 64-character SHA256 digest")
    return result


def _resolve_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_existing_file(value: Any, *, base: Path, label: str) -> Path:
    path = _resolve_path(value, base=base, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolve_profile_path(
    value: Any,
    *,
    experiment_directory: Path,
    repository_root: Path,
) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("resources.profile must be a profile name or path")
    requested = Path(value).expanduser()
    if requested.suffix in {".yaml", ".yml"} or requested.parent != Path("."):
        return _resolve_existing_file(
            value,
            base=experiment_directory,
            label="execution profile",
        )
    profile = (
        repository_root
        / "configs"
        / "rfd3_mosaic"
        / "execution"
        / f"{value}.yaml"
    )
    if not profile.is_file():
        raise FileNotFoundError(
            f"Unknown execution profile {value!r}; expected {profile}"
        )
    return profile.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_dependency_files(payload: dict[str, Any]) -> list[tuple[str, Path]]:
    """Resolve every input file read again inside the allocated worker."""

    topology = payload["topology"]
    project = Path(payload["project_directory"])
    dependencies: list[tuple[str, Path]] = []
    if topology["kind"] == "central_motif":
        template = Path(topology["template_input"])
        dependencies.append(("central motif template", template))
        template_payload = json.loads(template.read_text(encoding="utf-8"))
        if not isinstance(template_payload, dict) or len(template_payload) != 1:
            raise ValueError(
                "Central motif template must contain exactly one RFD3 example"
            )
        example = next(iter(template_payload.values()))
        if not isinstance(example, dict) or not example.get("input"):
            raise ValueError("Central motif template has no input structure")
        structure = Path(str(example["input"])).expanduser()
        if not structure.is_absolute():
            structure = template.parent / structure
        dependencies.append(("central motif structure", structure.resolve()))
    elif topology["kind"] == "interface_seed":
        config = Path(topology["config"])
        dependencies.append(("assembly specification", config))
        config_payload = _load_yaml(config)
        wrapped = config_payload.get("assembly")
        if wrapped is None:
            wrapped = config_payload.get("interface_seed", config_payload)
        if not isinstance(wrapped, dict):
            raise ValueError("Assembly specification must contain a mapping")
        fragments = wrapped.get("fragments") or {}
        if not isinstance(fragments, dict):
            raise ValueError("Assembly fragments must be a mapping")
        for fragment_id, fragment in fragments.items():
            if not isinstance(fragment, dict) or not fragment.get("source"):
                raise ValueError(
                    f"Assembly fragment {fragment_id!r} has no source"
                )
            source = Path(str(fragment["source"])).expanduser()
            if not source.is_absolute():
                source = project / source
            dependencies.append(
                (f"fragment source {fragment_id}", source.resolve())
            )
        manifest = topology.get("pose_candidate_manifest")
        if manifest is not None:
            dependencies.append(("pose candidate manifest", Path(manifest)))
    else:
        design_path = Path(topology["config"])
        design = load_user_design(design_path)
        dependencies.extend(
            [
                ("public user design", design_path),
                ("public design structure", design.input),
            ]
        )

    provenance = payload["provenance"]
    dependencies.extend(
        [
            ("experiment source", Path(provenance["experiment_source"])),
            ("execution profile", Path(provenance["profile_source"])),
            (
                "Foundry compatibility manifest",
                Path(provenance["foundry_compatibility"]["path"]),
            ),
        ]
    )
    unique: dict[Path, str] = {}
    for role, path in dependencies:
        unique.setdefault(path.expanduser().resolve(), role)
    return [(role, path) for path, role in unique.items()]


def _freeze_render_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Create the fail-closed identity contract consumed by the worker."""

    resources = payload["resources"]
    checkpoint = Path(resources["checkpoint"]).expanduser().resolve()
    checkpoint_record = file_identity(checkpoint, role="RFD3 checkpoint")
    declared_checkpoint_sha = resources.get("checkpoint_sha256")
    observed_checkpoint_sha = checkpoint_record["sha256"]
    if (
        declared_checkpoint_sha is not None
        and declared_checkpoint_sha != observed_checkpoint_sha
    ):
        raise ValueError(
            "Execution profile checkpoint_sha256 does not match the "
            f"checkpoint: {declared_checkpoint_sha} != {observed_checkpoint_sha}"
        )
    # A profile may omit the digest because the checkpoint exists only on the
    # target cluster.  Rendering there materializes the exact digest into the
    # resolved configuration before the job is submitted.
    resources["checkpoint_sha256"] = observed_checkpoint_sha
    repository = collect_repository_provenance(
        Path(payload["project_directory"])
    )
    dependency_records = [
        file_identity(path, role=role)
        for role, path in _runtime_dependency_files(payload)
    ]
    by_role = {record["role"]: record for record in dependency_records}
    provenance = payload["provenance"]
    expected_source_hashes = {
        "experiment source": provenance["experiment_sha256"],
        "execution profile": provenance["profile_sha256"],
        "Foundry compatibility manifest": provenance[
            "foundry_compatibility"
        ]["sha256"],
    }
    for role, expected_sha256 in expected_source_hashes.items():
        observed_sha256 = by_role[role]["sha256"]
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"{role} changed between resolution and render: "
                f"{expected_sha256} != {observed_sha256}"
            )
    return {
        "schema_version": 1,
        "repository": repository,
        "files": dependency_records,
        "checkpoint": checkpoint_record,
    }


@dataclass(frozen=True)
class ResolvedExperiment:
    """Fully resolved experiment plus its source provenance."""

    source_path: Path
    profile_path: Path
    payload: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def run_root(self) -> Path:
        output = self.payload["output"]
        return Path(output["root"]) / output["campaign"] / self.name


def build_execution_plan(experiment: ResolvedExperiment) -> dict[str, Any]:
    """Return a read-only, user-auditable description of one experiment."""

    payload = experiment.payload
    topology = payload["topology"]
    sampling = payload["sampling"]
    resources = payload["resources"]
    if topology["kind"] == "central_motif":
        effective_constraints = [
            {
                "operator": "fixed_xyz",
                "selector": topology["fixed_selector"],
                "atom_scope": "all_motif_atoms",
                "orbit_scope": "complete_symmetry_orbit",
                "source": "central_motif compatibility preset",
            }
        ]
        generation = {
            "n_terminal_length": topology["n_terminal_length"],
            "c_terminal_length": topology["c_terminal_length"],
        }
        input_record = {"template_input": topology["template_input"]}
    elif topology["kind"] == "interface_seed":
        effective_constraints = [
            {
                "operator": "fixed_xyz",
                "selector": "compiled interface-seed constraint groups",
                "atom_scope": "all_seed_atoms",
                "orbit_scope": "complete_symmetry_orbits",
                "source": "interface_seed compatibility preset",
            }
        ]
        generation = {"linker_length": topology["linker_length"]}
        input_record = {
            "config": topology["config"],
            "pose_candidate_manifest": topology["pose_candidate_manifest"],
            "pose_seed": topology["pose_seed"],
        }
    else:
        declared = load_user_design(topology["config"])
        constraint_plan = compile_constraint_plan(declared)
        effective_constraints = [
            {
                "operator": operator.operator,
                "selector": operator.selector,
                "atom_scope": operator.atoms.value,
                "orbit_scope": operator.orbit_scope.value,
                "controlled_dofs": list(operator.controlled_dofs),
                "parameters": operator.parameters,
                **(
                    {
                        "coupling_group": (
                            operator.coupling_group or operator.id
                        )
                    }
                    if operator.operator == "fixed_xyz"
                    else {}
                ),
                "source": "UserDesignSpec",
            }
            for operator in constraint_plan.operators
        ]
        generation = {
            "regions": [
                item.model_dump(mode="json")
                for item in declared.generation
            ]
        }
        input_record = {
            "config": topology["config"],
            "structure": str(declared.input),
            "symmetry": (
                declared.symmetry
                if isinstance(declared.symmetry, str)
                else declared.symmetry.id
            ),
        }

    compatibility = payload["provenance"]["foundry_compatibility"]
    repository = payload["provenance"]["repository"]
    return {
        "schema_version": 1,
        "name": experiment.name,
        "design": {
            "topology": topology["kind"],
            "input": input_record,
            "generation": generation,
            "effective_constraints": effective_constraints,
        },
        "sampling": {
            "preset": sampling["preset"],
            "timesteps": sampling["timesteps"],
            "seed": sampling["seed"],
            "execution_backend": sampling["execution_backend"],
            "neighbour_radius": sampling["neighbour_radius"],
            "low_memory_mode": sampling["low_memory_mode"],
            "effective_sampler": sampling["sampler"],
        },
        "execution": {
            "profile": resources["profile_name"],
            "slurm": resources["slurm"],
            "checkpoint": resources["checkpoint"],
            "checkpoint_sha256": resources["checkpoint_sha256"],
        },
        "output": {"run_root": str(experiment.run_root)},
        "software": {
            "commit": repository["commit"],
            "branch": repository["branch"],
            "tracked_dirty": repository["tracked_dirty"],
            "working_tree_diff_sha256": repository[
                "working_tree_diff_sha256"
            ],
            "compatibility_id": compatibility["manifest"]["engine_id"],
            "foundry_base_commit": compatibility["manifest"]["foundry"][
                "base_commit"
            ],
            "compatibility_manifest_sha256": compatibility["sha256"],
        },
    }


def resolve_experiment(
    path: str | Path,
    *,
    profile_override: str | Path | None = None,
) -> ResolvedExperiment:
    """Load, validate and resolve an experiment without changing state."""

    source_path = Path(path).expanduser().resolve()
    raw = _load_yaml(source_path)
    _reject_unknown(
        raw,
        {"schema_version", "name", "topology", "sampling", "resources", "output"},
        "experiment",
    )
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    name = _safe_name(raw.get("name"), "name")
    experiment_directory = source_path.parent
    repository_root = _repository_root()

    topology = raw.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("topology must be a mapping")
    kind = str(topology.get("kind") or "")
    if kind not in TOPOLOGY_KINDS:
        raise ValueError(f"topology.kind must be one of {sorted(TOPOLOGY_KINDS)}")
    resolved_topology: dict[str, Any] = {"kind": kind}
    if kind == "central_motif":
        _reject_unknown(
            topology,
            {
                "kind",
                "template_input",
                "fixed_selector",
                "n_terminal_length",
                "c_terminal_length",
            },
            "central_motif topology",
        )
        resolved_topology.update(
            {
                "template_input": str(
                    _resolve_existing_file(
                        topology.get("template_input"),
                        base=experiment_directory,
                        label="topology.template_input",
                    )
                ),
                "fixed_selector": str(topology.get("fixed_selector") or ""),
                "n_terminal_length": _positive_integer(
                    topology.get("n_terminal_length", 35),
                    "topology.n_terminal_length",
                ),
                "c_terminal_length": _positive_integer(
                    topology.get("c_terminal_length", 35),
                    "topology.c_terminal_length",
                ),
            }
        )
        if not resolved_topology["fixed_selector"]:
            raise ValueError("topology.fixed_selector is required")
    elif kind == "interface_seed":
        _reject_unknown(
            topology,
            {
                "kind",
                "config",
                "pose_candidate_manifest",
                "pose_seed",
                "linker_length",
                "example_id",
            },
            "interface_seed topology",
        )
        resolved_topology["config"] = str(
            _resolve_existing_file(
                topology.get("config"),
                base=experiment_directory,
                label="topology.config",
            )
        )
        manifest = topology.get("pose_candidate_manifest")
        pose_seed = topology.get("pose_seed")
        if manifest is not None and pose_seed is not None:
            raise ValueError(
                "pose_candidate_manifest and pose_seed are mutually exclusive"
            )
        if manifest is None and pose_seed is None:
            raise ValueError(
                "interface_seed requires pose_candidate_manifest or pose_seed "
                "for reproducible geometry"
            )
        resolved_topology["pose_candidate_manifest"] = (
            str(
                _resolve_existing_file(
                    manifest,
                    base=experiment_directory,
                    label="topology.pose_candidate_manifest",
                )
            )
            if manifest is not None
            else None
        )
        resolved_topology["pose_seed"] = (
            _nonnegative_integer(pose_seed, "topology.pose_seed")
            if pose_seed is not None
            else None
        )
        linker_length = topology.get("linker_length")
        resolved_topology["linker_length"] = (
            _nonnegative_integer(linker_length, "topology.linker_length")
            if linker_length is not None
            else None
        )
        resolved_topology["example_id"] = _safe_name(
            topology.get("example_id", name),
            "topology.example_id",
        )
    else:
        _reject_unknown(
            topology,
            {"kind", "config", "example_id"},
            "user_design topology",
        )
        resolved_topology["config"] = str(
            _resolve_existing_file(
                topology.get("config"),
                base=experiment_directory,
                label="topology.config",
            )
        )
        resolved_topology["example_id"] = _safe_name(
            topology.get("example_id", name),
            "topology.example_id",
        )

    sampling = raw.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be a mapping")
    _reject_unknown(
        sampling,
        {
            "preset",
            "timesteps",
            "seed",
            "low_memory_mode",
            "execution_backend",
            "neighbour_radius",
        },
        "sampling",
    )
    preset = str(sampling.get("preset", "exact_mosaic"))
    if preset not in SAMPLER_PRESETS:
        raise ValueError(
            "sampling.preset must be one of "
            f"{sorted(SAMPLER_PRESETS)}"
        )
    timesteps = _positive_integer(sampling.get("timesteps", 200), "sampling.timesteps")
    if not 2 <= timesteps <= 200:
        raise ValueError("sampling.timesteps must be between 2 and 200")
    backend = str(sampling.get("execution_backend", "explicit_all_copy"))
    if backend not in {"explicit_all_copy", "local_neighbourhood"}:
        raise ValueError(
            "sampling.execution_backend must be explicit_all_copy or "
            "local_neighbourhood"
        )
    if preset == "official_rfd3" and backend != "explicit_all_copy":
        raise ValueError(
            "sampling.preset=official_rfd3 requires "
            "execution_backend=explicit_all_copy"
        )
    resolved_sampling = {
        "preset": preset,
        "timesteps": timesteps,
        "seed": _nonnegative_integer(sampling.get("seed", 42), "sampling.seed"),
        "low_memory_mode": _boolean(
            sampling.get("low_memory_mode", True),
            "sampling.low_memory_mode",
        ),
        "execution_backend": backend,
        "neighbour_radius": _nonnegative_integer(
            sampling.get("neighbour_radius", 1),
            "sampling.neighbour_radius",
        ),
        "sampler": dict(SAMPLER_PRESETS[preset]),
    }

    resources = raw.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("resources must be a mapping")
    _reject_unknown(
        resources,
        {"profile", "walltime", "memory", "cpus", "partition"},
        "resources",
    )
    profile_path = _resolve_profile_path(
        profile_override or resources.get("profile"),
        experiment_directory=experiment_directory,
        repository_root=repository_root,
    )
    profile = _load_yaml(profile_path)
    _reject_unknown(
        profile,
        {
            "schema_version",
            "name",
            "slurm",
            "setup_commands",
            "checkpoint",
            "checkpoint_sha256",
            "foundry_checkpoint_dirs",
        },
        "execution profile",
    )
    if int(profile.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"profile schema_version must be {SCHEMA_VERSION}")
    slurm = profile.get("slurm")
    if not isinstance(slurm, dict):
        raise ValueError("profile.slurm must be a mapping")
    _reject_unknown(
        slurm,
        {"partition", "gres", "cpus", "memory", "walltime", "qos", "account"},
        "profile.slurm",
    )
    resolved_slurm = dict(slurm)
    for key in ("partition", "memory", "walltime"):
        if resources.get(key) is not None:
            resolved_slurm[key] = resources[key]
    if resources.get("cpus") is not None:
        resolved_slurm["cpus"] = _positive_integer(resources["cpus"], "resources.cpus")
    required_slurm = {"partition", "gres", "cpus", "memory", "walltime"}
    missing_slurm = required_slurm - set(resolved_slurm)
    if missing_slurm:
        raise ValueError(
            f"Execution profile lacks Slurm fields {sorted(missing_slurm)}"
        )
    resolved_slurm["cpus"] = _positive_integer(resolved_slurm["cpus"], "slurm.cpus")
    for key in ("partition", "gres", "memory", "walltime"):
        resolved_slurm[key] = _single_line(
            resolved_slurm[key],
            f"slurm.{key}",
        )
    for key in ("qos", "account"):
        if resolved_slurm.get(key) is not None:
            resolved_slurm[key] = _single_line(
                resolved_slurm[key],
                f"slurm.{key}",
            )

    setup_commands = profile.get("setup_commands")
    if not isinstance(setup_commands, list) or not setup_commands:
        raise ValueError("profile.setup_commands must be a non-empty list")
    if not all(isinstance(item, str) and item for item in setup_commands):
        raise ValueError("Every setup command must be a non-empty string")
    checkpoint = _resolve_path(
        profile.get("checkpoint"),
        base=profile_path.parent,
        label="profile.checkpoint",
    )
    foundry_dirs = _resolve_path(
        profile.get("foundry_checkpoint_dirs"),
        base=profile_path.parent,
        label="profile.foundry_checkpoint_dirs",
    )
    checkpoint_sha256 = _optional_sha256(
        profile.get("checkpoint_sha256"),
        "profile.checkpoint_sha256",
    )

    output = raw.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    _reject_unknown(output, {"root", "campaign"}, "output")
    resolved_output = {
        "root": str(
            _resolve_path(
                output.get("root"),
                base=experiment_directory,
                label="output.root",
            )
        ),
        "campaign": _safe_name(
            output.get("campaign", "rfd3_mosaic"),
            "output.campaign",
        ),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "topology": resolved_topology,
        "sampling": resolved_sampling,
        "resources": {
            "profile_name": _safe_name(profile.get("name"), "profile.name"),
            "slurm": resolved_slurm,
            "setup_commands": list(setup_commands),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "foundry_checkpoint_dirs": str(foundry_dirs),
        },
        "output": resolved_output,
        "project_directory": str(repository_root),
        "provenance": {
            "experiment_source": str(source_path),
            "experiment_sha256": _sha256(source_path),
            "profile_source": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "repository": collect_repository_provenance(repository_root),
            "foundry_compatibility": load_compatibility_manifest(
                repository_root
                / "configs"
                / "rfd3_mosaic"
                / "compatibility"
                / "foundry.yaml"
            ),
        },
    }
    return ResolvedExperiment(source_path, profile_path, payload)


def render_submission(
    experiment: ResolvedExperiment,
    *,
    output_directory: str | Path | None = None,
) -> Path:
    """Write a frozen config, provenance and a short auditable sbatch file."""

    if output_directory is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (
            experiment.run_root.parent
            / "_submissions"
            / experiment.name
            / timestamp
        )
    else:
        output = Path(output_directory).expanduser().resolve()
    resolved_payload = copy.deepcopy(experiment.payload)
    render_identity = _freeze_render_identity(resolved_payload)
    output.mkdir(parents=True, exist_ok=False)
    source_archive = output / "source_snapshot.tar.gz"
    try:
        render_identity["source_snapshot"] = create_source_snapshot(
            Path(resolved_payload["project_directory"]),
            source_archive,
        )
    except Exception:
        source_archive.unlink(missing_ok=True)
        output.rmdir()
        raise
    resolved_payload["provenance"]["render_identity"] = render_identity

    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False),
        encoding="utf-8",
    )
    provenance = copy.deepcopy(resolved_payload["provenance"])
    provenance["resolved_config_sha256"] = _sha256(resolved_path)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resources = resolved_payload["resources"]
    slurm = resources["slurm"]
    run_root = experiment.run_root
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={experiment.name}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --gres={slurm['gres']}",
        f"#SBATCH --cpus-per-task={slurm['cpus']}",
        f"#SBATCH --mem={slurm['memory']}",
        f"#SBATCH --time={slurm['walltime']}",
    ]
    if slurm.get("qos"):
        lines.append(f"#SBATCH --qos={slurm['qos']}")
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    lines.extend(
        [
            f"#SBATCH --output={output}/bootstrap-%j.out",
            f"#SBATCH --error={output}/bootstrap-%j.err",
            "",
            "set -euo pipefail",
            f"SOURCE_ARCHIVE={shlex.quote(str(source_archive))}",
            "SOURCE_SNAPSHOT_SHA256="
            + shlex.quote(
                render_identity["source_snapshot"]["archive"]["sha256"]
            ),
            f"RUN_ROOT={shlex.quote(str(run_root))}",
            'RUN_DIR="$RUN_ROOT/$SLURM_JOB_ID"',
            'if [[ -e "$RUN_DIR" ]]; then',
            '    echo "ERROR: refusing to reuse $RUN_DIR"',
            "    exit 2",
            "fi",
            'mkdir -p "$RUN_DIR"',
            'exec >"$RUN_DIR/slurm-$SLURM_JOB_ID.out" '
            '2>"$RUN_DIR/slurm-$SLURM_JOB_ID.err"',
            *resources["setup_commands"],
            'OBSERVED_SOURCE_SHA256=$(sha256sum "$SOURCE_ARCHIVE" '
            "| awk '{print $1}')",
            'if [[ "$OBSERVED_SOURCE_SHA256" != '
            '"$SOURCE_SNAPSHOT_SHA256" ]]; then',
            '    echo "ERROR: source snapshot SHA256 mismatch"',
            "    exit 3",
            "fi",
            'SOURCE_ROOT="$RUN_DIR/software"',
            'mkdir -p "$SOURCE_ROOT"',
            'tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE_ROOT"',
            'cd "$SOURCE_ROOT"',
            'export PYTHONPATH="$SOURCE_ROOT/src:'
            '$SOURCE_ROOT/models/rfd3/src:${PYTHONPATH:-}"',
            "export FOUNDRY_CHECKPOINT_DIRS="
            + shlex.quote(resources["foundry_checkpoint_dirs"]),
            "python -m rfd3_mosaic.experiment_worker "
            + "--resolved-config "
            + shlex.quote(str(resolved_path))
            + ' --run-dir "$RUN_DIR"'
            + ' --source-root "$SOURCE_ROOT"',
            "",
        ]
    )
    script_path = output / "generated_job.sbatch"
    script_path.write_text("\n".join(lines), encoding="utf-8")
    return script_path
