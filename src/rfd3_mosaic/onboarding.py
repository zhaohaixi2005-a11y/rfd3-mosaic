"""Site-independent first-run helpers for the public CLI."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from rfd3_mosaic.installation import (
    bundled_resource_path,
    source_repository_root,
)
from rfd3_mosaic.schema import UserDesignSpec

EXAMPLES = {
    "central-motif": Path("examples/rfd3_mosaic/simple_central_motif.yaml"),
    "supplied-interface": Path("examples/rfd3_mosaic/simple_interface_seed.yaml"),
    "supplied-interface-oligomer": Path(
        "examples/rfd3_mosaic/supplied_interface_higher_oligomer.yaml"
    ),
}
PUBLIC_PROFILES = {
    "local": Path("configs/rfd3_mosaic/execution/local.yaml"),
    "slurm-example": Path("configs/rfd3_mosaic/execution/slurm-example.yaml"),
}


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    ).strip("-._")
    if not cleaned:
        raise ValueError("Design name must contain at least one letter or digit")
    return cleaned[:64]


def _write_yaml(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def initialize_design(
    output: Path,
    *,
    task: str,
    input_path: Path,
    symmetry: str,
    name: str | None,
    profile: str,
    run_root: Path,
    motif_selector: str | None = None,
    side_a: str | None = None,
    side_b: str | None = None,
    interface_scaffold: str = "adjacent-linker",
    new_oligomer_interface: bool = False,
    sequence_conditioning: str = "fixed",
    redesign_motif_sidechains: bool = False,
    ligand_selectors: tuple[str, ...] = (),
    n_length: int = 35,
    c_length: int = 35,
    linker_minimum: int = 70,
    linker_maximum: int = 100,
    timesteps: int = 200,
    designs: int = 1,
    seed: int = 42,
    pose_radius_minimum: float | None = None,
    pose_radius_maximum: float | None = None,
    pose_axial_minimum: float = 0.0,
    pose_axial_maximum: float = 0.0,
    pose_orientation: str = "fixed",
    pose_maximum_tilt_deg: float = 30.0,
    pose_seed: int = 0,
    replicates_per_pose: int = 1,
    packing: str = "balanced",
    cavity: str = "auto",
    diversity: str = "medium",
    interface_area: str = "auto",
    component_motion: str = "locked",
    overwrite: bool = False,
) -> Path:
    """Write one short, schema-valid ordinary-user design."""

    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input structure does not exist: {source}")
    if timesteps < 2 or timesteps > 200:
        raise ValueError("timesteps must be between 2 and 200")
    if designs < 1 or designs > 10000:
        raise ValueError("designs must be between 1 and 10000")
    if replicates_per_pose < 1 or replicates_per_pose > designs:
        raise ValueError(
            "replicates_per_pose must satisfy 1 <= value <= designs"
        )
    if n_length < 1 or c_length < 1:
        raise ValueError("terminal generation lengths must be positive")
    if linker_minimum < 1 or linker_maximum < linker_minimum:
        raise ValueError("linker range must satisfy 1 <= minimum <= maximum")
    pose_radius_values = (pose_radius_minimum, pose_radius_maximum)
    if any(value is not None for value in pose_radius_values) and not all(
        value is not None for value in pose_radius_values
    ):
        raise ValueError(
            "pose radius requires both minimum and maximum"
        )
    if pose_radius_minimum is None and (
        pose_axial_minimum != 0.0
        or pose_axial_maximum != 0.0
        or pose_orientation != "fixed"
        or pose_seed != 0
    ):
        raise ValueError(
            "pose axial/orientation/seed options require an explicit pose "
            "radius interval"
        )
    if pose_orientation not in {
        "fixed",
        "uniform_so3",
        "principal_axis_cone",
    }:
        raise ValueError(
            "pose_orientation must be fixed, uniform_so3 or "
            "principal_axis_cone"
        )

    design_name = _safe_name(name or output.stem)
    common: dict[str, Any] = {
        "schema_version": 1,
        "name": design_name,
        "input": str(source),
        "symmetry": symmetry,
        "preferences": {
            "packing": packing,
            "cavity": cavity,
            "diversity": diversity,
            "interface_area": interface_area,
            "component_motion": component_motion,
        },
    }
    if task == "central-motif":
        if not motif_selector:
            raise ValueError("central-motif requires --motif-selector")
        common.update(
            {
                "task": "create_symmetric_interface",
                "generation": [
                    {
                        "kind": "terminal",
                        "anchor": motif_selector,
                        "terminus": "n",
                        "length": n_length,
                    },
                    {
                        "kind": "terminal",
                        "anchor": motif_selector,
                        "terminus": "c",
                        "length": c_length,
                    },
                ],
                "constraints": [{"kind": "fixed_xyz", "selector": motif_selector}],
            }
        )
    elif task == "supplied-interface":
        if not side_a or not side_b:
            raise ValueError("supplied-interface requires both --side-a and --side-b")
        if not symmetry.startswith("C") or not symmetry[1:].isdigit():
            raise ValueError(
                "The short supplied-interface initializer supports cyclic Cn "
                "symmetry only; use an explicit design graph for non-cyclic "
                "copy relations"
            )
        if interface_scaffold == "adjacent-linker":
            generation = [
                {
                    "kind": "between",
                    "from_selector": side_b,
                    "to_selector": side_a,
                    "orbit_offset": "nearest_adjacent",
                    "length": {
                        "minimum": linker_minimum,
                        "maximum": linker_maximum,
                    },
                }
            ]
        elif interface_scaffold == "terminal-extensions":
            generation = [
                {
                    "kind": "terminal",
                    "anchor": selector,
                    "terminus": terminus,
                    "length": n_length if terminus == "n" else c_length,
                }
                for selector in (side_a, side_b)
                for terminus in ("n", "c")
            ]
        else:
            raise ValueError(
                "interface_scaffold must be adjacent-linker or "
                "terminal-extensions"
            )
        conditioning: dict[str, Any] = {}
        if sequence_conditioning != "fixed":
            if sequence_conditioning not in {"masked", "glycine"}:
                raise ValueError(
                    "sequence_conditioning must be fixed, masked or glycine"
                )
            conditioning["sequence"] = [
                {"selector": selector, "mode": sequence_conditioning}
                for selector in (side_a, side_b)
            ]
        if ligand_selectors:
            conditioning["ligands"] = [
                {
                    "selector": selector,
                    "coupling_group": "supplied_interface",
                }
                for selector in ligand_selectors
            ]
        if redesign_motif_sidechains:
            conditioning["redesign_motif_sidechains"] = True
        sampling_overrides: dict[str, Any] = {}
        if new_oligomer_interface:
            sampling_overrides["scaffold_packing"] = "symmetric_generated"
        common.update(
            {
                "task": "preserve_supplied_geometry",
                "generation": generation,
                "constraints": [
                    {
                        "kind": "fixed_xyz",
                        "selector": side_a,
                        "coupling_group": "supplied_interface",
                    },
                    {
                        "kind": "fixed_xyz",
                        "selector": side_b,
                        "coupling_group": "supplied_interface",
                    },
                ],
                **({"conditioning": conditioning} if conditioning else {}),
            }
        )
    else:
        raise ValueError(f"Unknown initialization task: {task!r}")

    sampling: dict[str, Any] = {
        "timesteps": timesteps,
        "designs": designs,
        "replicates_per_pose": replicates_per_pose,
        "seed": seed,
        **(
            sampling_overrides
            if task == "supplied-interface"
            else {}
        ),
    }
    if pose_radius_minimum is not None:
        orientation: dict[str, Any] = {"method": pose_orientation}
        if pose_orientation == "principal_axis_cone":
            orientation["maximum_tilt_deg"] = pose_maximum_tilt_deg
        sampling["initial_pose"] = {
            "radius": {
                "minimum": pose_radius_minimum,
                "maximum": pose_radius_maximum,
            },
            "axial_offset": {
                "minimum": pose_axial_minimum,
                "maximum": pose_axial_maximum,
            },
            "orientation": orientation,
            "seed": pose_seed,
        }

    common.update(
        {
            "sampling": sampling,
            "resources": {"profile": profile},
            "output": {
                "root": str(run_root.expanduser().resolve()),
                "campaign": design_name,
            },
        }
    )
    UserDesignSpec.model_validate(common)
    return _write_yaml(output, common, overwrite=overwrite)


def available_examples() -> list[dict[str, str]]:
    """Return maintained example identifiers and resolved paths."""

    return [
        {"id": example_id, "path": str(bundled_resource_path(relative))}
        for example_id, relative in EXAMPLES.items()
    ]


def copy_example(example_id: str, output: Path, *, overwrite: bool) -> Path:
    """Copy a maintained example as a portable, immediately editable YAML."""

    if example_id not in EXAMPLES:
        raise ValueError(
            f"Unknown example {example_id!r}; choose one of {sorted(EXAMPLES)}"
        )
    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    source = bundled_resource_path(EXAMPLES[example_id])
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bundled example is not a YAML mapping: {source}")
    structure = Path(str(payload["input"])).expanduser()
    if not structure.is_absolute():
        structure = source.parent / structure
    payload["input"] = str(structure.resolve())
    payload.setdefault("resources", {})["profile"] = "local"
    payload["output"] = {
        "root": str((destination.parent / "runs").resolve()),
        "campaign": str(payload.get("name", example_id)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def available_profiles() -> list[dict[str, str]]:
    """Describe public profiles and optional source-checkout site profiles."""

    result = [
        {
            "id": profile_id,
            "scope": "public",
            "path": str(bundled_resource_path(relative)),
        }
        for profile_id, relative in PUBLIC_PROFILES.items()
    ]
    repository = source_repository_root()
    if repository is not None:
        site_root = repository / "configs" / "rfd3_mosaic" / "sites"
        for site in sorted(site_root.glob("*")):
            if not site.is_dir():
                continue
            for profile in sorted(site.glob("*.yaml")):
                result.append(
                    {
                        "id": profile.stem,
                        "scope": f"site:{site.name}",
                        "path": str(profile.resolve()),
                    }
                )
    return result


def copy_slurm_profile(output: Path, *, overwrite: bool) -> Path:
    """Copy the generic scheduler profile for site customization."""

    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        bundled_resource_path(PUBLIC_PROFILES["slurm-example"]),
        destination,
    )
    return destination
