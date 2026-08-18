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
    n_length: int = 35,
    c_length: int = 35,
    linker_minimum: int = 70,
    linker_maximum: int = 100,
    timesteps: int = 200,
    designs: int = 1,
    seed: int = 42,
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
    if n_length < 1 or c_length < 1:
        raise ValueError("terminal generation lengths must be positive")
    if linker_minimum < 1 or linker_maximum < linker_minimum:
        raise ValueError("linker range must satisfy 1 <= minimum <= maximum")

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
        if component_motion != "locked":
            raise ValueError(
                "supplied-interface preserves the complete supplied geometry "
                "and therefore requires --component-motion locked"
            )
        common.update(
            {
                "task": "preserve_supplied_geometry",
                "generation": [
                    {
                        "kind": "between",
                        "from_selector": side_b,
                        "to_selector": side_a,
                        "orbit_offset": 1,
                        "length": {
                            "minimum": linker_minimum,
                            "maximum": linker_maximum,
                        },
                    }
                ],
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
            }
        )
    else:
        raise ValueError(f"Unknown initialization task: {task!r}")

    common.update(
        {
            "sampling": {
                "timesteps": timesteps,
                "designs": designs,
                "seed": seed,
            },
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
