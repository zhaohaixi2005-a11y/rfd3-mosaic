"""Build a controlled symmetric central-motif RFD3 probe input."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)-([0-9]+)$")


def _load_single_example(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Template input must contain exactly one example")
    example_id, example = next(iter(payload.items()))
    if not isinstance(example, dict):
        raise ValueError("Template example must be a JSON object")
    return str(example_id), example


def _parse_selector(selector: str) -> tuple[str, int, int]:
    match = _SELECTOR.fullmatch(selector)
    if match is None:
        raise ValueError(
            "fixed selector must be one contiguous range such as B1-31"
        )
    chain, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError("fixed selector range is reversed")
    return chain, start, end


def _symmetry_multiplicity(symmetry_id: str) -> int:
    if len(symmetry_id) < 2 or not symmetry_id[1:].isdigit():
        raise ValueError(f"Unsupported symmetry ID {symmetry_id!r}")
    order = int(symmetry_id[1:])
    if order < 2:
        raise ValueError("Symmetry order must be at least two")
    prefix = symmetry_id[0].upper()
    if prefix == "C":
        return order
    if prefix == "D":
        return 2 * order
    raise ValueError(f"Unsupported symmetry ID {symmetry_id!r}")


def build_central_motif_probe_input(
    template_input: str | Path,
    output_directory: str | Path,
    *,
    fixed_selector: str,
    n_terminal_length: int = 35,
    c_terminal_length: int = 35,
    example_id: str = "central_motif_c3_probe",
    use_declared_frames: bool = True,
) -> Path:
    """Convert one symmetric adapter example into a central-motif probe.

    The source structure and symmetry definition are preserved.  Only one
    indexed motif range is retained, with generated residues placed before
    and after it.  Each symmetry copy is represented as one ``fixed_motif``
    constraint group in a single fixed orbit.
    """

    if n_terminal_length < 1 or c_terminal_length < 1:
        raise ValueError("Both terminal diffusion lengths must be positive")
    template_path = Path(template_input).resolve()
    _, template = _load_single_example(template_path)
    chain, start, end = _parse_selector(fixed_selector)

    symmetry = dict(template.get("symmetry") or {})
    symmetry_id = str(symmetry.get("id") or "")
    multiplicity = _symmetry_multiplicity(symmetry_id)
    symmetry["is_symmetric_motif"] = True
    symmetry["use_declared_frames"] = bool(use_declared_frames)
    template_extra = dict(template.get("extra") or {})
    if use_declared_frames:
        declared_order = template_extra.get("registry_transform_order")
        declared_matrices = template_extra.get(
            "registry_transform_matrices"
        )
        if not declared_order or not declared_matrices:
            raise ValueError(
                "Declared-frame central motif input requires the template "
                "registry_transform_order and registry_transform_matrices"
            )
        symmetry["declared_transform_order"] = list(declared_order)
        symmetry["declared_transform_matrices"] = dict(declared_matrices)

    source_value = template.get("input")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("Template example has no input structure")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = template_path.parent / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Template structure does not exist: {source_path}")

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied_structure = output / source_path.name
    if copied_structure != source_path:
        shutil.copy2(source_path, copied_structure)

    components = [f"{chain}{residue}" for residue in range(start, end + 1)]
    group_ids = [
        f"central_motif@{symmetry_id}[{transform_id}]"
        for transform_id in range(multiplicity)
    ]
    groups = [
        {
            "group_id": group_id,
            "constraint_kind": "fixed_motif",
            "orbit_id": "central_motif_orbit",
            "members": [
                {
                    "role": "motif",
                    "source_fragment_id": "central_motif",
                    "src_components": components,
                    "sym_transform_id": transform_id,
                }
            ],
        }
        for transform_id, group_id in enumerate(group_ids)
    ]
    orbit = {
        "orbit_id": "central_motif_orbit",
        "group_ids": group_ids,
        "master_group_id": group_ids[0],
        "group_transform_ids": list(range(multiplicity)),
        "mobility_mode": "fixed",
        "max_translation": 0.0,
        "max_rotation_deg": 0.0,
    }

    extra = template_extra
    extra.update(
        {
            "compiler": "rfd3_mosaic.central_motif_probe",
            "probe_topology": "central_motif_bidirectional_growth",
            "probe_template_input": str(template_path),
            "probe_fixed_selector": fixed_selector,
            "probe_n_terminal_length": n_terminal_length,
            "probe_c_terminal_length": c_terminal_length,
            "symmetry_multiplicity": multiplicity,
            "motif_constraint_groups": groups,
            "motif_constraint_orbits": [orbit],
        }
    )
    example = {
        "dialect": int(template.get("dialect", 2)),
        "input": copied_structure.name,
        "contig": (
            f"{n_terminal_length}-{n_terminal_length},"
            f"{fixed_selector},"
            f"{c_terminal_length}-{c_terminal_length}"
        ),
        "select_fixed_atoms": {fixed_selector: "ALL"},
        "redesign_motif_sidechains": False,
        "is_non_loopy": bool(template.get("is_non_loopy", True)),
        "symmetry": symmetry,
        "extra": extra,
    }
    output_path = output / "rfd3_input.json"
    output_path.write_text(
        json.dumps({example_id: example}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a symmetric RFD3 probe with one fixed central motif and "
            "generated N/C-terminal regions."
        )
    )
    parser.add_argument("--template-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fixed-selector", required=True)
    parser.add_argument("--n-terminal-length", type=int, default=35)
    parser.add_argument("--c-terminal-length", type=int, default=35)
    parser.add_argument("--example-id", default="central_motif_c3_probe")
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--use-declared-frames",
        action="store_true",
        dest="use_declared_frames",
    )
    frame_group.add_argument(
        "--recover-input-frames",
        action="store_false",
        dest="use_declared_frames",
    )
    parser.set_defaults(use_declared_frames=True)
    arguments = parser.parse_args()
    output_path = build_central_motif_probe_input(
        arguments.template_input,
        arguments.output_dir,
        fixed_selector=arguments.fixed_selector,
        n_terminal_length=arguments.n_terminal_length,
        c_terminal_length=arguments.c_terminal_length,
        example_id=arguments.example_id,
        use_declared_frames=arguments.use_declared_frames,
    )
    print(f"Central-motif probe input: {output_path}")


if __name__ == "__main__":
    main()
