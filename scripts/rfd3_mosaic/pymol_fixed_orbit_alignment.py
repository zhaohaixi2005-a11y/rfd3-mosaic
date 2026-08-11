"""PyMOL helper for reproducing the fixed-constraint orbit audit alignment.

Load the run's ``presymmetrized_input.cif`` as ``ref`` and its result
structure as ``design``, then run::

    run scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
    mosaic_align_fixed ref, design, /absolute/path/to/run

The command creates ``design_fixed_aligned`` and leaves ``design`` unchanged.
It uses the compiled residue map and declared symmetry registry instead of
asking PyMOL to infer a 2N-fragment-to-N-chain correspondence from sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from pymol import cmd


_SELECTOR = re.compile(r"^([^0-9,+-]+)([0-9]+)-([0-9]+)$")
_COMPONENT = re.compile(r"^([^0-9]+)([0-9]+)$")


def _single_example(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Compiled input must contain exactly one example")
    example = next(iter(payload.values()))
    if not isinstance(example, dict):
        raise ValueError("Compiled RFD3 example must be an object")
    return example


def _find_run_files(run_dir: Path) -> tuple[Path, Path]:
    compiled_candidates = [
        run_dir / "input" / "rfd3_input.json",
        run_dir / "rfd3_input.json",
    ]
    compiled_input = next(
        (path for path in compiled_candidates if path.is_file()),
        None,
    )
    if compiled_input is None:
        raise FileNotFoundError(
            f"No compiled rfd3_input.json found below {run_dir}"
        )
    result_paths = sorted(run_dir.glob("*model_0.json"))
    if len(result_paths) != 1:
        raise ValueError(
            "Run directory must contain exactly one *model_0.json; "
            f"found {len(result_paths)}"
        )
    return compiled_input, result_paths[0]


def _reference_structure(example: dict, compiled_path: Path) -> Path:
    path = Path(str(example["input"]))
    if not path.is_absolute():
        path = (compiled_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference structure does not exist: {path}")
    return path


def _result_structure(result_path: Path) -> Path:
    stem = result_path.with_suffix("")
    for suffix in (".cif.gz", ".cif", ".pdb"):
        candidate = Path(f"{stem}{suffix}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No result CIF/PDB exists beside {result_path}"
    )


def _parse_selector(value: str) -> tuple[str, int, int]:
    match = _SELECTOR.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Fixed selector {value!r} is not a contiguous chain range"
        )
    chain, start, end = match.groups()
    return chain, int(start), int(end)


def _parse_component(value: str) -> tuple[str, int]:
    match = _COMPONENT.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid residue component {value!r}")
    chain, residue = match.groups()
    return chain, int(residue)


def _residue_number(value: str) -> int:
    match = re.match(r"^-?[0-9]+", value)
    if match is None:
        raise ValueError(f"Cannot parse PyMOL residue identifier {value!r}")
    return int(match.group())


def _is_heavy(atom) -> bool:
    symbol = str(getattr(atom, "symbol", "")).upper()
    name = str(atom.name).lstrip("0123456789").upper()
    return not (symbol.startswith("H") or name.startswith("H"))


def _atom_lookup(
    object_name: str,
    *,
    label_order: bool = False,
) -> dict[tuple[str, int, str], np.ndarray]:
    atoms = [
        atom
        for atom in cmd.get_model(object_name).atom
        if _is_heavy(atom)
    ]
    label_residue: dict[tuple[str, int], int] = {}
    if label_order:
        next_index: dict[str, int] = {}
        for atom in atoms:
            chain = str(atom.chain)
            auth_residue = _residue_number(str(atom.resi))
            identity = (chain, auth_residue)
            if identity not in label_residue:
                next_index[chain] = next_index.get(chain, 0) + 1
                label_residue[identity] = next_index[chain]

    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for atom in atoms:
        chain = str(atom.chain)
        auth_residue = _residue_number(str(atom.resi))
        key = (
            chain,
            (
                label_residue[(chain, auth_residue)]
                if label_order
                else auth_residue
            ),
            str(atom.name).upper(),
        )
        if key in lookup:
            raise ValueError(
                f"Object {object_name!r} has duplicate heavy atom {key}"
            )
        lookup[key] = np.asarray(atom.coord, dtype=float)
    return lookup


def _chain_sort_key(chain: str) -> tuple[int, ...]:
    """Sort A..Z, AA..AZ in compiler chain-allocation order."""

    if chain and chain.isalpha() and chain.isupper():
        value = 0
        for character in chain:
            value = value * 26 + ord(character) - ord("A") + 1
        return (0, value)
    return (1, *chain.encode("utf-8"))


def _runtime_action_index(value, registry_order: list[str]) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean symmetry action {value!r}")
    if isinstance(value, int):
        index = value
    elif isinstance(value, str) and value in registry_order:
        index = registry_order.index(value)
    else:
        try:
            index = int(str(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unknown runtime symmetry action {value!r}"
            ) from error
    if not 0 <= index < len(registry_order):
        raise ValueError(
            f"Runtime symmetry action {index} is outside the declared "
            f"registry of size {len(registry_order)}"
        )
    return index


def _mapped_output_coordinate(
    source_key: tuple[str, int, str],
    *,
    action_index: int,
    index_map: dict,
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    output_chains: list[str],
    asu_chain_count: int,
) -> np.ndarray:
    source_chain, source_residue, atom_name = source_key
    destination = index_map.get(f"{source_chain}{source_residue}")
    if destination is None:
        raise ValueError(
            f"Result JSON does not map fixed residue {source_chain}{source_residue}"
        )
    master_chain, output_residue = _parse_component(str(destination))
    try:
        master_position = output_chains.index(master_chain)
    except ValueError as error:
        raise ValueError(
            f"Mapped master output chain {master_chain!r} is not present in "
            "the loaded design"
        ) from error
    asu_chain_index = master_position % asu_chain_count
    output_index = action_index * asu_chain_count + asu_chain_index
    output_chain = output_chains[output_index]
    try:
        return output_lookup[(output_chain, output_residue, atom_name)]
    except KeyError as error:
        raise ValueError(
            "Output object is missing mapped fixed atom "
            f"{(output_chain, output_residue, atom_name)!r}"
        ) from error


def _fixed_alignment_coordinates(
    *,
    example: dict,
    result: dict,
    source_lookup: dict[tuple[str, int, str], np.ndarray],
    output_lookup: dict[tuple[str, int, str], np.ndarray],
    output_chains: list[str],
    asu_chain_count: int,
    registry_order: list[str],
    registry_matrices: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build provenance-exact fixed atom pairs across all ASU chains.

    A physical constraint group can span several output chains and can use a
    different symmetry action for each source fragment.  The compiler's
    ``motif_constraint_groups`` plus ``diffused_index_map`` are therefore the
    authority; chain count and sequence matching are not.
    """

    extra = example.get("extra") or {}
    index_map = result.get("diffused_index_map") or {}
    if not isinstance(index_map, dict) or not index_map:
        raise ValueError("Result JSON contains no fixed-residue index mapping")
    expected: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    orbits = extra.get("motif_constraint_orbits")
    groups = extra.get("motif_constraint_groups")
    fixed_orbits = [
        orbit
        for orbit in orbits or []
        if isinstance(orbit, dict)
        and orbit.get("mobility_mode", "fixed") == "fixed"
    ]

    if fixed_orbits and isinstance(groups, list) and groups:
        for orbit in fixed_orbits:
            orbit_id = str(orbit.get("constraint_orbit_id", ""))
            residue_ids = {
                _parse_component(str(value))
                for value in orbit.get("source_components", [])
            }
            orbit_keys = {
                key for key in source_lookup if (key[0], key[1]) in residue_ids
            }
            if not orbit_keys:
                raise ValueError(
                    f"Fixed constraint orbit {orbit_id!r} matched no loaded "
                    "reference atoms"
                )
            selected_groups = [
                group
                for group in groups
                if isinstance(group, dict)
                and str(group.get("constraint_orbit_id", "")) == orbit_id
            ]
            group_ids = orbit.get("group_ids")
            if isinstance(group_ids, list) and group_ids:
                by_id = {
                    str(group.get("group_id")): group
                    for group in selected_groups
                }
                missing = [
                    str(group_id)
                    for group_id in group_ids
                    if str(group_id) not in by_id
                ]
                if missing:
                    raise ValueError(
                        f"Fixed constraint orbit {orbit_id!r} is missing "
                        f"runtime groups {missing}"
                    )
                selected_groups = [
                    by_id[str(group_id)] for group_id in group_ids
                ]
            if not selected_groups:
                raise ValueError(
                    f"Fixed constraint orbit {orbit_id!r} has no runtime groups"
                )

            for group in selected_groups:
                members = group.get("members")
                if not isinstance(members, list) or not members:
                    raise ValueError(
                        f"Runtime motif group {group.get('group_id')!r} has "
                        "no members"
                    )
                group_keys: set[tuple[str, int, str]] = set()
                for member in members:
                    components = member.get("src_components")
                    if not isinstance(components, list) or not components:
                        raise ValueError(
                            "Runtime motif group member declares no source "
                            "components"
                        )
                    member_residues = {
                        _parse_component(str(value)) for value in components
                    }
                    member_keys = sorted(
                        key
                        for key in orbit_keys
                        if (key[0], key[1]) in member_residues
                    )
                    if not member_keys:
                        raise ValueError(
                            "Runtime motif group member matched no loaded "
                            f"reference atoms: {components!r}"
                        )
                    overlap = group_keys.intersection(member_keys)
                    if overlap:
                        raise ValueError(
                            "Runtime motif group members overlap on fixed "
                            f"atoms: {sorted(overlap)[:5]}"
                        )
                    group_keys.update(member_keys)
                    action_index = _runtime_action_index(
                        member.get("sym_transform_id"),
                        registry_order,
                    )
                    matrix = np.asarray(
                        registry_matrices[registry_order[action_index]],
                        dtype=float,
                    )
                    for key in member_keys:
                        coordinate = source_lookup[key]
                        expected.append(
                            coordinate @ matrix[:3, :3].T + matrix[:3, 3]
                        )
                        observed.append(
                            _mapped_output_coordinate(
                                key,
                                action_index=action_index,
                                index_map=index_map,
                                output_lookup=output_lookup,
                                output_chains=output_chains,
                                asu_chain_count=asu_chain_count,
                            )
                        )
                if group_keys != orbit_keys:
                    missing = orbit_keys - group_keys
                    raise ValueError(
                        f"Runtime motif group {group.get('group_id')!r} does "
                        "not cover its complete fixed orbit: "
                        f"missing={sorted(missing)[:5]}"
                    )
    else:
        atom_keys = sorted(source_lookup)
        for action_index, transform_id in enumerate(registry_order):
            matrix = np.asarray(
                registry_matrices[transform_id],
                dtype=float,
            )
            for key in atom_keys:
                coordinate = source_lookup[key]
                expected.append(
                    coordinate @ matrix[:3, :3].T + matrix[:3, 3]
                )
                observed.append(
                    _mapped_output_coordinate(
                        key,
                        action_index=action_index,
                        index_map=index_map,
                        output_lookup=output_lookup,
                        output_chains=output_chains,
                        asu_chain_count=asu_chain_count,
                    )
                )
    if not expected:
        raise ValueError("No globally fixed atoms were available for alignment")
    return np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _infer_loaded_objects(
    reference: str,
    design: str,
) -> tuple[str, str]:
    objects = [
        name
        for name in cmd.get_names("objects", enabled_only=0)
        if not name.endswith("_aligned")
    ]
    if reference and design:
        return reference, design
    if reference or design:
        raise ValueError(
            "Provide both reference and design, or omit both for automatic "
            "detection"
        )
    if len(objects) != 2:
        raise ValueError(
            "Automatic mosaic_align requires exactly two loaded raw "
            f"objects; found {objects}. Pass reference and design explicitly."
        )

    reference_candidates = [
        name
        for name in objects
        if "presymmetrized" in name.lower()
        or name.lower().endswith("_ref")
        or name.lower() == "ref"
    ]
    design_candidates = [
        name
        for name in objects
        if "model_0" in name.lower()
        or "design" in name.lower()
    ]
    if len(reference_candidates) == 1 and len(design_candidates) == 1:
        if reference_candidates[0] != design_candidates[0]:
            return reference_candidates[0], design_candidates[0]

    chain_counts = {
        name: len(cmd.get_chains(name))
        for name in objects
    }
    ordered = sorted(objects, key=lambda name: chain_counts[name])
    if chain_counts[ordered[0]] == chain_counts[ordered[1]]:
        raise ValueError(
            "Cannot infer reference/design from equally sized objects. "
            "Use: mosaic_align reference_object, design_object"
        )
    return ordered[1], ordered[0]


def _find_run_for_design(design: str, search_root: Path) -> Path:
    if not search_root.is_dir():
        raise FileNotFoundError(
            f"Mosaic alignment search root does not exist: {search_root}"
        )
    design_key = _normalized_name(design)
    candidates = []
    for path in search_root.rglob("*model_0.json"):
        if _normalized_name(path.stem) != design_key:
            continue
        try:
            _find_run_files(path.parent)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find run metadata for PyMOL object {design!r} "
            f"below {search_root}. Use mosaic_align with an explicit run_dir."
        )
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            f"Several local runs match {design!r}: "
            + ", ".join(str(path) for path in unique)
            + ". Pass run_dir explicitly."
        )
    return unique[0]


def _fixed_source_residues(example: dict) -> set[tuple[str, int]]:
    extra = example.get("extra") or {}
    selectors = extra.get("fixed_constraint_selectors")
    if selectors is None:
        selectors = extra.get("probe_fixed_selectors")
    if selectors is None:
        single = extra.get("probe_fixed_selector")
        selectors = [single] if single is not None else None
    if selectors is None:
        selectors = list((example.get("select_fixed_atoms") or {}).keys())
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("Compiled input declares no fixed selectors")

    residues = {
        (chain, residue)
        for selector in selectors
        for chain, start, end in [_parse_selector(str(selector))]
        for residue in range(start, end + 1)
    }

    orbits = extra.get("motif_constraint_orbits")
    if not isinstance(orbits, list) or not orbits:
        return residues
    fixed_components = [
        orbit
        for orbit in orbits
        if isinstance(orbit, dict)
        and orbit.get("mobility_mode", "fixed") == "fixed"
    ]
    if not fixed_components:
        raise ValueError(
            "This run has no globally fixed orbit; independently mobile "
            "components require component-specific visualization"
        )
    declared = {
        _parse_component(str(component))
        for orbit in fixed_components
        for component in orbit.get("source_components", [])
    }
    return residues.intersection(declared) if declared else residues


def _kabsch(
    moving: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    moving_zero = moving - moving_center
    reference_zero = reference - reference_center
    left, _, right_t = np.linalg.svd(moving_zero.T @ reference_zero)
    correction = np.eye(3)
    correction[-1, -1] = (
        -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
    )
    rotation = left @ correction @ right_t
    translation = reference_center - moving_center @ rotation
    aligned = moving @ rotation + translation
    rmsd = float(
        np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1)))
    )
    return rotation, translation, rmsd


def mosaic_align_fixed(
    reference: str = "ref",
    design: str = "design",
    run_dir: str = ".",
    output_object: str = "design_fixed_aligned",
) -> None:
    """Align one output to its complete fixed orbit using run provenance."""

    objects = set(cmd.get_names("objects", enabled_only=0))
    for required in (reference, design):
        if required not in objects:
            raise ValueError(f"PyMOL object {required!r} is not loaded")
    if output_object in objects:
        raise ValueError(
            f"Output object {output_object!r} already exists; rename or "
            "delete it explicitly before rerunning"
        )

    compiled_path, result_path = _find_run_files(
        Path(run_dir).expanduser().resolve()
    )
    example = _single_example(compiled_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_residues = _fixed_source_residues(example)
    source_lookup = {
        key: coordinate
        for key, coordinate in _atom_lookup(
            reference,
            label_order=True,
        ).items()
        if (key[0], key[1]) in source_residues
    }
    if not source_lookup:
        raise ValueError(
            "The loaded reference object contains none of the compiled "
            "fixed atoms"
        )
    output_lookup = _atom_lookup(design, label_order=True)
    output_chains = sorted(
        {chain for chain, _, _ in output_lookup},
        key=_chain_sort_key,
    )

    extra = example.get("extra") or {}
    order = extra.get("registry_transform_order")
    matrices = extra.get("registry_transform_matrices")
    multiplicity = int(extra.get("symmetry_multiplicity", 0))
    if not isinstance(order, list) or not isinstance(matrices, dict):
        raise ValueError("Compiled input lacks the symmetry registry")
    if (
        len(order) != multiplicity
        or not output_chains
        or len(output_chains) % multiplicity != 0
    ):
        raise ValueError(
            "Symmetry multiplicity mismatch: "
            f"transforms={len(order)}, output_chains={len(output_chains)}, "
            f"declared={multiplicity}"
        )
    asu_chain_count = len(output_chains) // multiplicity
    expected, observed = _fixed_alignment_coordinates(
        example=example,
        result=result,
        source_lookup=source_lookup,
        output_lookup=output_lookup,
        output_chains=output_chains,
        asu_chain_count=asu_chain_count,
        registry_order=[str(value) for value in order],
        registry_matrices=matrices,
    )
    rotation, translation, rmsd = _kabsch(observed, expected)

    cmd.create(output_object, design)
    coordinates = np.asarray(cmd.get_coords(output_object), dtype=float)
    cmd.load_coords(coordinates @ rotation + translation, output_object)
    print(
        f"mosaic_align_fixed: RMSD={rmsd:.6f} A "
        f"({len(observed)} fixed heavy atoms); created {output_object!r}"
    )


def mosaic_load_run(
    run_dir: str,
    prefix: str = "mosaic",
    style: str = "1",
) -> None:
    """Load, provenance-align and optionally style one complete Mosaic run."""

    run_path = Path(run_dir).expanduser().resolve()
    compiled_path, result_path = _find_run_files(run_path)
    example = _single_example(compiled_path)
    reference_path = _reference_structure(example, compiled_path)
    design_path = _result_structure(result_path)

    reference_object = f"{prefix}_ref"
    design_object = f"{prefix}_design_raw"
    aligned_object = f"{prefix}_design_aligned"
    existing = set(cmd.get_names("objects", enabled_only=0))
    collisions = sorted(
        existing.intersection(
            {reference_object, design_object, aligned_object}
        )
    )
    if collisions:
        raise ValueError(
            "Refusing to overwrite existing PyMOL objects: "
            + ", ".join(collisions)
        )

    cmd.load(str(reference_path), reference_object)
    cmd.load(str(design_path), design_object)
    mosaic_align_fixed(
        reference_object,
        design_object,
        str(run_path),
        aligned_object,
    )

    if str(style).lower() not in {"0", "false", "no", "off"}:
        cmd.hide("everything", reference_object)
        cmd.hide("everything", design_object)
        cmd.hide("everything", aligned_object)
        cmd.show("cartoon", reference_object)
        cmd.show("cartoon", aligned_object)
        cmd.color("cyan", reference_object)
        cmd.color("magenta", aligned_object)
        cmd.set("cartoon_transparency", 0.6, reference_object)
        cmd.disable(design_object)
        cmd.zoom(f"{reference_object} or {aligned_object}")

    print(f"reference: {reference_path}")
    print(f"raw design: {design_path}")
    print(
        "objects: "
        f"{reference_object}, {design_object}, {aligned_object}"
    )


def mosaic_align(
    reference: str = "",
    design: str = "",
    run_dir: str = "",
    output_object: str = "mosaic_aligned",
    search_root: str = "/home/haixi/Documents/template",
) -> None:
    """Align two already loaded objects using their Mosaic run metadata.

    With exactly two raw objects loaded, ``mosaic_align`` needs no arguments.
    Explicit fallback::

        mosaic_align ref, design, /absolute/path/to/run
    """

    reference_object, design_object = _infer_loaded_objects(
        str(reference).strip(),
        str(design).strip(),
    )
    run_path = (
        Path(run_dir).expanduser().resolve()
        if str(run_dir).strip()
        else _find_run_for_design(
            design_object,
            Path(search_root).expanduser().resolve(),
        )
    )
    mosaic_align_fixed(
        reference_object,
        design_object,
        str(run_path),
        output_object,
    )
    cmd.hide("everything", reference_object)
    cmd.hide("everything", output_object)
    cmd.show("cartoon", reference_object)
    cmd.show("cartoon", output_object)
    cmd.color("cyan", reference_object)
    cmd.color("magenta", output_object)
    cmd.set("cartoon_transparency", 0.6, reference_object)
    cmd.disable(design_object)
    cmd.zoom(f"{reference_object} or {output_object}")
    print(f"mosaic_align run: {run_path}")


cmd.extend("mosaic_align_fixed", mosaic_align_fixed)
cmd.extend("mosaic_load_run", mosaic_load_run)
cmd.extend("mosaic_align", mosaic_align)
print(
    "Mosaic fixed-orbit alignment loaded: drag exactly two run structures "
    "and type mosaic_align"
)
