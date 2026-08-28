"""PyMOL helpers for visually aligning Mosaic reference and design CIFs.

Drag ``presymmetrized_input.cif`` and one result ``*model_0.cif`` into PyMOL,
then run::

    run scripts/rfd3_mosaic/pymol_fixed_orbit_alignment.py
    mosaic_align

The routine command needs only those two loaded structures.  It locates shared
motif fragments by sequence and selects the symmetry-copy correspondence with
one consistent three-dimensional transform.  It creates ``mosaic_aligned``
and leaves both input objects unchanged.  ``mosaic_align_fixed`` remains
available when a provenance-exact audit alignment is specifically required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    preexpanded: bool = False,
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

    # Mixed stabilizer/coset designs are materialized before RFD3.  Their
    # reference chains already occupy the declared physical group frames, so
    # applying a registry transform here a second time is incorrect.  The
    # result metadata gives a direct residue mapping for every fixed source
    # atom and is the complete authority for this layout.
    if preexpanded:
        for key in sorted(source_lookup):
            source_chain, source_residue, atom_name = key
            destination = index_map.get(f"{source_chain}{source_residue}")
            if destination is None:
                raise ValueError(
                    "Result JSON does not map preexpanded fixed residue "
                    f"{source_chain}{source_residue}"
                )
            output_chain, output_residue = _parse_component(str(destination))
            output_key = (output_chain, output_residue, atom_name)
            try:
                output_coordinate = output_lookup[output_key]
            except KeyError as error:
                raise ValueError(
                    "Output object is missing mapped preexpanded fixed atom "
                    f"{output_key!r}"
                ) from error
            expected.append(source_lookup[key])
            observed.append(output_coordinate)
        if not expected:
            raise ValueError(
                "No preexpanded globally fixed atoms were available for "
                "alignment"
            )
        return np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)

    orbits = extra.get("motif_constraint_orbits")
    groups = extra.get("motif_constraint_groups")
    fixed_orbits = [
        orbit
        for orbit in orbits or []
        if isinstance(orbit, dict)
        and orbit.get("mobility_mode", "fixed") == "fixed"
    ]
    alignment_orbits = fixed_orbits or [
        orbit
        for orbit in orbits or []
        if isinstance(orbit, dict)
        and orbit.get("mobility_mode") == "orbit_rigid"
    ]

    if alignment_orbits and isinstance(groups, list) and groups:
        for orbit in alignment_orbits:
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


def _structure_residues(object_name: str) -> dict[str, list[dict]]:
    """Return ordered heavy-atom residues from one loaded PyMOL object."""

    chains: dict[str, list[dict]] = {}
    residue_lookup: dict[tuple[str, str], dict] = {}
    for atom in cmd.get_model(object_name).atom:
        if not _is_heavy(atom):
            continue
        chain = str(atom.chain)
        residue_id = str(atom.resi)
        identity = (chain, residue_id)
        residue = residue_lookup.get(identity)
        if residue is None:
            residue = {
                "chain": chain,
                "resi": residue_id,
                "resn": str(atom.resn).upper(),
                "atoms": {},
            }
            residue_lookup[identity] = residue
            chains.setdefault(chain, []).append(residue)
        atom_name = str(atom.name).upper()
        residue["atoms"].setdefault(
            atom_name,
            np.asarray(atom.coord, dtype=float),
        )
    return chains


def _sequence_coverage(
    source: dict[str, list[dict]],
    target: dict[str, list[dict]],
) -> float:
    """Fraction of source residues in chains found intact inside target."""

    target_sequences = [
        tuple(residue["resn"] for residue in residues)
        for residues in target.values()
    ]
    covered = 0
    total = 0
    for residues in source.values():
        sequence = tuple(residue["resn"] for residue in residues)
        total += len(sequence)
        if sequence and any(
            sequence == candidate[start : start + len(sequence)]
            for candidate in target_sequences
            for start in range(len(candidate) - len(sequence) + 1)
        ):
            covered += len(sequence)
    return covered / total if total else 0.0


def _infer_loaded_objects(
    reference: str,
    design: str,
) -> tuple[str, str]:
    objects = [
        name
        for name in cmd.get_names("objects", enabled_only=0)
        if not name.lower().endswith("_aligned")
        and name.lower() not in {"mosaic_aligned", "design_fixed_aligned"}
    ]
    if reference and design:
        return reference, design
    if reference or design:
        raise ValueError(
            "Provide both reference and design, or omit both for automatic "
            "detection"
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

    if len(objects) != 2:
        raise ValueError(
            "Automatic mosaic_align could not choose one reference and one "
            f"design from loaded raw objects {objects}. Either delete old "
            "raw objects or run `mosaic_align reference, design`."
        )

    first, second = objects
    first_residues = _structure_residues(first)
    second_residues = _structure_residues(second)
    first_coverage = _sequence_coverage(first_residues, second_residues)
    second_coverage = _sequence_coverage(second_residues, first_residues)
    if first_coverage > second_coverage + 0.05:
        return first, second
    if second_coverage > first_coverage + 0.05:
        return second, first

    residue_counts = {
        first: sum(map(len, first_residues.values())),
        second: sum(map(len, second_residues.values())),
    }
    ordered = sorted(objects, key=lambda name: residue_counts[name])
    if residue_counts[ordered[0]] == residue_counts[ordered[1]]:
        raise ValueError(
            "Cannot infer reference/design from the two CIF structures. "
            "Use: mosaic_align reference_object, design_object"
        )
    return ordered[0], ordered[1]


def _matched_fragment_coordinates(
    reference_residues: list[dict],
    design_residues: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    reference_coordinates: list[np.ndarray] = []
    design_coordinates: list[np.ndarray] = []
    for reference_residue, design_residue in zip(
        reference_residues,
        design_residues,
        strict=True,
    ):
        common_atoms = sorted(
            set(reference_residue["atoms"]).intersection(
                design_residue["atoms"]
            )
        )
        for atom_name in common_atoms:
            reference_coordinates.append(
                reference_residue["atoms"][atom_name]
            )
            design_coordinates.append(design_residue["atoms"][atom_name])
    return (
        np.asarray(reference_coordinates, dtype=float),
        np.asarray(design_coordinates, dtype=float),
    )


def _structure_match_candidates(
    reference: str,
    design: str,
    *,
    internal_rmsd_limit: float = 2.0,
) -> tuple[list[list[dict]], int]:
    """Find sequence-identical reference fragments in the design object."""

    reference_chains = _structure_residues(reference)
    design_chains = _structure_residues(design)
    candidates_by_fragment: list[list[dict]] = []
    reference_residue_count = 0
    for reference_chain, reference_residues in reference_chains.items():
        reference_sequence = tuple(
            residue["resn"] for residue in reference_residues
        )
        reference_residue_count += len(reference_sequence)
        fragment_candidates: list[dict] = []
        for design_chain, design_residues in design_chains.items():
            design_sequence = tuple(
                residue["resn"] for residue in design_residues
            )
            for start in range(
                len(design_sequence) - len(reference_sequence) + 1
            ):
                if (
                    design_sequence[start : start + len(reference_sequence)]
                    != reference_sequence
                ):
                    continue
                matched_design = design_residues[
                    start : start + len(reference_sequence)
                ]
                expected, observed = _matched_fragment_coordinates(
                    reference_residues,
                    matched_design,
                )
                if len(expected) < 3:
                    continue
                rotation, translation, internal_rmsd = _kabsch(
                    observed,
                    expected,
                )
                if internal_rmsd > internal_rmsd_limit:
                    continue
                fragment_candidates.append(
                    {
                        "reference_chain": reference_chain,
                        "design_chain": design_chain,
                        "design_start": start,
                        "residue_count": len(reference_sequence),
                        "expected": expected,
                        "observed": observed,
                        "rotation": rotation,
                        "translation": translation,
                        "internal_rmsd": internal_rmsd,
                    }
                )
        if not fragment_candidates:
            sequence_text = "-".join(reference_sequence)
            raise ValueError(
                "The design CIF contains no geometry-compatible copy of "
                f"reference chain {reference_chain!r} ({sequence_text}). "
                "Confirm that the two loaded CIFs belong to the same design."
            )
        candidates_by_fragment.append(fragment_candidates)
    if not candidates_by_fragment:
        raise ValueError("The reference CIF contains no heavy-atom residues")
    return candidates_by_fragment, reference_residue_count


def _transformed_rmsd(
    candidate: dict,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    delta = candidate["observed"] @ rotation + translation
    delta -= candidate["expected"]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _select_structural_correspondence(
    candidates_by_fragment: list[list[dict]],
    *,
    inlier_rmsd: float = 1.5,
) -> tuple[list[dict], np.ndarray, np.ndarray, float]:
    """Choose the largest set of fragments sharing one rigid transform."""

    proposals = [
        candidate
        for fragment_candidates in candidates_by_fragment
        for candidate in fragment_candidates
    ]
    best_selection: list[dict] | None = None
    best_key: tuple | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    for proposal in proposals:
        selection = [
            min(
                fragment_candidates,
                key=lambda candidate: _transformed_rmsd(
                    candidate,
                    proposal["rotation"],
                    proposal["translation"],
                ),
            )
            for fragment_candidates in candidates_by_fragment
        ]
        residuals = [
            _transformed_rmsd(
                candidate,
                proposal["rotation"],
                proposal["translation"],
            )
            for candidate in selection
        ]
        inlier_atoms = sum(
            len(candidate["observed"])
            for candidate, residual in zip(selection, residuals, strict=True)
            if residual <= inlier_rmsd
        )
        robust_error = sum(
            len(candidate["observed"])
            * min(residual, inlier_rmsd * 2.0) ** 2
            for candidate, residual in zip(selection, residuals, strict=True)
        )
        key = (-inlier_atoms, robust_error, proposal["internal_rmsd"])
        if best_key is None or key < best_key:
            best_key = key
            best_selection = selection
            rotation = proposal["rotation"]
            translation = proposal["translation"]

    assert best_selection is not None
    assert rotation is not None
    assert translation is not None
    for _ in range(3):
        selected = [
            min(
                fragment_candidates,
                key=lambda candidate: _transformed_rmsd(
                    candidate,
                    rotation,
                    translation,
                ),
            )
            for fragment_candidates in candidates_by_fragment
        ]
        inliers = [
            candidate
            for candidate in selected
            if _transformed_rmsd(candidate, rotation, translation)
            <= inlier_rmsd
        ]
        if not inliers:
            break
        expected = np.concatenate(
            [candidate["expected"] for candidate in inliers]
        )
        observed = np.concatenate(
            [candidate["observed"] for candidate in inliers]
        )
        rotation, translation, _ = _kabsch(observed, expected)
        best_selection = selected

    inliers = [
        candidate
        for candidate in best_selection
        if _transformed_rmsd(candidate, rotation, translation) <= inlier_rmsd
    ]
    if not inliers:
        raise ValueError(
            "The two CIFs share motif sequences but no consistent rigid "
            "three-dimensional correspondence"
        )
    expected = np.concatenate(
        [candidate["expected"] for candidate in inliers]
    )
    observed = np.concatenate(
        [candidate["observed"] for candidate in inliers]
    )
    rotation, translation, rmsd = _kabsch(observed, expected)
    return inliers, rotation, translation, rmsd


def mosaic_align_cifs(
    reference: str,
    design: str,
    output_object: str = "mosaic_aligned",
    replace: str = "1",
) -> None:
    """Align two loaded CIFs without requiring external run metadata."""

    objects = set(cmd.get_names("objects", enabled_only=0))
    for required in (reference, design):
        if required not in objects:
            raise ValueError(f"PyMOL object {required!r} is not loaded")
    replace_existing = str(replace).lower() not in {
        "0", "false", "no", "off",
    }
    if output_object in objects:
        if not replace_existing:
            raise ValueError(
                f"Output object {output_object!r} already exists; pass "
                "replace=1 or delete it before rerunning"
            )
        if output_object in {reference, design}:
            raise ValueError(
                "The aligned output object cannot overwrite either input"
            )
        cmd.delete(output_object)

    candidates, reference_residue_count = _structure_match_candidates(
        reference,
        design,
    )
    inliers, rotation, translation, rmsd = (
        _select_structural_correspondence(candidates)
    )
    matched_residues = sum(
        candidate["residue_count"] for candidate in inliers
    )
    matched_atoms = sum(len(candidate["observed"]) for candidate in inliers)

    cmd.create(output_object, design)
    coordinates = np.asarray(cmd.get_coords(output_object), dtype=float)
    cmd.load_coords(coordinates @ rotation + translation, output_object)
    print(
        f"mosaic_align: RMSD={rmsd:.6f} A "
        f"({matched_atoms} shared heavy atoms; "
        f"{matched_residues}/{reference_residue_count} reference residues); "
        f"created {output_object!r} from the two CIFs only"
    )


def _find_run_for_design(design: str, search_root: Path) -> Path:
    if not search_root.is_dir():
        raise FileNotFoundError(
            f"Mosaic alignment search root does not exist: {search_root}"
        )
    design_key = _normalized_name(design)
    # PyMOL appends _0001, _0002, ... when an object name collides.  That
    # suffix is not part of the result filename.
    design_keys = {
        design_key,
        re.sub(r"_[0-9]{4}$", "", design_key),
    }
    candidates = []
    for path in search_root.rglob("*model_0.json"):
        if _normalized_name(path.stem) not in design_keys:
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
    alignment_components = fixed_components or [
        orbit
        for orbit in orbits
        if isinstance(orbit, dict)
        and orbit.get("mobility_mode") == "orbit_rigid"
    ]
    if not alignment_components:
        raise ValueError(
            "This run has neither a globally fixed orbit nor an internally "
            "rigid mobile orbit available for provenance alignment"
        )
    declared = {
        _parse_component(str(component))
        for orbit in alignment_components
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
    replace: str = "1",
) -> None:
    """Align one output to its complete fixed orbit using run provenance."""

    objects = set(cmd.get_names("objects", enabled_only=0))
    for required in (reference, design):
        if required not in objects:
            raise ValueError(f"PyMOL object {required!r} is not loaded")
    replace_existing = str(replace).lower() not in {
        "0", "false", "no", "off",
    }
    if output_object in objects:
        if not replace_existing:
            raise ValueError(
                f"Output object {output_object!r} already exists; pass "
                "replace=1 or delete it explicitly before rerunning"
            )
        if output_object in {reference, design}:
            raise ValueError(
                "The aligned output object cannot overwrite the reference "
                "or raw design object"
            )
        cmd.delete(output_object)

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
    preexpanded_layout = extra.get("preexpanded_chain_layout")
    preexpanded = isinstance(preexpanded_layout, list)
    if not isinstance(order, list) or not isinstance(matrices, dict):
        raise ValueError("Compiled input lacks the symmetry registry")
    if len(order) != multiplicity or not output_chains:
        raise ValueError(
            "Symmetry multiplicity mismatch: "
            f"transforms={len(order)}, output_chains={len(output_chains)}, "
            f"declared={multiplicity}"
        )
    if preexpanded:
        if len(preexpanded_layout) != len(output_chains):
            raise ValueError(
                "Preexpanded chain layout does not match the loaded design: "
                f"layout={len(preexpanded_layout)}, "
                f"output_chains={len(output_chains)}"
            )
        asu_chain_count = int(extra.get("asu_chain_count", 1))
    else:
        if len(output_chains) % multiplicity != 0:
            raise ValueError(
                "Output chain count is incompatible with the declared "
                "symmetry action: "
                f"output_chains={len(output_chains)}, "
                f"multiplicity={multiplicity}"
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
        preexpanded=preexpanded,
    )
    rotation, translation, rmsd = _kabsch(observed, expected)

    orbits = extra.get("motif_constraint_orbits") or []
    has_global_fixed_orbit = any(
        isinstance(orbit, dict)
        and orbit.get("mobility_mode", "fixed") == "fixed"
        for orbit in orbits
    )
    alignment_basis = (
        "globally fixed orbit"
        if has_global_fixed_orbit
        else "rigid mobile orbit (pose change retained as residual)"
    )

    cmd.create(output_object, design)
    coordinates = np.asarray(cmd.get_coords(output_object), dtype=float)
    cmd.load_coords(coordinates @ rotation + translation, output_object)
    print(
        f"mosaic_align_fixed: RMSD={rmsd:.6f} A "
        f"({len(observed)} constrained heavy atoms; {alignment_basis}); "
        f"created {output_object!r}"
    )


def mosaic_load_run(
    run_dir: str,
    prefix: str = "mosaic",
    style: str = "1",
    replace: str = "1",
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
        replace_existing = str(replace).lower() not in {
            "0", "false", "no", "off",
        }
        if not replace_existing:
            raise ValueError(
                "Refusing to overwrite existing PyMOL objects: "
                + ", ".join(collisions)
            )
        for object_name in collisions:
            cmd.delete(object_name)

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
    search_root: str = ".",
) -> None:
    """Align two already loaded Mosaic CIFs.

    With exactly two raw CIF objects loaded, ``mosaic_align`` needs no
    arguments and no external run metadata.
    Explicit fallback::

        mosaic_align ref, design, /absolute/path/to/run
    """

    reference_value = str(reference).strip()
    design_value = str(design).strip()
    run_value = str(run_dir).strip()

    # Make the routine path-only command do what users naturally expect:
    # load the complete run and align it.  This also guarantees that the
    # required compiled metadata accompanies the two structures.
    path_candidate = Path(reference_value).expanduser()
    if (
        reference_value
        and not design_value
        and not run_value
        and path_candidate.is_dir()
    ):
        mosaic_load_run(str(path_candidate.resolve()), prefix="mosaic")
        return
    if (
        reference_value
        and not design_value
        and not run_value
        and ("/" in reference_value or reference_value.startswith("~"))
    ):
        raise FileNotFoundError(
            "The Mosaic run directory is not accessible on the machine "
            f"running PyMOL: {path_candidate}. If this is a remote-cluster "
            "path, drag its presymmetrized_input.cif and result "
            "CIF into PyMOL instead, then run mosaic_align with no arguments."
        )

    reference_object, design_object = _infer_loaded_objects(
        reference_value,
        design_value,
    )
    if run_value:
        run_path = Path(run_dir).expanduser().resolve()
        mosaic_align_fixed(
            reference_object,
            design_object,
            str(run_path),
            output_object,
        )
        alignment_source = f"provenance from {run_path}"
    else:
        mosaic_align_cifs(
            reference_object,
            design_object,
            output_object,
        )
        alignment_source = "the two loaded CIFs"
    cmd.hide("everything", reference_object)
    cmd.hide("everything", output_object)
    cmd.show("cartoon", reference_object)
    cmd.show("cartoon", output_object)
    cmd.color("cyan", reference_object)
    cmd.color("magenta", output_object)
    cmd.set("cartoon_transparency", 0.6, reference_object)
    cmd.disable(design_object)
    cmd.zoom(f"{reference_object} or {output_object}")
    print(f"mosaic_align source: {alignment_source}")
    print(
        f"mosaic_align reference (cyan, unchanged): {reference_object} "
        "[compiled pre-diffusion input]"
    )
    print(
        f"mosaic_align generated result (original object hidden): "
        f"{design_object}"
    )
    print(
        f"mosaic_align display copy (magenta): {output_object} "
        "[alignment-only object; no new CIF was generated]"
    )


cmd.extend("mosaic_align_fixed", mosaic_align_fixed)
cmd.extend("mosaic_load_run", mosaic_load_run)
cmd.extend("mosaic_align_cifs", mosaic_align_cifs)
cmd.extend("mosaic_align", mosaic_align)
print(
    "Mosaic alignment loaded: drag one compiled presymmetrized_input.cif "
    "and one generated *model_0.cif into PyMOL, then type mosaic_align. "
    "The command creates only a display copy, not another design."
)
