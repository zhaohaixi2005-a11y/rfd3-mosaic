"""Materialize independent interface-seed files into one canonical input.

Ordinary users are allowed to supply every interface seed in its own PDB or
mmCIF coordinate frame.  Those arbitrary file frames are not assembly
coordinates.  This module extracts only the declared seed atoms, assigns
collision-free canonical chain identifiers, and expresses every complete
interface in a deterministic local frame.  The downstream public compiler
still receives exactly one structure and therefore remains the sole
execution path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rfd3_mosaic.design_compiler import parse_public_selector
from rfd3_mosaic.schema.simple_intent import SimpleCageIntentSpec
from rfd3_mosaic.structure import AtomRecord, read_structure_atoms


@dataclass(frozen=True)
class SeedLibraryMaterialization:
    intent: SimpleCageIntentSpec
    structure_path: Path
    independent_frames: bool
    manifest: dict[str, Any]


def _is_heavy(atom: AtomRecord) -> bool:
    return not (
        atom.element.strip().upper().startswith("H")
        or atom.atom_name.strip().upper().startswith("H")
    )


def _selected_atoms(
    atoms: tuple[AtomRecord, ...],
    selector: str,
) -> tuple[AtomRecord, ...]:
    segments = parse_public_selector(selector)
    selected = tuple(
        atom
        for atom in atoms
        if any(
            atom.chain_id == segment.chain_id
            and segment.residue_start
            <= atom.residue_number
            <= segment.residue_end
            for segment in segments
        )
    )
    if not selected:
        raise ValueError(f"Interface selector {selector!r} matched no atoms")
    return selected


def _ordered_participant_segments(selector: str):
    """Return one participant's same-chain fixed fragments in chain order.

    One physical interface face may be made from several helices or loops.
    They remain one rigid participant, but their intervening sequence is a
    polymer-path question.  Ordinary mode can infer that path only when all
    selected fragments belong to one source chain; it must not invent a
    covalent connection between unrelated chains.
    """

    segments = tuple(
        sorted(
            parse_public_selector(selector),
            key=lambda item: (
                item.chain_id,
                item.residue_start,
                item.residue_end,
            ),
        )
    )
    chains = {segment.chain_id for segment in segments}
    if len(chains) != 1:
        raise NotImplementedError(
            "One ordinary interface participant may contain several "
            "fixed fragments only when they belong to one source polymer "
            "chain; use expert component paths for cross-chain participants"
        )
    for left, right in zip(segments, segments[1:]):
        if left.residue_end >= right.residue_start:
            raise ValueError(
                "Interface participant fixed ranges must be ordered and "
                "non-overlapping"
            )
    return segments


def _canonical_frame(
    participant_atoms: tuple[tuple[AtomRecord, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return center and a right-handed world-from-local rotation matrix."""

    heavy_groups = [
        tuple(atom for atom in group if _is_heavy(atom))
        for group in participant_atoms
    ]
    if any(not group for group in heavy_groups):
        raise ValueError("Every interface participant must contain heavy atoms")
    coordinates = np.asarray(
        [atom.coordinate for group in heavy_groups for atom in group],
        dtype=np.float64,
    )
    center = coordinates.mean(axis=0)
    participant_centers = np.asarray(
        [
            np.asarray([atom.coordinate for atom in group], dtype=np.float64)
            .mean(axis=0)
            for group in heavy_groups
        ]
    )
    x_axis = participant_centers[1] - participant_centers[0]
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-8:
        raise ValueError(
            "Interface participant centers coincide; canonical seed frame "
            "is undefined"
        )
    x_axis /= x_norm

    centered = coordinates - center
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    y_axis: np.ndarray | None = None
    for candidate in eigenvectors.T[::-1]:
        orthogonal = candidate - np.dot(candidate, x_axis) * x_axis
        norm = float(np.linalg.norm(orthogonal))
        if norm > 1e-8:
            y_axis = orthogonal / norm
            break
    if y_axis is None:
        trial = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(trial, x_axis))) > 0.9:
            trial = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        y_axis = trial - np.dot(trial, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis)

    # Eigenvector signs are arbitrary.  Tie the sign to the first atom that
    # has a measurable projection so a rigidly transformed input produces
    # the same canonical coordinates.
    for coordinate in centered:
        projection = float(np.dot(coordinate, y_axis))
        if abs(projection) > 1e-8:
            if projection < 0.0:
                y_axis *= -1.0
            break
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return center, np.column_stack((x_axis, y_axis, z_axis))


def _cif_token(value: str) -> str:
    if not value:
        return "."
    if any(character.isspace() for character in value):
        return "'" + value.replace("'", "''") + "'"
    return value


def _write_canonical_cif(
    path: Path,
    atoms: list[tuple[AtomRecord, str, tuple[float, float, float], int]],
) -> None:
    headers = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    )
    lines = [
        "data_rfd3_mosaic_seed_library",
        "#",
        "loop_",
        *headers,
    ]
    for atom_id, (atom, chain_id, coordinate, entity_id) in enumerate(
        atoms,
        start=1,
    ):
        element = atom.element.strip() or atom.atom_name.strip()[:1]
        x, y, z = coordinate
        lines.append(
            " ".join(
                (
                    _cif_token(atom.record_type or "ATOM"),
                    str(atom_id),
                    _cif_token(element),
                    _cif_token(atom.atom_name),
                    ".",
                    _cif_token(atom.residue_name),
                    _cif_token(chain_id),
                    str(entity_id),
                    str(atom.residue_number),
                    _cif_token(atom.insertion_code),
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    "1.00",
                    "20.00",
                    str(atom.residue_number),
                    _cif_token(atom.residue_name),
                    _cif_token(chain_id),
                    _cif_token(atom.atom_name),
                    "1",
                )
            )
        )
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_seed_library(
    intent: SimpleCageIntentSpec,
    output_directory: str | Path,
) -> SeedLibraryMaterialization:
    """Create one canonical structure when seeds use independent sources."""

    sources = {
        seed_id: (seed.source or intent.input).expanduser().resolve()
        for seed_id, seed in intent.interface_seeds.items()
    }
    explicit_sources = any(
        seed.source is not None for seed in intent.interface_seeds.values()
    )
    distinct_source_count = len(set(sources.values()))
    if intent.seed_layout == "preserve_input" and distinct_source_count > 1:
        raise ValueError(
            "seed_layout=preserve_input requires every supplied interface "
            "seed to use one shared input structure. Use seed_layout=solve "
            "when separate files are arbitrary local coordinate frames"
        )
    solve_layout = (
        (
            intent.seed_layout == "solve"
            and len(intent.interface_seeds) > 1
        )
        or (
            intent.seed_layout == "auto"
            and explicit_sources
            and distinct_source_count > 1
        )
    )
    if not solve_layout:
        return SeedLibraryMaterialization(
            intent=intent,
            structure_path=intent.input,
            independent_frames=False,
            manifest={
                "mode": "shared_input_frame",
                "requested_seed_layout": intent.seed_layout,
                "relative_seed_pose": "preserved",
                "canonical_structure": str(intent.input),
                "sources": {
                    seed_id: str(path) for seed_id, path in sources.items()
                },
            },
        )

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    structure_path = output / "canonical_interface_seed_library.cif"
    canonical_atoms: list[
        tuple[AtomRecord, str, tuple[float, float, float], int]
    ] = []
    normalized_seeds = {}
    normalized_participants: dict[tuple[str, str], str] = {}
    seed_manifest: dict[str, Any] = {}
    for seed_index, (seed_id, seed) in enumerate(
        sorted(intent.interface_seeds.items()),
        start=1,
    ):
        source = sources[seed_id]
        atoms = read_structure_atoms(source)
        participant_atoms = tuple(
            _selected_atoms(atoms, seed.selectors[participant])
            for participant in seed.participants
        )
        center, rotation = _canonical_frame(participant_atoms)
        participants: list[str] = []
        selectors: dict[str, str] = {}
        participant_manifest = []
        seen_identities: set[tuple[str, int, str, str]] = set()
        for participant_index, (participant, selected) in enumerate(
            zip(seed.participants, participant_atoms, strict=True),
            start=1,
        ):
            chain_id = f"S{seed_index:03d}P{participant_index:03d}"
            participants.append(chain_id)
            normalized_participants[(seed_id, participant)] = chain_id
            ranges = _ordered_participant_segments(
                seed.selectors[participant]
            )
            selectors[chain_id] = ",".join(
                f"{chain_id}/{segment.residue_start}-"
                f"{segment.residue_end}/*"
                for segment in ranges
            )
            for atom in selected:
                identity = (
                    atom.chain_id,
                    atom.residue_number,
                    atom.insertion_code,
                    atom.atom_name,
                )
                if identity in seen_identities:
                    raise ValueError(
                        f"Interface seed {seed_id!r} selectors overlap at "
                        f"{identity}"
                    )
                seen_identities.add(identity)
                local = rotation.T @ (
                    np.asarray(atom.coordinate, dtype=np.float64) - center
                )
                canonical_atoms.append(
                    (
                        atom,
                        chain_id,
                        tuple(float(value) for value in local),
                        seed_index,
                    )
                )
            participant_manifest.append(
                {
                    "source_participant": participant,
                    "source_selector": seed.selectors[participant],
                    "canonical_chain": chain_id,
                    "canonical_selector": selectors[chain_id],
                }
            )
        normalized_seeds[seed_id] = seed.model_copy(
            update={
                "source": structure_path,
                "participants": tuple(participants),
                "selectors": selectors,
            }
        )
        seed_manifest[seed_id] = {
            "source": str(source),
            "source_sha256": _sha256(source),
            "canonical_center": [float(value) for value in center],
            "world_from_local_rotation": rotation.tolist(),
            "participants": participant_manifest,
            "requested_physical_instances": seed.use.description,
        }

    _write_canonical_cif(structure_path, canonical_atoms)
    normalized_connections = tuple(
        connection.model_copy(
            update={
                "from_endpoint": connection.from_endpoint.model_copy(
                    update={
                        "participant": normalized_participants[
                            (
                                connection.from_endpoint.interface,
                                connection.from_endpoint.participant,
                            )
                        ]
                    }
                ),
                "to_endpoint": connection.to_endpoint.model_copy(
                    update={
                        "participant": normalized_participants[
                            (
                                connection.to_endpoint.interface,
                                connection.to_endpoint.participant,
                            )
                        ]
                    }
                ),
            }
        )
        for connection in intent.polymer_connections
    )
    normalized_intent = intent.model_copy(
        update={
            "input": structure_path,
            "interface_seeds": normalized_seeds,
            "polymer_connections": normalized_connections,
        }
    )
    return SeedLibraryMaterialization(
        intent=normalized_intent,
        structure_path=structure_path,
        independent_frames=True,
        manifest={
            "mode": "independent_seed_local_frames",
            "requested_seed_layout": intent.seed_layout,
            "relative_seed_pose": "solve",
            "canonical_structure": str(structure_path),
            "canonical_structure_sha256": _sha256(structure_path),
            "seed_count": len(normalized_seeds),
            "seeds": seed_manifest,
        },
    )


__all__ = ["SeedLibraryMaterialization", "materialize_seed_library"]
