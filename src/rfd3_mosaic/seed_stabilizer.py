"""Infer strict regular stabilizer actions from supplied interface seeds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np

from rfd3_mosaic.design_compiler import parse_public_selector
from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.structure import AtomRecord, read_structure_atoms
from rfd3_mosaic.topology.stabilizer_cosets import (
    StabilizerCosetHypothesis,
    stabilizer_coset_hypotheses,
)
from rfd3_mosaic.topology.symmetry_connectivity import finite_symmetry_spec

_BACKBONE = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True)
class SeedStabilizerEvidence:
    interface_id: str
    symmetry: str
    orbit_size: int
    participant_count: int
    maximum_fit_rmsd: float
    closure_rotation_error_deg: float
    closure_translation_error: float
    common_center_residual: float
    symmetry_axis: tuple[float, float, float]
    symmetry_secondary_axis: tuple[float, float, float]
    symmetry_center: tuple[float, float, float]
    stabilizer_transform_ids: tuple[str, ...]
    coset_representative_ids: tuple[str, ...]
    transform_to_coset_representative: tuple[tuple[str, str], ...]
    canonical_to_participant: tuple[tuple[str, str], ...]

    @property
    def finite_action_payload(self) -> dict[str, object]:
        return {
            "coset_representative_ids": self.coset_representative_ids,
            "stabilizer_transform_ids": self.stabilizer_transform_ids,
            "transform_to_coset_representative": dict(
                self.transform_to_coset_representative
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _selected_backbone_atoms(
    atoms: tuple[AtomRecord, ...],
    selector: str,
) -> tuple[AtomRecord, ...]:
    segments = parse_public_selector(selector)
    selected = tuple(
        atom
        for atom in atoms
        if atom.atom_name.upper() in _BACKBONE
        and any(
            atom.chain_id == segment.chain_id
            and segment.residue_start
            <= atom.residue_number
            <= segment.residue_end
            for segment in segments
        )
    )
    if not selected:
        raise ValueError(
            f"Stabilizer selector {selector!r} matched no backbone atoms"
        )
    return selected


def _ordered_coordinates(
    atoms: tuple[AtomRecord, ...],
) -> tuple[tuple[tuple[int, str, str], ...], np.ndarray]:
    residue_ids = sorted(
        {(atom.residue_number, atom.insertion_code) for atom in atoms}
    )
    residue_offsets = {
        residue_id: offset for offset, residue_id in enumerate(residue_ids)
    }
    keyed = sorted(
        (
            (
                residue_offsets[(atom.residue_number, atom.insertion_code)],
                atom.residue_name,
                atom.atom_name.upper(),
            ),
            atom,
        )
        for atom in atoms
    )
    signature = tuple(key for key, _ in keyed)
    coordinates = np.asarray(
        [atom.coordinate for _, atom in keyed],
        dtype=np.float64,
    )
    return signature, coordinates


def _fit_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    fitted = (rotation @ source.T).T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _transform_error(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    rotation_error = _rotation_angle_deg(left[0] @ right[0].T)
    translation_error = float(np.linalg.norm(left[1] - right[1]))
    return rotation_error, translation_error


def _compose(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return left[0] @ right[0], left[0] @ right[1] + left[1]


def _fitted_multiplication_table(
    transforms: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    rotation_tolerance_deg: float,
    translation_tolerance: float,
) -> tuple[tuple[tuple[int, ...], ...], float, float]:
    table: list[tuple[int, ...]] = []
    maximum_rotation = 0.0
    maximum_translation = 0.0
    for left in transforms:
        row: list[int] = []
        for right in transforms:
            composed = _compose(left, right)
            errors = tuple(
                _transform_error(composed, candidate)
                for candidate in transforms
            )
            best = min(
                range(len(errors)),
                key=lambda index: (errors[index][0], errors[index][1], index),
            )
            rotation_error, translation_error = errors[best]
            if (
                rotation_error > rotation_tolerance_deg
                or translation_error > translation_tolerance
            ):
                raise ValueError(
                    "Participant rigid transforms do not form a closed "
                    "rotation group"
                )
            maximum_rotation = max(maximum_rotation, rotation_error)
            maximum_translation = max(
                maximum_translation,
                translation_error,
            )
            row.append(best)
        table.append(tuple(row))
    return tuple(table), maximum_rotation, maximum_translation


def _canonical_subgroup_table(
    symmetry: str,
    hypothesis: StabilizerCosetHypothesis,
) -> tuple[tuple[int, ...], ...]:
    registry = build_transform_registry(finite_symmetry_spec(symmetry))
    ids = hypothesis.stabilizer_transform_ids
    indices = {transform_id: index for index, transform_id in enumerate(ids)}
    return tuple(
        tuple(
            indices[registry.compose_ids(left_id, right_id)]
            for right_id in ids
        )
        for left_id in ids
    )


def _group_isomorphism(
    symmetry: str,
    hypothesis: StabilizerCosetHypothesis,
    fitted_table: tuple[tuple[int, ...], ...],
    fitted_rotations: tuple[np.ndarray, ...],
) -> tuple[int, ...] | None:
    size = len(hypothesis.stabilizer_transform_ids)
    if size != len(fitted_table) or size > 8:
        return None
    registry = build_transform_registry(finite_symmetry_spec(symmetry))
    canonical_table = _canonical_subgroup_table(symmetry, hypothesis)
    canonical_angles = tuple(
        _rotation_angle_deg(registry.transform(item)[:3, :3])
        for item in hypothesis.stabilizer_transform_ids
    )
    fitted_angles = tuple(_rotation_angle_deg(item) for item in fitted_rotations)
    for remainder in permutations(range(1, size)):
        mapping = (0, *remainder)
        if any(
            abs(canonical_angles[index] - fitted_angles[mapping[index]])
            > 1.0
            for index in range(size)
        ):
            continue
        if all(
            mapping[canonical_table[left][right]]
            == fitted_table[mapping[left]][mapping[right]]
            for left in range(size)
            for right in range(size)
        ):
            return mapping
    return None


def _rotation_axis(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    angle = math.radians(_rotation_angle_deg(rotation))
    if angle < 1e-7:
        raise ValueError("Identity rotation has no unique axis")
    if abs(angle - math.pi) < 1e-5:
        values, vectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(values - 1.0)))
        axis = np.real(vectors[:, index])
    else:
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
    axis /= np.linalg.norm(axis)
    return axis, angle


def _frame_from_two_axes(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    z_axis = first / np.linalg.norm(first)
    x_axis = second - float(np.dot(second, z_axis)) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        trial = np.asarray((1.0, 0.0, 0.0))
        if abs(float(np.dot(trial, source))) > 0.9:
            trial = np.asarray((0.0, 1.0, 0.0))
        axis = np.cross(source, trial)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def _conjugating_frame(
    symmetry: str,
    hypothesis: StabilizerCosetHypothesis,
    mapping: tuple[int, ...],
    fitted_rotations: tuple[np.ndarray, ...],
) -> np.ndarray:
    registry = build_transform_registry(finite_symmetry_spec(symmetry))
    pairs = []
    for canonical_index, transform_id in enumerate(
        hypothesis.stabilizer_transform_ids
    ):
        if canonical_index == 0:
            continue
        canonical_axis, canonical_angle = _rotation_axis(
            registry.transform(transform_id)[:3, :3]
        )
        fitted_axis, fitted_angle = _rotation_axis(
            fitted_rotations[mapping[canonical_index]]
        )
        pairs.append(
            (canonical_axis, fitted_axis, canonical_angle, fitted_angle)
        )
    candidates: list[np.ndarray] = []
    for left_index, left in enumerate(pairs):
        for right in pairs[left_index + 1 :]:
            if abs(float(np.dot(left[0], right[0]))) > 0.95:
                continue
            left_signs = (-1.0, 1.0) if abs(left[3] - math.pi) < 1e-5 else (1.0,)
            right_signs = (-1.0, 1.0) if abs(right[3] - math.pi) < 1e-5 else (1.0,)
            for left_sign, right_sign in product(left_signs, right_signs):
                canonical_frame = _frame_from_two_axes(left[0], right[0])
                fitted_frame = _frame_from_two_axes(
                    left_sign * left[1],
                    right_sign * right[1],
                )
                candidates.append(fitted_frame @ canonical_frame.T)
    if not candidates:
        canonical_axis, fitted_axis, _, fitted_angle = pairs[0]
        signs = (-1.0, 1.0) if abs(fitted_angle - math.pi) < 1e-5 else (1.0,)
        candidates.extend(
            _align_vectors(canonical_axis, sign * fitted_axis)
            for sign in signs
        )

    def residual(frame: np.ndarray) -> float:
        return max(
            float(
                np.linalg.norm(
                    frame
                    @ registry.transform(transform_id)[:3, :3]
                    @ frame.T
                    - fitted_rotations[mapping[index]]
                )
            )
            for index, transform_id in enumerate(
                hypothesis.stabilizer_transform_ids
            )
        )

    frame = min(candidates, key=residual)
    if residual(frame) > 5e-2:
        raise ValueError(
            "Participant symmetry is abstractly compatible but no stable "
            "common orientation maps it onto the selected stabilizer"
        )
    return frame


def _common_center(
    transforms: tuple[tuple[np.ndarray, np.ndarray], ...],
    seed_center: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrices = []
    targets = []
    for rotation, translation in transforms[1:]:
        matrices.append(np.eye(3) - rotation)
        targets.append(translation)
    matrix = np.concatenate(matrices, axis=0)
    target = np.concatenate(targets, axis=0)
    center, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    _, _, right = np.linalg.svd(matrix)
    null_vectors = right[np.linalg.norm(matrix @ right.T, axis=0) < 1e-7]
    for vector in null_vectors:
        center += vector * float(np.dot(seed_center - center, vector))
    residual = float(np.sqrt(np.mean((matrix @ center - target) ** 2)))
    return center, residual


def resolve_seed_stabilizer(
    *,
    source: str | Path,
    interface_id: str,
    participants: tuple[str, ...],
    selectors: dict[str, str],
    symmetry: str,
    orbit_size: int,
    maximum_fit_rmsd: float = 0.25,
    rotation_tolerance_deg: float = 2.0,
    translation_tolerance: float = 0.5,
) -> SeedStabilizerEvidence:
    """Resolve one regular participant action onto a finite stabilizer."""

    atoms = read_structure_atoms(
        source,
        mmcif_identifier_namespace="label",
    )
    signatures = []
    coordinates = []
    for participant in participants:
        signature, participant_coordinates = _ordered_coordinates(
            _selected_backbone_atoms(atoms, selectors[participant])
        )
        signatures.append(signature)
        coordinates.append(participant_coordinates)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            f"Interface {interface_id!r} participants do not have identical "
            "ordered backbone atom signatures and cannot be permuted by a "
            "regular stabilizer action"
        )
    fitted = [(np.eye(3), np.zeros(3))]
    fit_rmsds = [0.0]
    for target in coordinates[1:]:
        rotation, translation, rmsd = _fit_transform(coordinates[0], target)
        if rmsd > maximum_fit_rmsd:
            raise ValueError(
                f"Interface {interface_id!r} participant rigid-fit RMSD "
                f"{rmsd:.3f} A exceeds {maximum_fit_rmsd:.3f} A"
            )
        fitted.append((rotation, translation))
        fit_rmsds.append(rmsd)
    fitted_transforms = tuple(fitted)
    fitted_table, closure_rotation, closure_translation = (
        _fitted_multiplication_table(
            fitted_transforms,
            rotation_tolerance_deg=rotation_tolerance_deg,
            translation_tolerance=translation_tolerance,
        )
    )
    seed_center = np.concatenate(coordinates, axis=0).mean(axis=0)
    center, center_residual = _common_center(fitted_transforms, seed_center)
    if center_residual > 0.1:
        raise ValueError(
            f"Interface {interface_id!r} participant transforms do not "
            "share a common rotational center"
        )
    fitted_rotations = tuple(item[0] for item in fitted_transforms)
    for hypothesis in stabilizer_coset_hypotheses(symmetry, orbit_size):
        if hypothesis.stabilizer_order != len(participants):
            continue
        mapping = _group_isomorphism(
            symmetry,
            hypothesis,
            fitted_table,
            fitted_rotations,
        )
        if mapping is None:
            continue
        frame = _conjugating_frame(
            symmetry,
            hypothesis,
            mapping,
            fitted_rotations,
        )
        axis = frame[:, 2]
        secondary = frame[:, 0]
        return SeedStabilizerEvidence(
            interface_id=interface_id,
            symmetry=symmetry,
            orbit_size=orbit_size,
            participant_count=len(participants),
            maximum_fit_rmsd=max(fit_rmsds),
            closure_rotation_error_deg=closure_rotation,
            closure_translation_error=closure_translation,
            common_center_residual=center_residual,
            symmetry_axis=tuple(float(value) for value in axis),
            symmetry_secondary_axis=tuple(
                float(value) for value in secondary
            ),
            symmetry_center=tuple(float(value) for value in center),
            stabilizer_transform_ids=(
                hypothesis.stabilizer_transform_ids
            ),
            coset_representative_ids=(
                hypothesis.coset_representative_ids
            ),
            transform_to_coset_representative=(
                hypothesis.transform_to_coset_representative
            ),
            canonical_to_participant=tuple(
                (
                    transform_id,
                    participants[mapping[index]],
                )
                for index, transform_id in enumerate(
                    hypothesis.stabilizer_transform_ids
                )
            ),
        )
    raise ValueError(
        f"Interface {interface_id!r} participant symmetry is not isomorphic "
        f"to any order-{len(participants)} stabilizer producing a "
        f"{symmetry} orbit of size {orbit_size}"
    )


__all__ = ["SeedStabilizerEvidence", "resolve_seed_stabilizer"]
