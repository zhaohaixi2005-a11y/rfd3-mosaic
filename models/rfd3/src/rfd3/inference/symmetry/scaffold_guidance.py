"""Scaffold-derived rigid-pose guidance for symmetry motif orbits.

This module contains only deterministic geometry operations.  It does not
change RFD3 model weights and it never moves symmetry copies independently.
The caller optimizes one master pose, then expands that pose through every
declared group action.  Axis-dependent objectives use the primary cyclic
subgroup of Cn or Dn; orbit materialization itself is group-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class BoundaryTopology:
    """Atom indices needed by the coarse scaffold objective."""

    junction_pairs: torch.Tensor
    fixed_ca_atom_indices: torch.Tensor
    generated_ca_atom_indices: torch.Tensor
    generated_atom_mask: torch.Tensor


@dataclass(frozen=True)
class CyclicAxis:
    """Primary cyclic-subgroup axis used by axis-dependent objectives.

    The historical class name is retained for API compatibility.  For Dn,
    ``transform_ids`` contains the identity and rotations in the primary Cn
    subgroup, while the complete Dn transform registry remains responsible
    for orbit expansion and exact projection.
    """

    point: torch.Tensor
    direction: torch.Tensor
    transform_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScaffoldGuidanceConfig:
    """Weights and geometric targets for the first mobility pilot."""

    junction_weight: float = 1.0
    clash_weight: float = 1.0
    tilt_weight: float = 0.25
    prior_weight: float = 0.05
    junction_target_distance: float = 3.8
    junction_huber_delta: float = 0.25
    clash_distance: float = 3.0
    maximum_tilt_degrees: float = 20.0
    translation_prior_scale: float = 1.0
    rotation_prior_scale_degrees: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "junction_weight",
            "clash_weight",
            "tilt_weight",
            "prior_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "junction_target_distance",
            "junction_huber_delta",
            "clash_distance",
            "translation_prior_scale",
            "rotation_prior_scale_degrees",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.maximum_tilt_degrees < 90.0:
            raise ValueError("maximum_tilt_degrees must be in [0, 90)")


@dataclass(frozen=True)
class ScaffoldGuidanceEnergy:
    """Differentiable objective and its individual terms."""

    total: torch.Tensor
    junction: torch.Tensor
    clash: torch.Tensor
    tilt: torch.Tensor
    prior: torch.Tensor
    maximum_junction_error: torch.Tensor
    minimum_clash_distance: torch.Tensor
    tilt_degrees: torch.Tensor
    junction_distances: torch.Tensor
    minimum_clash_distances: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu().item()),
            "junction": float(self.junction.detach().cpu().item()),
            "clash": float(self.clash.detach().cpu().item()),
            "tilt": float(self.tilt.detach().cpu().item()),
            "prior": float(self.prior.detach().cpu().item()),
            "maximum_junction_error": float(
                self.maximum_junction_error.detach().cpu().item()
            ),
            "minimum_clash_distance": float(
                self.minimum_clash_distance.detach().cpu().item()
            ),
            "tilt_degrees": float(self.tilt_degrees.detach().cpu().item()),
        }


@dataclass(frozen=True)
class SE3Proposal:
    """One accepted or rejected bounded master-pose proposal."""

    rotation: torch.Tensor
    translation: torch.Tensor
    delta_rotation: torch.Tensor
    delta_translation: torch.Tensor
    initial_energy: torch.Tensor
    proposed_energy: torch.Tensor
    accepted: bool
    line_search_scale: float
    rotation_gradient_norm: float
    translation_gradient_norm: float
    projected_rotation_gradient_norm: float
    projected_translation_gradient_norm: float
    line_search_trials: tuple[dict[str, float | bool | None], ...]


def _as_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def build_boundary_topology(
    f: dict[str, Any],
    fixed_mask: torch.Tensor,
) -> BoundaryTopology:
    """Find fixed/generated CA boundaries from runtime topology features.

    Explicit ``token_bonds`` cover non-polymer covalent bonds, but Foundry
    does not require ordinary peptide neighbours to be present in that
    matrix.  Protein-polymer neighbours are therefore also reconstructed
    from consecutive ``residue_index`` values within one ``asym_id``.  A
    boundary has opposite fixed-coordinate states on its two tokens.  The
    fixed atom is always stored first in ``junction_pairs``.
    """

    required = {
        "atom_to_token_map",
        "asym_id",
        "is_ca",
        "token_bonds",
    }
    missing = required - set(f)
    if missing:
        raise ValueError("Scaffold guidance requires features " f"{sorted(missing)}")

    fixed = _as_tensor(fixed_mask, dtype=torch.bool)
    if fixed.ndim != 1:
        raise ValueError("fixed_mask must have shape [L]")
    device = fixed.device
    atom_to_token = _as_tensor(
        f["atom_to_token_map"],
        dtype=torch.long,
        device=device,
    )
    is_ca = _as_tensor(
        f["is_ca"],
        dtype=torch.bool,
        device=device,
    )
    if (
        atom_to_token.ndim != 1
        or is_ca.ndim != 1
        or atom_to_token.shape != fixed.shape
        or is_ca.shape != fixed.shape
    ):
        raise ValueError("atom_to_token_map, is_ca, and fixed_mask must have shape [L]")
    if atom_to_token.numel() == 0 or torch.any(atom_to_token < 0):
        raise ValueError("atom_to_token_map must contain non-negative tokens")

    token_count = int(atom_to_token.max().item()) + 1
    asym_id = _as_tensor(
        f["asym_id"],
        dtype=torch.long,
        device=device,
    )
    token_bonds = _as_tensor(
        f["token_bonds"],
        dtype=torch.bool,
        device=device,
    )
    if asym_id.shape != (token_count,):
        raise ValueError("asym_id must have shape [N_tokens]")
    if token_bonds.shape != (token_count, token_count):
        raise ValueError("token_bonds must have shape [N_tokens, N_tokens]")
    is_protein = _as_tensor(
        f.get(
            "is_protein",
            torch.ones(token_count, dtype=torch.bool, device=device),
        ),
        dtype=torch.bool,
        device=device,
    )
    if is_protein.shape != (token_count,):
        raise ValueError("is_protein must have shape [N_tokens]")
    residue_index = None
    if "residue_index" in f:
        residue_index = _as_tensor(
            f["residue_index"],
            dtype=torch.long,
            device=device,
        )
        if residue_index.shape != (token_count,):
            raise ValueError("residue_index must have shape [N_tokens]")
    is_virtual = _as_tensor(
        f.get(
            "is_virtual",
            torch.zeros_like(fixed, dtype=torch.bool),
        ),
        dtype=torch.bool,
        device=device,
    )
    if is_virtual.shape != fixed.shape:
        raise ValueError("is_virtual must have shape [L]")
    considered_atom = ~is_virtual

    token_atom_counts = torch.zeros(
        token_count,
        dtype=torch.long,
        device=device,
    )
    token_fixed_counts = torch.zeros_like(token_atom_counts)
    token_atom_counts.index_add_(
        0,
        atom_to_token,
        considered_atom.to(dtype=torch.long),
    )
    token_fixed_counts.index_add_(
        0,
        atom_to_token,
        (fixed & considered_atom).to(dtype=torch.long),
    )
    # A fixed-backbone motif with redesignable side chains is intentionally
    # only partially fixed at atom level.  It is nevertheless one fixed
    # *residue* for scaffold topology: its CA belongs to the rigid target and
    # its side-chain atoms must not be mistaken for generated scaffold.  Use
    # the unique protein CA as the authoritative token-level state.  Retain
    # the all-atom rule for non-protein tokens, which have no CA convention.
    token_ca_counts = torch.zeros_like(token_atom_counts)
    token_fixed_ca_counts = torch.zeros_like(token_atom_counts)
    ca_atoms = is_ca & considered_atom
    token_ca_counts.index_add_(
        0,
        atom_to_token,
        ca_atoms.to(dtype=torch.long),
    )
    token_fixed_ca_counts.index_add_(
        0,
        atom_to_token,
        (fixed & ca_atoms).to(dtype=torch.long),
    )
    invalid_protein_ca = is_protein & (token_ca_counts != 1)
    if torch.any(invalid_protein_ca):
        tokens = torch.nonzero(invalid_protein_ca, as_tuple=False).flatten().tolist()
        raise ValueError(
            "Every protein token must have exactly one CA representative; "
            f"invalid tokens: {tokens}"
        )
    partially_fixed_nonprotein = (
        ~is_protein
        & (token_fixed_counts > 0)
        & (token_fixed_counts < token_atom_counts)
    )
    if torch.any(partially_fixed_nonprotein):
        tokens = (
            torch.nonzero(partially_fixed_nonprotein, as_tuple=False)
            .flatten()
            .tolist()
        )
        raise ValueError(
            "Non-protein scaffold-boundary tokens require whole-token fixed "
            f"states; partially fixed tokens: {tokens}"
        )
    fixed_token = (token_atom_counts > 0) & (
        token_fixed_counts == token_atom_counts
    )
    fixed_token[is_protein] = token_fixed_ca_counts[is_protein] == 1
    fixed_topology_atom = fixed_token[atom_to_token] & considered_atom

    ca_by_token: dict[int, int] = {}
    for token_id_tensor in torch.unique(atom_to_token):
        token_id = int(token_id_tensor.item())
        matches = torch.nonzero(
            (atom_to_token == token_id) & is_ca,
            as_tuple=False,
        ).flatten()
        if len(matches) == 1:
            ca_by_token[token_id] = int(matches[0].item())

    undirected_bonds = token_bonds | token_bonds.T
    candidate_pairs = {
        (int(pair[0].item()), int(pair[1].item()))
        for pair in torch.nonzero(
            torch.triu(undirected_bonds, diagonal=1),
            as_tuple=False,
        )
    }
    if residue_index is not None:
        # Protein residues have one CA-bearing token each.  Sorting those
        # tokens by within-chain residue index recovers peptide neighbours
        # even when ``token_bonds`` contains no standard polymer edges.
        for chain_id_tensor in torch.unique(asym_id):
            chain_id = int(chain_id_tensor.item())
            chain_ca_tokens = [
                token_id
                for token_id in ca_by_token
                if int(asym_id[token_id].item()) == chain_id
                and bool(is_protein[token_id])
            ]
            chain_ca_tokens.sort(
                key=lambda token_id: (
                    int(residue_index[token_id].item()),
                    token_id,
                )
            )
            for left_token, right_token in zip(
                chain_ca_tokens,
                chain_ca_tokens[1:],
            ):
                left_residue = int(residue_index[left_token].item())
                right_residue = int(residue_index[right_token].item())
                if right_residue != left_residue + 1:
                    continue
                candidate_pairs.add(
                    (
                        min(left_token, right_token),
                        max(left_token, right_token),
                    )
                )

    junctions: list[tuple[int, int]] = []
    for left_token, right_token in sorted(candidate_pairs):
        if asym_id[left_token] != asym_id[right_token]:
            continue
        if not (bool(is_protein[left_token]) and bool(is_protein[right_token])):
            continue
        if bool(fixed_token[left_token]) == bool(fixed_token[right_token]):
            continue
        if left_token not in ca_by_token or right_token not in ca_by_token:
            raise ValueError(
                "Every fixed/generated protein boundary token must have "
                "exactly one CA representative"
            )
        if bool(fixed_token[left_token]):
            fixed_atom = ca_by_token[left_token]
            generated_atom = ca_by_token[right_token]
        else:
            fixed_atom = ca_by_token[right_token]
            generated_atom = ca_by_token[left_token]
        junctions.append((fixed_atom, generated_atom))

    if not junctions:
        raise ValueError(
            "No fixed/generated protein boundaries were found from "
            "explicit token bonds or consecutive within-chain residues"
        )
    junction_pairs = torch.tensor(
        junctions,
        dtype=torch.long,
        device=device,
    )
    return BoundaryTopology(
        junction_pairs=junction_pairs,
        fixed_ca_atom_indices=torch.nonzero(
            fixed_topology_atom & is_ca,
            as_tuple=False,
        ).flatten(),
        generated_ca_atom_indices=torch.nonzero(
            ~fixed_topology_atom & is_ca,
            as_tuple=False,
        ).flatten(),
        generated_atom_mask=~fixed_topology_atom & considered_atom,
    )


def _normalize_direction(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-8:
        raise ValueError(f"{label} has no finite non-zero direction")
    direction = vector / norm
    largest = int(torch.argmax(torch.abs(direction)).item())
    if float(direction[largest].item()) < 0.0:
        direction = -direction
    return direction


def _normalized_transforms(
    sym_transforms: dict[Any, tuple[Any, Any]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    normalized: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for transform_id_raw, transform in sym_transforms.items():
        transform_id = int(transform_id_raw)
        if transform_id in normalized:
            raise ValueError(f"Duplicate symmetry transform ID {transform_id}")
        if not isinstance(transform, (tuple, list)) or len(transform) != 2:
            raise ValueError("Each symmetry transform must be (rotation, translation)")
        rotation = _as_tensor(transform[0])
        if not rotation.is_floating_point():
            rotation = rotation.to(dtype=torch.float64)
        translation = _as_tensor(
            transform[1],
            dtype=rotation.dtype,
            device=rotation.device,
        )
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "Symmetry rotations/translations must have shapes [3,3]/[3]"
            )
        if not (torch.isfinite(rotation).all() and torch.isfinite(translation).all()):
            raise ValueError("Symmetry transform contains NaN or Inf")
        identity = torch.eye(
            3,
            dtype=rotation.dtype,
            device=rotation.device,
        )
        if (
            not torch.allclose(
                rotation.T @ rotation,
                identity,
                atol=1e-5,
                rtol=1e-5,
            )
            or float(torch.linalg.det(rotation).item()) <= 0.0
        ):
            raise ValueError(
                f"Symmetry transform {transform_id} is not a proper rotation"
            )
        normalized[transform_id] = (rotation, translation)
    if len(normalized) < 2:
        raise ValueError("A cyclic transform set requires identity and a rotation")
    return normalized


def extract_cyclic_axis(
    sym_transforms: dict[Any, tuple[Any, Any]],
    *,
    tolerance: float = 1e-4,
) -> CyclicAxis:
    """Recover the common cyclic fixed line from proper rigid transforms."""

    if tolerance <= 0.0:
        raise ValueError("axis tolerance must be positive")
    transforms = _normalized_transforms(sym_transforms)
    ordered = sorted(transforms)
    reference_rotation = None
    reference_translation = None
    largest_angle = -1.0
    for transform_id in ordered:
        rotation, translation = transforms[transform_id]
        cosine = torch.clamp(
            (torch.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        angle = float(torch.acos(cosine).item())
        if angle > largest_angle and angle > tolerance:
            largest_angle = angle
            reference_rotation = rotation
            reference_translation = translation
    if reference_rotation is None or reference_translation is None:
        raise ValueError("A cyclic transform set requires a non-identity rotation")

    _, _, vh = torch.linalg.svd(
        reference_rotation
        - torch.eye(
            3,
            dtype=reference_rotation.dtype,
            device=reference_rotation.device,
        )
    )
    direction = _normalize_direction(
        vh[-1],
        label="cyclic rotation axis",
    )
    identity = torch.eye(
        3,
        dtype=reference_rotation.dtype,
        device=reference_rotation.device,
    )
    coefficient = torch.cat(
        (
            identity - reference_rotation,
            direction[None, :],
        ),
        dim=0,
    )
    right_hand_side = torch.cat(
        (
            reference_translation,
            torch.zeros(
                1,
                dtype=direction.dtype,
                device=direction.device,
            ),
        )
    )
    point = torch.linalg.lstsq(
        coefficient,
        right_hand_side,
    ).solution

    for transform_id in ordered:
        rotation, translation = transforms[transform_id]
        _, _, transform_vh = torch.linalg.svd(rotation - identity)
        if torch.linalg.matrix_norm(rotation - identity) > tolerance:
            candidate_direction = _normalize_direction(
                transform_vh[-1],
                label=f"transform {transform_id} axis",
            )
            if abs(float(torch.dot(candidate_direction, direction).item())) < (
                1.0 - tolerance
            ):
                raise ValueError("Symmetry transforms do not share one cyclic axis")
        residual = rotation @ point + translation - point
        if float(torch.linalg.vector_norm(residual).item()) > tolerance:
            raise ValueError("Symmetry transforms do not share one cyclic fixed line")

    return CyclicAxis(
        point=point,
        direction=direction,
        transform_ids=tuple(ordered),
    )


def extract_symmetry_primary_axis(
    sym_transforms: dict[Any, tuple[Any, Any]],
    *,
    symmetry_id: str | None,
    tolerance: float = 1e-4,
) -> CyclicAxis:
    """Resolve the primary Cn axis without assuming all group axes coincide.

    Cn consists entirely of one cyclic subgroup, so this is identical to
    :func:`extract_cyclic_axis`.  A proper Dn registry is ordered as
    ``e, r1, ..., r(n-1), s0, ..., s(n-1)`` by Mosaic's transform registry.
    Only the first ``n`` rotations share the principal axis; the secondary
    two-fold coset must remain in the full registry but must not be passed to
    the common-axis solver.
    """

    normalized_id = str(symmetry_id or "").upper()
    if len(normalized_id) < 2 or not normalized_id[1:].isdigit():
        raise ValueError(
            "Axis-dependent scaffold guidance requires a runtime Cn or Dn "
            "symmetry_id"
        )
    family = normalized_id[0]
    order = int(normalized_id[1:])
    if family not in {"C", "D"} or order < 2:
        raise ValueError(
            "Axis-dependent scaffold guidance currently supports Cn and Dn"
        )

    transforms = _normalized_transforms(sym_transforms)
    ordered_ids = tuple(sorted(transforms))
    expected_count = order if family == "C" else 2 * order
    if len(ordered_ids) != expected_count:
        raise ValueError(
            f"{normalized_id} requires {expected_count} runtime transforms, "
            f"observed {len(ordered_ids)}"
        )
    if family == "C":
        subgroup_ids = ordered_ids
    else:
        subgroup_ids = ordered_ids[:order]

    axis = extract_cyclic_axis(
        {transform_id: transforms[transform_id] for transform_id in subgroup_ids},
        tolerance=tolerance,
    )
    return CyclicAxis(
        point=axis.point,
        direction=axis.direction,
        transform_ids=tuple(subgroup_ids),
    )


def principal_axis_from_points(points: torch.Tensor) -> torch.Tensor:
    """Return a deterministic principal direction for one master motif."""

    coordinates = _as_tensor(points)
    if coordinates.ndim == 3:
        if coordinates.shape[0] != 1:
            raise ValueError("principal-axis estimation supports one pose batch")
        coordinates = coordinates[0]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("principal-axis points must have shape [M, 3]")
    if coordinates.shape[0] < 2:
        raise ValueError("principal-axis estimation requires at least two points")
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    if float(torch.linalg.vector_norm(centered).item()) <= 1e-8:
        raise ValueError("principal-axis points have degenerate coordinates")
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return _normalize_direction(vh[0], label="motif principal axis")


def _coordinate_matrix(
    coordinates: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    value = _as_tensor(coordinates)
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{label} supports one pose batch")
        value = value[0]
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{label} must have shape [L, 3] or [1, L, 3]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return value


def _zero_like_energy(coordinates: torch.Tensor) -> torch.Tensor:
    return coordinates.sum() * 0.0


def scaffold_orbit_energy(
    motif_coordinates: torch.Tensor,
    scaffold_coordinates: torch.Tensor,
    topology: BoundaryTopology,
    axis: CyclicAxis | None,
    *,
    principal_axis: torch.Tensor | None = None,
    pose_rotation: torch.Tensor | None = None,
    pose_translation: torch.Tensor | None = None,
    config: ScaffoldGuidanceConfig | None = None,
) -> ScaffoldGuidanceEnergy:
    """Evaluate junction, clash, tilt, and pose-prior penalties."""

    if config is None:
        config = ScaffoldGuidanceConfig()
    motif = _coordinate_matrix(
        motif_coordinates,
        label="motif_coordinates",
    )
    scaffold = _coordinate_matrix(
        scaffold_coordinates,
        label="scaffold_coordinates",
    )
    if motif.shape != scaffold.shape:
        raise ValueError("motif and scaffold coordinate shapes must match")
    device = motif.device
    dtype = motif.dtype
    junction_pairs = topology.junction_pairs.to(device=device)
    fixed_ca = topology.fixed_ca_atom_indices.to(device=device)
    generated_ca = topology.generated_ca_atom_indices.to(device=device)
    if not len(junction_pairs):
        raise ValueError("Scaffold guidance requires at least one junction")
    if not len(fixed_ca) or not len(generated_ca):
        raise ValueError("Scaffold guidance requires fixed and generated CA atoms")

    fixed_junction = motif[junction_pairs[:, 0]]
    generated_junction = scaffold[junction_pairs[:, 1]]
    junction_distances = torch.linalg.vector_norm(
        fixed_junction - generated_junction,
        dim=-1,
    )
    junction_error = junction_distances - config.junction_target_distance
    delta = torch.as_tensor(
        config.junction_huber_delta,
        dtype=dtype,
        device=device,
    )
    junction_term = torch.mean(
        delta * delta * (torch.sqrt(1.0 + torch.square(junction_error / delta)) - 1.0)
    )

    clash_distances = torch.cdist(
        motif[fixed_ca],
        scaffold[generated_ca],
    )
    bonded = torch.zeros_like(clash_distances, dtype=torch.bool)
    fixed_lookup = {
        int(atom_index): row for row, atom_index in enumerate(fixed_ca.tolist())
    }
    generated_lookup = {
        int(atom_index): column
        for column, atom_index in enumerate(generated_ca.tolist())
    }
    for pair in junction_pairs.tolist():
        fixed_row = fixed_lookup.get(int(pair[0]))
        generated_column = generated_lookup.get(int(pair[1]))
        if fixed_row is not None and generated_column is not None:
            bonded[fixed_row, generated_column] = True
    nonbonded_distances = clash_distances[~bonded]
    if nonbonded_distances.numel():
        clash_penalty = torch.relu(config.clash_distance - nonbonded_distances)
        clash_term = torch.mean(torch.square(clash_penalty))
        minimum_clash_distance = nonbonded_distances.min()
    else:
        clash_term = _zero_like_energy(motif)
        minimum_clash_distance = torch.full(
            (),
            float("inf"),
            dtype=dtype,
            device=device,
        )

    if pose_rotation is None:
        pose_rotation = torch.eye(3, dtype=dtype, device=device)
    if pose_translation is None:
        pose_translation = torch.zeros(3, dtype=dtype, device=device)
    rotation = _as_tensor(pose_rotation, dtype=dtype, device=device)
    translation = _as_tensor(pose_translation, dtype=dtype, device=device)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("pose_rotation/pose_translation must have shapes [3,3]/[3]")
    if principal_axis is None:
        tilt_term = _zero_like_energy(motif)
        tilt_degrees = _zero_like_energy(motif)
    else:
        if axis is None:
            raise ValueError(
                "A motif principal-axis tilt objective requires a Cn/Dn "
                "symmetry axis"
            )
        initial_direction = _normalize_direction(
            _as_tensor(
                principal_axis,
                dtype=dtype,
                device=device,
            ),
            label="motif principal axis",
        )
        cyclic_direction = _normalize_direction(
            axis.direction.to(dtype=dtype, device=device),
            label="cyclic axis",
        )
        moved_direction = _normalize_direction(
            rotation @ initial_direction,
            label="moved motif principal axis",
        )
        absolute_cosine = torch.clamp(
            torch.abs(torch.dot(moved_direction, cyclic_direction)),
            0.0,
            1.0,
        )
        maximum_tilt_cosine = math.cos(math.radians(config.maximum_tilt_degrees))
        tilt_excess = torch.relu(
            torch.as_tensor(
                maximum_tilt_cosine,
                dtype=dtype,
                device=device,
            )
            - absolute_cosine
        )
        tilt_term = torch.square(tilt_excess)
        tilt_degrees = torch.rad2deg(torch.acos(absolute_cosine))

    identity = torch.eye(3, dtype=dtype, device=device)
    translation_prior = torch.sum(
        torch.square(translation / config.translation_prior_scale)
    )
    rotation_scale = 2.0 * math.sin(
        math.radians(config.rotation_prior_scale_degrees) / 2.0
    )
    rotation_prior = torch.sum(torch.square(rotation - identity)) / (
        2.0 * rotation_scale * rotation_scale
    )
    prior_term = translation_prior + rotation_prior

    total = (
        config.junction_weight * junction_term
        + config.clash_weight * clash_term
        + config.tilt_weight * tilt_term
        + config.prior_weight * prior_term
    )
    return ScaffoldGuidanceEnergy(
        total=total,
        junction=junction_term,
        clash=clash_term,
        tilt=tilt_term,
        prior=prior_term,
        maximum_junction_error=torch.max(torch.abs(junction_error)),
        minimum_clash_distance=minimum_clash_distance,
        tilt_degrees=tilt_degrees,
        junction_distances=junction_distances,
        minimum_clash_distances=nonbonded_distances,
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _rotation_from_vector(vector: torch.Tensor) -> torch.Tensor:
    theta_squared = torch.sum(torch.square(vector))
    theta = torch.sqrt(theta_squared.clamp_min(1e-16))
    small = theta_squared < 1e-8
    sine_scale = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_squared * theta_squared / 120.0,
        torch.sin(theta) / theta,
    )
    cosine_scale = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_squared * theta_squared / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-16),
    )
    skew = _skew(vector)
    identity = torch.eye(
        3,
        dtype=vector.dtype,
        device=vector.device,
    )
    return identity + sine_scale * skew + cosine_scale * (skew @ skew)


def _rotation_axis_angle(
    rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.clamp(
        (torch.trace(rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle = torch.acos(cosine)
    if float(angle.detach().item()) < 1e-7:
        return torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=rotation.dtype,
            device=rotation.device,
        ), angle
    vector = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if float(torch.linalg.vector_norm(vector).detach().item()) < 1e-7:
        _, _, vh = torch.linalg.svd(
            rotation
            - torch.eye(
                3,
                dtype=rotation.dtype,
                device=rotation.device,
            )
        )
        axis = _normalize_direction(vh[-1], label="rotation axis")
    else:
        axis = vector / torch.linalg.vector_norm(vector)
    return axis, angle


def _clamp_vector(vector: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    if maximum_norm < 0.0:
        raise ValueError("vector norm bound cannot be negative")
    norm = torch.linalg.vector_norm(vector)
    if float(norm.detach().item()) <= maximum_norm:
        return vector
    return vector * (maximum_norm / max(float(norm.detach().item()), 1e-12))


def _clamp_rotation(
    rotation: torch.Tensor,
    maximum_angle_degrees: float,
) -> torch.Tensor:
    if maximum_angle_degrees < 0.0:
        raise ValueError("rotation bound cannot be negative")
    axis, angle = _rotation_axis_angle(rotation)
    maximum = math.radians(maximum_angle_degrees)
    if float(angle.detach().item()) <= maximum:
        return rotation
    return _rotation_from_vector(
        axis
        * torch.as_tensor(
            maximum,
            dtype=rotation.dtype,
            device=rotation.device,
        )
    )


def _energy_scalar(value: Any) -> torch.Tensor:
    if isinstance(value, ScaffoldGuidanceEnergy):
        value = value.total
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("energy_function must return one scalar tensor")
    return value.reshape(())


def _project_onto_basis(
    vector: torch.Tensor,
    basis: torch.Tensor | None,
    *,
    label: str,
) -> torch.Tensor:
    """Project one 3-vector onto the span of explicit physical axes."""

    if basis is None:
        return vector
    basis = _as_tensor(
        basis,
        dtype=vector.dtype,
        device=vector.device,
    )
    if basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError(f"{label} basis must have shape [K, 3]")
    if not torch.isfinite(basis).all():
        raise ValueError(f"{label} basis contains NaN or Inf")
    if basis.shape[0] == 0:
        return torch.zeros_like(vector)
    if int(torch.linalg.matrix_rank(basis).item()) != basis.shape[0]:
        raise ValueError(f"{label} basis vectors must be independent")
    orthonormal, _ = torch.linalg.qr(basis.T, mode="reduced")
    return orthonormal @ (orthonormal.T @ vector)


def propose_bounded_se3_step(
    current_rotation: torch.Tensor,
    current_translation: torch.Tensor,
    energy_function: Callable[[torch.Tensor, torch.Tensor], Any],
    *,
    maximum_step_translation: float,
    maximum_step_rotation_degrees: float,
    maximum_total_translation: float,
    maximum_total_rotation_degrees: float,
    translation_step_size: float | None = None,
    rotation_step_size_degrees: float | None = None,
    translation_basis: torch.Tensor | None = None,
    rotation_basis: torch.Tensor | None = None,
    line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25),
    deterministic_multistart: bool = False,
    selection_seed: int | None = None,
    minimum_best_gain_fraction: float = 0.75,
) -> SE3Proposal:
    """Take one bounded SE(3) proposal with a deterministic line search.

    ``deterministic_multistart`` supplements the local gradient with signed
    basis probes and coupled translation/rotation probes.  It is intended for
    the early capture phase, where a single gradient can be trapped by an
    initially poor pose.  The search is reproducible and remains inside both
    per-step and cumulative motion bounds.  When ``selection_seed`` is given,
    the accepted early proposal is sampled reproducibly from candidates whose
    energy gain is at least ``minimum_best_gain_fraction`` of the best gain.
    This dimensionless near-optimal set preserves pose diversity without ever
    accepting a non-improving candidate.
    """

    rotation = _as_tensor(current_rotation)
    if not rotation.is_floating_point():
        rotation = rotation.to(dtype=torch.float64)
    translation = _as_tensor(
        current_translation,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            "current_rotation/current_translation must have shapes [3,3]/[3]"
        )
    for name, value in (
        ("maximum_step_translation", maximum_step_translation),
        (
            "maximum_step_rotation_degrees",
            maximum_step_rotation_degrees,
        ),
        ("maximum_total_translation", maximum_total_translation),
        (
            "maximum_total_rotation_degrees",
            maximum_total_rotation_degrees,
        ),
    ):
        if value < 0.0:
            raise ValueError(f"{name} cannot be negative")
    if translation_step_size is None:
        translation_step_size = maximum_step_translation
    if rotation_step_size_degrees is None:
        rotation_step_size_degrees = maximum_step_rotation_degrees
    if translation_step_size < 0.0 or rotation_step_size_degrees < 0.0:
        raise ValueError("proposal step sizes cannot be negative")
    if not line_search_scales or any(scale <= 0.0 for scale in line_search_scales):
        raise ValueError("line_search_scales must be positive")
    if not 0.0 < minimum_best_gain_fraction <= 1.0:
        raise ValueError("minimum_best_gain_fraction must be in (0, 1]")
    if selection_seed is not None and selection_seed < 0:
        raise ValueError("selection_seed cannot be negative")

    with torch.enable_grad():
        rotation_vector = torch.zeros(
            3,
            dtype=rotation.dtype,
            device=rotation.device,
            requires_grad=True,
        )
        translation_delta = torch.zeros(
            3,
            dtype=translation.dtype,
            device=translation.device,
            requires_grad=True,
        )
        trial_rotation = _rotation_from_vector(rotation_vector) @ rotation
        trial_translation = translation + translation_delta
        initial_energy = _energy_scalar(
            energy_function(trial_rotation, trial_translation)
        )
        if not torch.isfinite(initial_energy):
            raise ValueError("energy_function returned NaN or Inf at the current pose")
        rotation_gradient, translation_gradient = torch.autograd.grad(
            initial_energy,
            (rotation_vector, translation_delta),
            allow_unused=True,
        )
    if rotation_gradient is None:
        rotation_gradient = torch.zeros_like(rotation_vector)
    if translation_gradient is None:
        translation_gradient = torch.zeros_like(translation_delta)
    if not (
        torch.isfinite(rotation_gradient).all()
        and torch.isfinite(translation_gradient).all()
    ):
        raise ValueError("Pose objective gradient contains NaN or Inf")

    rotation_direction = _project_onto_basis(
        -rotation_gradient.detach(),
        rotation_basis,
        label="rotation",
    )
    rotation_gradient_norm = float(
        torch.linalg.vector_norm(rotation_gradient.detach()).item()
    )
    rotation_norm = torch.linalg.vector_norm(rotation_direction)
    projected_rotation_gradient_norm = float(rotation_norm.item())
    if float(rotation_norm.item()) > 1e-12:
        rotation_direction = rotation_direction / rotation_norm
    translation_direction = _project_onto_basis(
        -translation_gradient.detach(),
        translation_basis,
        label="translation",
    )
    translation_gradient_norm = float(
        torch.linalg.vector_norm(translation_gradient.detach()).item()
    )
    translation_norm = torch.linalg.vector_norm(translation_direction)
    projected_translation_gradient_norm = float(translation_norm.item())
    if float(translation_norm.item()) > 1e-12:
        translation_direction = translation_direction / translation_norm

    raw_rotation_vector = rotation_direction * math.radians(rotation_step_size_degrees)
    raw_translation = translation_direction * translation_step_size

    def allowed_axes(
        basis: torch.Tensor | None,
        reference: torch.Tensor,
        *,
        label: str,
    ) -> tuple[torch.Tensor, ...]:
        if basis is None:
            matrix = torch.eye(
                3,
                dtype=reference.dtype,
                device=reference.device,
            )
        else:
            matrix = _as_tensor(
                basis,
                dtype=reference.dtype,
                device=reference.device,
            )
            if matrix.ndim != 2 or matrix.shape[1] != 3:
                raise ValueError(f"{label} basis must have shape [K, 3]")
            if matrix.shape[0] == 0:
                return ()
            orthonormal, _ = torch.linalg.qr(matrix.T, mode="reduced")
            matrix = orthonormal.T
        return tuple(matrix[index] for index in range(matrix.shape[0]))

    zero_rotation = torch.zeros_like(raw_rotation_vector)
    zero_translation = torch.zeros_like(raw_translation)
    directions: list[tuple[torch.Tensor, torch.Tensor, bool]] = [
        (raw_rotation_vector, raw_translation, True)
    ]
    if deterministic_multistart:
        translation_axes = allowed_axes(
            translation_basis,
            translation,
            label="translation",
        )
        rotation_axes = allowed_axes(
            rotation_basis,
            translation,
            label="rotation",
        )
        translation_probes = tuple(
            sign * axis * translation_step_size
            for axis in translation_axes
            for sign in (-1.0, 1.0)
            if translation_step_size > 0.0
        )
        rotation_probes = tuple(
            sign * axis * math.radians(rotation_step_size_degrees)
            for axis in rotation_axes
            for sign in (-1.0, 1.0)
            if rotation_step_size_degrees > 0.0
        )
        directions.extend((zero_rotation, probe, False) for probe in translation_probes)
        directions.extend((probe, zero_translation, False) for probe in rotation_probes)
        # Coupled probes let the optimizer cross a shallow saddle that neither
        # a translation-only nor a rotation-only immediate-descent test can
        # cross.  Pairing by index keeps the number of objective evaluations
        # bounded rather than taking a full Cartesian product.
        coupled_count = max(len(translation_probes), len(rotation_probes))
        if translation_probes and rotation_probes:
            directions.extend(
                (
                    rotation_probes[index % len(rotation_probes)],
                    translation_probes[index % len(translation_probes)],
                    False,
                )
                for index in range(coupled_count)
            )

    directions = [
        (
            _clamp_vector(
                rotation_probe,
                math.radians(maximum_step_rotation_degrees),
            ),
            _clamp_vector(translation_probe, maximum_step_translation),
            is_gradient,
        )
        for rotation_probe, translation_probe, is_gradient in directions
        if (
            float(torch.linalg.vector_norm(rotation_probe).detach().item()) > 1e-12
            or float(torch.linalg.vector_norm(translation_probe).detach().item())
            > 1e-12
        )
    ]

    initial_detached = initial_energy.detach()
    identity = torch.eye(
        3,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    line_search_trials: list[dict[str, float | bool | None]] = []
    improving_candidates: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            float,
            int,
        ]
    ] = []
    best = None
    for direction_index, (
        candidate_rotation_vector,
        candidate_translation_vector,
        is_gradient,
    ) in enumerate(directions):
        for scale in line_search_scales:
            delta_rotation = _rotation_from_vector(candidate_rotation_vector * scale)
            delta_translation = candidate_translation_vector * scale
            candidate_rotation = delta_rotation @ rotation
            candidate_translation = translation + delta_translation
            candidate_rotation = _clamp_rotation(
                candidate_rotation.detach(),
                maximum_total_rotation_degrees,
            )
            candidate_translation = _clamp_vector(
                candidate_translation.detach(),
                maximum_total_translation,
            )
            actual_delta_rotation = candidate_rotation @ rotation.T
            actual_delta_translation = candidate_translation - translation
            with torch.no_grad():
                candidate_energy = _energy_scalar(
                    energy_function(
                        candidate_rotation,
                        candidate_translation,
                    )
                ).detach()
            finite = bool(torch.isfinite(candidate_energy).item())
            candidate_value = float(candidate_energy.item()) if finite else None
            improves = bool(
                finite
                and candidate_value is not None
                and candidate_value < float(initial_detached.item())
            )
            trial_index = len(line_search_trials)
            line_search_trials.append(
                {
                    "direction_index": float(direction_index),
                    "gradient_direction": is_gradient,
                    "scale": float(scale),
                    "energy": candidate_value,
                    "finite": finite,
                    "improves": improves,
                    "selected": False,
                }
            )
            if not finite:
                continue
            if improves:
                candidate = (
                    candidate_rotation.detach(),
                    candidate_translation.detach(),
                    actual_delta_rotation.detach(),
                    actual_delta_translation.detach(),
                    candidate_energy,
                    float(scale),
                    trial_index,
                )
                if not deterministic_multistart:
                    best = candidate
                    break
                improving_candidates.append(candidate)
        if best is not None and not deterministic_multistart:
            break

    if deterministic_multistart and improving_candidates:
        best_energy = min(float(candidate[4]) for candidate in improving_candidates)
        initial_value = float(initial_detached)
        best_gain = initial_value - best_energy
        eligible = [
            candidate
            for candidate in improving_candidates
            if initial_value - float(candidate[4])
            >= minimum_best_gain_fraction * best_gain
        ]
        if selection_seed is None or len(eligible) == 1:
            best = min(eligible, key=lambda candidate: float(candidate[4]))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(selection_seed))
            selected_index = int(
                torch.randint(
                    len(eligible),
                    (1,),
                    generator=generator,
                    device="cpu",
                ).item()
            )
            best = eligible[selected_index]

    if best is not None:
        (
            best_rotation,
            best_translation,
            best_delta_rotation,
            best_delta_translation,
            best_energy,
            best_scale,
            best_trial_index,
        ) = best
        line_search_trials[best_trial_index]["selected"] = True
        return SE3Proposal(
            rotation=best_rotation,
            translation=best_translation,
            delta_rotation=best_delta_rotation,
            delta_translation=best_delta_translation,
            initial_energy=initial_detached,
            proposed_energy=best_energy,
            accepted=True,
            line_search_scale=best_scale,
            rotation_gradient_norm=rotation_gradient_norm,
            translation_gradient_norm=translation_gradient_norm,
            projected_rotation_gradient_norm=(projected_rotation_gradient_norm),
            projected_translation_gradient_norm=(projected_translation_gradient_norm),
            line_search_trials=tuple(line_search_trials),
        )

    return SE3Proposal(
        rotation=rotation.detach().clone(),
        translation=translation.detach().clone(),
        delta_rotation=identity,
        delta_translation=torch.zeros_like(translation),
        initial_energy=initial_detached,
        proposed_energy=initial_detached,
        accepted=False,
        line_search_scale=0.0,
        rotation_gradient_norm=rotation_gradient_norm,
        translation_gradient_norm=translation_gradient_norm,
        projected_rotation_gradient_norm=projected_rotation_gradient_norm,
        projected_translation_gradient_norm=(projected_translation_gradient_norm),
        line_search_trials=tuple(line_search_trials),
    )


def expand_master_orbit(
    master_coordinates: torch.Tensor,
    sym_transforms: dict[Any, tuple[Any, Any]],
) -> torch.Tensor:
    """Return one transformed copy of the master for every group action.

    The output has shape ``[G, D, M, 3]``.  Keeping the group-action dimension
    explicit makes it difficult for callers to accidentally apply one global
    rigid motion to the complete symmetry-orbit union.
    """

    master = _as_tensor(master_coordinates)
    if not master.is_floating_point():
        master = master.to(dtype=torch.float64)
    if master.ndim == 2:
        master = master[None, ...]
    if master.ndim != 3 or master.shape[-1] != 3:
        raise ValueError("master_coordinates must have shape [M, 3] or [D, M, 3]")
    transforms = _normalized_transforms(sym_transforms)
    copies = []
    for transform_id in sorted(transforms):
        rotation, translation = transforms[transform_id]
        rotation = rotation.to(dtype=master.dtype, device=master.device)
        translation = translation.to(
            dtype=master.dtype,
            device=master.device,
        )
        copies.append(master @ rotation.T + translation)
    return torch.stack(copies, dim=0)


def insert_master_orbit(
    base_coordinates: torch.Tensor,
    master_coordinates: torch.Tensor,
    group_atom_indices: torch.Tensor,
    group_transform_ids: torch.Tensor,
    sym_transforms: dict[Any, tuple[Any, Any]],
) -> torch.Tensor:
    """Insert one master coordinate set into dense atom coordinates."""

    base = _as_tensor(base_coordinates)
    master = _as_tensor(
        master_coordinates,
        dtype=base.dtype,
        device=base.device,
    )
    squeeze = False
    if base.ndim == 2:
        base = base[None, ...]
        squeeze = True
    if master.ndim == 2:
        master = master[None, ...]
    if (
        base.ndim != 3
        or master.ndim != 3
        or base.shape[0] != master.shape[0]
        or base.shape[-1] != 3
        or master.shape[-1] != 3
    ):
        raise ValueError(
            "base/master coordinates must have compatible [D, L/M, 3] shapes"
        )
    indices = _as_tensor(
        group_atom_indices,
        dtype=torch.long,
        device=base.device,
    )
    transform_ids = _as_tensor(
        group_transform_ids,
        dtype=torch.long,
        device=base.device,
    )
    if (
        indices.ndim != 2
        or indices.shape[1] != master.shape[1]
        or transform_ids.shape != (indices.shape[0],)
    ):
        raise ValueError("group indices/transform IDs do not match master coordinates")
    transforms = _normalized_transforms(sym_transforms)
    expanded = base.clone()
    for group_row, transform_id_tensor in enumerate(transform_ids):
        transform_id = int(transform_id_tensor.item())
        if transform_id not in transforms:
            raise ValueError(f"Unknown symmetry transform ID {transform_id}")
        rotation, translation = transforms[transform_id]
        rotation = rotation.to(dtype=base.dtype, device=base.device)
        translation = translation.to(dtype=base.dtype, device=base.device)
        expanded[:, indices[group_row], :] = master @ rotation.T + translation
    return expanded[0] if squeeze else expanded
