"""Bounded rigid motion for complete cross-chain motif symmetry orbits.

The controller is deliberately separate from atomwise diffusion.  One master
interface pose is estimated from all symmetry-related observations, clamped,
and then expanded through the runtime group action.  Individual fragments or
copies never receive independent rigid motions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


def mobility_window_weight(
    progress: float,
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    """Smoothly ramp motion in/out and freeze the final denoising steps."""

    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError(
            "mobility fractions must satisfy "
            "0 <= start_fraction < end_fraction <= 1"
        )
    if progress <= start_fraction or progress >= end_fraction:
        return 0.0
    unit = (progress - start_fraction) / (
        end_fraction - start_fraction
    )
    return math.sin(math.pi * unit) ** 2


def _apply_frame(points, rotation, translation):
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation


def _invert_frame(points, rotation, translation):
    return torch.matmul(points - translation, rotation)


def _proper_rotation_from_cross_covariance(covariance):
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.eye(
        3,
        dtype=covariance.dtype,
        device=covariance.device,
    ).expand(covariance.shape[0], 3, 3).clone()
    correction[:, -1, -1] = torch.sign(
        torch.linalg.det(torch.matmul(u, vh))
    )
    return torch.matmul(torch.matmul(u, correction), vh)


def fit_centered_rigid_pose(template, proposal):
    """Fit ``proposal = (template-c) @ R.T + c + t`` in batch."""

    if template.ndim != 3 or proposal.ndim != 3:
        raise ValueError("template and proposal must have shape [D, M, 3]")
    if template.shape[-2:] != proposal.shape[-2:]:
        raise ValueError("template and proposal atom dimensions must match")
    if template.shape[0] == 1 and proposal.shape[0] > 1:
        template = template.expand(proposal.shape[0], -1, -1)
    if template.shape[0] != proposal.shape[0]:
        raise ValueError("template and proposal batches must match")
    template_center = template.mean(dim=1)
    proposal_center = proposal.mean(dim=1)
    centered_template = template - template_center[:, None, :]
    centered_proposal = proposal - proposal_center[:, None, :]
    covariance = torch.matmul(
        centered_template.transpose(-1, -2),
        centered_proposal,
    )
    row_rotation = _proper_rotation_from_cross_covariance(covariance)
    rotation = row_rotation.transpose(-1, -2)
    translation = proposal_center - template_center
    fitted = (
        torch.matmul(
            centered_template,
            rotation.transpose(-1, -2),
        )
        + template_center[:, None, :]
        + translation[:, None, :]
    )
    rmsd = torch.sqrt(
        torch.mean(
            torch.sum(torch.square(fitted - proposal), dim=-1),
            dim=-1,
        )
    )
    return rotation, translation, rmsd


def _axis_angle(rotation):
    """Convert one proper 3x3 rotation to a stable axis and angle."""

    cosine = torch.clamp(
        (torch.trace(rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle = torch.acos(cosine)
    angle_value = float(angle.item())
    if angle_value < 1e-7:
        return torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=rotation.dtype,
            device=rotation.device,
        ), angle
    if abs(math.pi - angle_value) > 1e-4:
        axis = torch.stack(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-8)
        return axis, angle

    diagonal = torch.clamp(
        (torch.diagonal(rotation) + 1.0) / 2.0,
        min=0.0,
    )
    largest = int(torch.argmax(diagonal).item())
    axis = torch.zeros(
        3,
        dtype=rotation.dtype,
        device=rotation.device,
    )
    axis[largest] = torch.sqrt(diagonal[largest]).clamp_min(1e-8)
    other = [index for index in range(3) if index != largest]
    axis[other[0]] = (
        rotation[largest, other[0]]
        + rotation[other[0], largest]
    ) / (4.0 * axis[largest])
    axis[other[1]] = (
        rotation[largest, other[1]]
        + rotation[other[1], largest]
    ) / (4.0 * axis[largest])
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-8)
    return axis, angle


def _rotation_from_axis_angle(axis, angle):
    x, y, z = axis
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return (
        identity
        + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * torch.matmul(skew, skew)
    )


def _scaled_rotation(rotation, maximum_angle_radians, response):
    axis, angle = _axis_angle(rotation)
    target_angle = torch.minimum(
        angle * response,
        torch.as_tensor(
            maximum_angle_radians,
            dtype=angle.dtype,
            device=angle.device,
        ),
    )
    return _rotation_from_axis_angle(axis, target_angle)


def _clamp_vector(vector, maximum_norm):
    norm = torch.linalg.vector_norm(vector)
    scale = min(
        1.0,
        maximum_norm / max(float(norm.item()), 1e-12),
    )
    return vector * scale


@dataclass
class OrbitRigidPoseState:
    """Pose parameters for one motif orbit and diffusion batch."""

    rotation: torch.Tensor
    translation: torch.Tensor
    last_proposal_rmsd: torch.Tensor


@dataclass
class OrbitRigidMotif:
    group_indices: torch.Tensor
    group_atom_indices: torch.Tensor
    group_transform_ids: torch.Tensor
    template_master: torch.Tensor
    maximum_translation: float
    maximum_rotation_degrees: float
    state: OrbitRigidPoseState


class OrbitRigidMotifController:
    """Maintain bounded master-interface poses and exact symmetry copies."""

    def __init__(
        self,
        *,
        motifs: list[OrbitRigidMotif],
        sym_transforms: dict[int, tuple[torch.Tensor, torch.Tensor]],
        base_target: torch.Tensor,
        start_fraction: float = 0.10,
        end_fraction: float = 0.85,
        response: float = 0.25,
        per_step_translation: float = 0.25,
        per_step_rotation_degrees: float = 1.0,
    ):
        if not 0.0 < response <= 1.0:
            raise ValueError("motif mobility response must be in (0, 1]")
        if per_step_translation <= 0.0:
            raise ValueError("per-step translation bound must be positive")
        if per_step_rotation_degrees <= 0.0:
            raise ValueError("per-step rotation bound must be positive")
        mobility_window_weight(
            0.5,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
        )
        self.motifs = motifs
        self.sym_transforms = sym_transforms
        self.base_target = base_target
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.response = response
        self.per_step_translation = per_step_translation
        self.per_step_rotation_degrees = per_step_rotation_degrees

    @classmethod
    def from_features(
        cls,
        f: dict[str, Any],
        fixed_target: torch.Tensor,
        **kwargs,
    ) -> "OrbitRigidMotifController | None":
        mobility_modes = f.get("motif_constraint_orbit_mobility_mode")
        if mobility_modes is None:
            return None
        mobility_modes = torch.as_tensor(
            mobility_modes,
            dtype=torch.long,
            device=fixed_target.device,
        )
        mobile_orbits = torch.nonzero(
            mobility_modes == 1,
            as_tuple=False,
        ).flatten()
        if not len(mobile_orbits):
            return None

        group_orbits = torch.as_tensor(
            f["motif_constraint_group_orbit_index"],
            dtype=torch.long,
            device=fixed_target.device,
        )
        group_transform_ids = torch.as_tensor(
            f["motif_constraint_group_orbit_transform_id"],
            dtype=torch.long,
            device=fixed_target.device,
        )
        group_atom_indices = torch.as_tensor(
            f["motif_constraint_group_atom_indices"],
            dtype=torch.long,
            device=fixed_target.device,
        )
        group_atom_mask = torch.as_tensor(
            f["motif_constraint_group_atom_mask"],
            dtype=torch.bool,
            device=fixed_target.device,
        )
        master_group_indices = torch.as_tensor(
            f["motif_constraint_orbit_master_group_index"],
            dtype=torch.long,
            device=fixed_target.device,
        )
        bounds = torch.as_tensor(
            f["motif_constraint_orbit_bounds"],
            dtype=torch.float32,
            device=fixed_target.device,
        )
        sym_transforms = {
            int(transform_id): (
                torch.as_tensor(
                    transform[0],
                    dtype=fixed_target.dtype,
                    device=fixed_target.device,
                ),
                torch.as_tensor(
                    transform[1],
                    dtype=fixed_target.dtype,
                    device=fixed_target.device,
                ),
            )
            for transform_id, transform in f["sym_transform"].items()
        }

        motifs = []
        for orbit_index_tensor in mobile_orbits:
            orbit_index = int(orbit_index_tensor.item())
            orbit_group_indices = torch.nonzero(
                group_orbits == orbit_index,
                as_tuple=False,
            ).flatten()
            master_group_index = int(
                master_group_indices[orbit_index].item()
            )
            master_mask = group_atom_mask[master_group_index]
            master_indices = group_atom_indices[
                master_group_index,
                master_mask,
            ]
            atom_count = len(master_indices)
            compact_indices = []
            for group_index_tensor in orbit_group_indices:
                group_index = int(group_index_tensor.item())
                valid = group_atom_mask[group_index]
                indices = group_atom_indices[group_index, valid]
                if len(indices) != atom_count:
                    raise ValueError(
                        "All mobile motif groups must have equal atom counts"
                    )
                compact_indices.append(indices)
            compact_indices_tensor = torch.stack(
                compact_indices,
                dim=0,
            )
            template_master = fixed_target[:, master_indices, :].clone()
            batch_size = fixed_target.shape[0]
            motifs.append(
                OrbitRigidMotif(
                    group_indices=orbit_group_indices,
                    group_atom_indices=compact_indices_tensor,
                    group_transform_ids=group_transform_ids[
                        orbit_group_indices
                    ],
                    template_master=template_master,
                    maximum_translation=float(bounds[orbit_index, 0]),
                    maximum_rotation_degrees=float(
                        bounds[orbit_index, 1]
                    ),
                    state=OrbitRigidPoseState(
                        rotation=torch.eye(
                            3,
                            dtype=fixed_target.dtype,
                            device=fixed_target.device,
                        )
                        .expand(batch_size, 3, 3)
                        .clone(),
                        translation=torch.zeros(
                            (batch_size, 3),
                            dtype=fixed_target.dtype,
                            device=fixed_target.device,
                        ),
                        last_proposal_rmsd=torch.zeros(
                            batch_size,
                            dtype=fixed_target.dtype,
                            device=fixed_target.device,
                        ),
                    ),
                )
            )
        return cls(
            motifs=motifs,
            sym_transforms=sym_transforms,
            base_target=fixed_target.clone(),
            **kwargs,
        )

    @staticmethod
    def _master_coordinates(motif: OrbitRigidMotif):
        center = motif.template_master.mean(dim=1)
        centered = motif.template_master - center[:, None, :]
        return (
            torch.matmul(
                centered,
                motif.state.rotation.transpose(-1, -2),
            )
            + center[:, None, :]
            + motif.state.translation[:, None, :]
        )

    def _inverse_average_proposal(self, motif, raw_coordinates):
        canonical_copies = []
        for group_row, transform_id_tensor in enumerate(
            motif.group_transform_ids
        ):
            transform_id = int(transform_id_tensor.item())
            rotation, translation = self.sym_transforms[transform_id]
            observed = raw_coordinates[
                :,
                motif.group_atom_indices[group_row],
                :,
            ]
            canonical_copies.append(
                _invert_frame(observed, rotation, translation)
            )
        return torch.stack(canonical_copies, dim=0).mean(dim=0)

    def update(
        self,
        raw_coordinates: torch.Tensor,
        *,
        progress: float,
    ) -> torch.Tensor:
        """Update mobile poses and return one dense fixed-target tensor."""

        raw_coordinates = raw_coordinates.to(
            dtype=self.base_target.dtype,
            device=self.base_target.device,
        )
        if raw_coordinates.shape != self.base_target.shape:
            raise ValueError(
                "Mobility proposal coordinates must match the fixed target "
                "shape"
            )
        if not torch.isfinite(raw_coordinates).all():
            raise ValueError(
                "Mobility proposal coordinates contain NaN or Inf"
            )
        window = mobility_window_weight(
            progress,
            start_fraction=self.start_fraction,
            end_fraction=self.end_fraction,
        )
        if window > 0.0:
            for motif in self.motifs:
                proposal = self._inverse_average_proposal(
                    motif,
                    raw_coordinates,
                )
                desired_rotation, desired_translation, proposal_rmsd = (
                    fit_centered_rigid_pose(
                        motif.template_master,
                        proposal,
                    )
                )
                if not (
                    torch.isfinite(desired_rotation).all()
                    and torch.isfinite(desired_translation).all()
                    and torch.isfinite(proposal_rmsd).all()
                ):
                    raise ValueError(
                        "Rigid motif pose fitting produced NaN or Inf"
                    )
                updated_rotations = []
                updated_translations = []
                for batch_index in range(raw_coordinates.shape[0]):
                    current_rotation = motif.state.rotation[batch_index]
                    relative_rotation = (
                        desired_rotation[batch_index]
                        @ current_rotation.T
                    )
                    increment = _scaled_rotation(
                        relative_rotation,
                        math.radians(
                            self.per_step_rotation_degrees * window
                        ),
                        self.response * window,
                    )
                    rotation = increment @ current_rotation
                    rotation = _scaled_rotation(
                        rotation,
                        math.radians(
                            motif.maximum_rotation_degrees
                        ),
                        1.0,
                    )
                    delta_translation = (
                        desired_translation[batch_index]
                        - motif.state.translation[batch_index]
                    ) * (self.response * window)
                    delta_translation = _clamp_vector(
                        delta_translation,
                        self.per_step_translation * window,
                    )
                    translation = (
                        motif.state.translation[batch_index]
                        + delta_translation
                    )
                    translation = _clamp_vector(
                        translation,
                        motif.maximum_translation,
                    )
                    updated_rotations.append(rotation)
                    updated_translations.append(translation)
                motif.state = OrbitRigidPoseState(
                    rotation=torch.stack(updated_rotations),
                    translation=torch.stack(updated_translations),
                    last_proposal_rmsd=proposal_rmsd,
                )

        target = self.base_target.clone()
        for motif in self.motifs:
            master_coordinates = self._master_coordinates(motif)
            for group_row, transform_id_tensor in enumerate(
                motif.group_transform_ids
            ):
                transform_id = int(transform_id_tensor.item())
                rotation, translation = self.sym_transforms[transform_id]
                target[
                    :,
                    motif.group_atom_indices[group_row],
                    :,
                ] = _apply_frame(
                    master_coordinates,
                    rotation,
                    translation,
                )
        return target
