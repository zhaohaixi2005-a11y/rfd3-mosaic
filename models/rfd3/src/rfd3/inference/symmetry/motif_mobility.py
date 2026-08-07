"""Bounded rigid motion for complete motif symmetry orbits.

The controller is deliberately separate from atomwise diffusion.  One master
master pose is estimated from all symmetry-related observations, clamped,
and then expanded through the runtime group action.  Individual fragments or
copies never receive independent rigid motions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from rfd3.inference.symmetry.constraint_orbit import (
    ConstraintOrbitLayout,
)
from rfd3.inference.symmetry.scaffold_guidance import (
    BoundaryTopology,
    CyclicAxis,
    ScaffoldGuidanceConfig,
    insert_master_orbit,
    propose_bounded_se3_step,
    scaffold_orbit_energy,
)


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
    constraint_orbit_id: str
    coupling_group_id: str
    group_indices: torch.Tensor
    group_atom_indices: torch.Tensor
    group_transform_ids: torch.Tensor
    master_atom_indices: torch.Tensor
    template_master: torch.Tensor
    maximum_translation: float
    maximum_rotation_degrees: float
    mobility_subspace: str
    proposal_source: str
    objective_ids: tuple[str, ...]
    start_fraction: float
    end_fraction: float
    response: float
    per_step_translation: float
    per_step_rotation_degrees: float
    state: OrbitRigidPoseState


class OrbitRigidMotifController:
    """Maintain bounded master-orbit poses and exact symmetry copies."""

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
        self.update_calls = 0
        self.active_window_calls = 0
        self.last_update_applied = False
        self._diagnostic_trajectory: list[dict[str, Any]] = []

    @classmethod
    def from_features(
        cls,
        f: dict[str, Any],
        fixed_target: torch.Tensor,
        **kwargs,
    ) -> "OrbitRigidMotifController | None":
        layout = ConstraintOrbitLayout.from_features(
            f,
            atom_count=fixed_target.shape[1],
            device=fixed_target.device,
        )
        if layout is None or not layout.mobile_orbits:
            return None
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
        default_schedule = (
            float(kwargs.get("start_fraction", 0.10)),
            float(kwargs.get("end_fraction", 0.85)),
            float(kwargs.get("response", 0.25)),
            float(kwargs.get("per_step_translation", 0.25)),
            float(kwargs.get("per_step_rotation_degrees", 1.0)),
        )
        for orbit in layout.mobile_orbits:
            if orbit.mobility_subspace not in {
                "bounded_se3",
                "radial",
                "radial_axial",
            }:
                raise ValueError(
                    "Dynamic motif conditioning does not yet execute "
                    f"mobility_subspace={orbit.mobility_subspace!r}"
                )
            if orbit.proposal_source == "hoyeung_drag_compat":
                raise ValueError(
                    "hoyeung_drag_compat is an explicit migration marker, "
                    "not a native RFD3 runtime proposal; use "
                    "denoiser_fit or scaffold_objectives"
                )
            if orbit.proposal_source not in {
                "denoiser_fit",
                "scaffold_objectives",
            }:
                raise ValueError(
                    "A mobile constraint orbit has no executable proposal "
                    "source"
                )
            if (
                orbit.mobility_subspace in {"radial", "radial_axial"}
                and orbit.proposal_source != "scaffold_objectives"
            ):
                raise ValueError(
                    f"{orbit.mobility_subspace} mobility requires the "
                    "topology-defined axis provided by scaffold_objectives"
                )
            schedule = orbit.schedule or default_schedule
            orbit_group_indices = torch.tensor(
                orbit.group_indices,
                dtype=torch.long,
                device=fixed_target.device,
            )
            master_indices = layout.groups[
                orbit.master_group_index
            ].atom_indices
            atom_count = len(master_indices)
            compact_indices = []
            for group_index in orbit.group_indices:
                indices = layout.groups[group_index].atom_indices
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
                    constraint_orbit_id=orbit.constraint_orbit_id,
                    coupling_group_id=orbit.coupling_group_id,
                    group_indices=orbit_group_indices,
                    group_atom_indices=compact_indices_tensor,
                    group_transform_ids=torch.tensor(
                        orbit.transform_ids,
                        dtype=torch.long,
                        device=fixed_target.device,
                    ),
                    master_atom_indices=master_indices,
                    template_master=template_master,
                    maximum_translation=orbit.maximum_translation,
                    maximum_rotation_degrees=(
                        orbit.maximum_rotation_degrees
                    ),
                    mobility_subspace=orbit.mobility_subspace,
                    proposal_source=orbit.proposal_source,
                    objective_ids=orbit.objective_ids,
                    start_fraction=schedule[0],
                    end_fraction=schedule[1],
                    response=schedule[2],
                    per_step_translation=schedule[3],
                    per_step_rotation_degrees=schedule[4],
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

    @staticmethod
    def _master_coordinates_for_pose(
        motif: OrbitRigidMotif,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        if motif.template_master.shape[0] != 1:
            raise ValueError(
                "Scaffold-derived motif guidance supports one pose batch"
            )
        center = motif.template_master[0].mean(dim=0)
        centered = motif.template_master[0] - center[None, :]
        return (
            centered @ rotation.T
            + center[None, :]
            + translation[None, :]
        )

    def materialize_target(self) -> torch.Tensor:
        """Return the dense fixed target for the current master poses."""

        target = self.base_target.clone()
        for motif in self.motifs:
            master_coordinates = self._master_coordinates(motif)
            target = insert_master_orbit(
                target,
                master_coordinates,
                motif.group_atom_indices,
                motif.group_transform_ids,
                self.sym_transforms,
            )
        return target

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
        self.update_calls += 1
        self.last_update_applied = False
        windows = [
            mobility_window_weight(
                progress,
                start_fraction=motif.start_fraction,
                end_fraction=motif.end_fraction,
            )
            for motif in self.motifs
        ]
        window = max(windows, default=0.0)
        if window > 0.0:
            self.active_window_calls += 1
            for motif, motif_window in zip(self.motifs, windows):
                if motif_window <= 0.0:
                    continue
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
                            motif.per_step_rotation_degrees * motif_window
                        ),
                        motif.response * motif_window,
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
                    ) * (motif.response * motif_window)
                    delta_translation = _clamp_vector(
                        delta_translation,
                        motif.per_step_translation * motif_window,
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
                self.last_update_applied = True
        target = self.materialize_target()
        self._diagnostic_trajectory.append(
            self._diagnostic_snapshot(
                progress=progress,
                window_weight=window,
                extra={
                    "proposal_source": "denoiser",
                    "applied": self.last_update_applied,
                },
            )
        )
        return target

    def update_from_scaffold(
        self,
        scaffold_coordinates: torch.Tensor,
        *,
        progress: float,
        topology: BoundaryTopology,
        axis: CyclicAxis,
        principal_axis: torch.Tensor,
        config: ScaffoldGuidanceConfig,
        apply_update: bool,
    ) -> torch.Tensor:
        """Compatibility wrapper for one scaffold-driven motif orbit."""

        if len(self.motifs) != 1:
            raise ValueError(
                "update_from_scaffold is the single-orbit compatibility "
                "entry point; use update_orbits_from_scaffold for multiple "
                "mobile motif orbits"
            )
        return self.update_orbits_from_scaffold(
            scaffold_coordinates,
            progress=progress,
            topology=topology,
            axis=axis,
            principal_axes=(principal_axis,),
            config=config,
            apply_update=apply_update,
        )

    def _joint_scaffold_energy(
        self,
        target: torch.Tensor,
        scaffold: torch.Tensor,
        *,
        topology: BoundaryTopology,
        axis: CyclicAxis,
        principal_axes: tuple[torch.Tensor, ...],
        rotations: tuple[torch.Tensor, ...],
        translations: tuple[torch.Tensor, ...],
        config: ScaffoldGuidanceConfig,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Evaluate assembly geometry once and one pose prior per orbit."""

        if not (
            len(self.motifs)
            == len(principal_axes)
            == len(rotations)
            == len(translations)
        ):
            raise ValueError(
                "Joint scaffold energy inputs must match the mobile orbit "
                "count"
            )
        geometry_config = replace(
            config,
            tilt_weight=0.0,
            prior_weight=0.0,
        )
        geometry = scaffold_orbit_energy(
            target,
            scaffold,
            topology,
            axis,
            config=geometry_config,
        )
        inter_orbit_clash, minimum_inter_orbit_distance = (
            self._inter_orbit_clash_energy(
                target,
                topology=topology,
                clash_distance=config.clash_distance,
            )
        )
        combined_clash = geometry.clash + inter_orbit_clash
        total = geometry.total + config.clash_weight * inter_orbit_clash
        orbit_terms = []
        pose_config = replace(
            config,
            junction_weight=0.0,
            clash_weight=0.0,
        )
        for motif, principal_axis, rotation, translation in zip(
            self.motifs,
            principal_axes,
            rotations,
            translations,
        ):
            pose = scaffold_orbit_energy(
                target,
                scaffold,
                topology,
                axis,
                principal_axis=principal_axis,
                pose_rotation=rotation,
                pose_translation=translation,
                config=pose_config,
            )
            total = total + pose.total
            orbit_terms.append(
                {
                    "tilt": float(pose.tilt.detach().cpu().item()),
                    "weighted_tilt": float(
                        (config.tilt_weight * pose.tilt)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "prior": float(pose.prior.detach().cpu().item()),
                    "weighted_prior": float(
                        (config.prior_weight * pose.prior)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "tilt_degrees": float(
                        pose.tilt_degrees.detach().cpu().item()
                    ),
                }
            )
        minimum_clash_distance = torch.minimum(
            geometry.minimum_clash_distance,
            minimum_inter_orbit_distance,
        )
        return total, {
            "total": float(total.detach().cpu().item()),
            "junction": float(geometry.junction.detach().cpu().item()),
            "weighted_junction": float(
                (config.junction_weight * geometry.junction)
                .detach()
                .cpu()
                .item()
            ),
            "clash": float(combined_clash.detach().cpu().item()),
            "scaffold_clash": float(
                geometry.clash.detach().cpu().item()
            ),
            "inter_orbit_clash": float(
                inter_orbit_clash.detach().cpu().item()
            ),
            "weighted_clash": float(
                (config.clash_weight * combined_clash)
                .detach()
                .cpu()
                .item()
            ),
            "maximum_junction_error": float(
                geometry.maximum_junction_error.detach().cpu().item()
            ),
            "minimum_clash_distance": float(
                minimum_clash_distance.detach().cpu().item()
            ),
            "minimum_inter_orbit_distance": float(
                minimum_inter_orbit_distance.detach().cpu().item()
            ),
            "orbits": orbit_terms,
        }

    def _inter_orbit_clash_energy(
        self,
        target: torch.Tensor,
        *,
        topology: BoundaryTopology,
        clash_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalize CA overlap between independently mobile motif orbits.

        The ordinary scaffold objective compares fixed motif atoms with the
        generated scaffold.  With several independently mobile orbits that is
        insufficient: two individually acceptable proposals can collide only
        after their poses are materialized together.  This term is evaluated
        only by the joint objective, so it preserves Jacobi proposal semantics
        while allowing the atomic acceptance decision to reject that conflict.
        """

        if target.ndim != 2 or target.shape[-1] != 3:
            raise ValueError("Joint motif target must have shape [L, 3]")
        zero = target.sum() * 0.0
        if len(self.motifs) < 2:
            return zero, torch.full(
                (),
                float("inf"),
                dtype=target.dtype,
                device=target.device,
            )

        fixed_ca = {
            int(value)
            for value in (
                topology.fixed_ca_atom_indices.detach().cpu().tolist()
            )
        }
        orbit_ca_indices: list[torch.Tensor] = []
        for motif in self.motifs:
            indices = sorted(
                {
                    int(value)
                    for value in motif.group_atom_indices.detach()
                    .cpu()
                    .flatten()
                    .tolist()
                    if int(value) in fixed_ca
                }
            )
            orbit_ca_indices.append(
                torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=target.device,
                )
            )

        penalties = []
        minimum_distances = []
        for left_index in range(len(orbit_ca_indices)):
            left = orbit_ca_indices[left_index]
            if not len(left):
                continue
            for right_index in range(left_index + 1, len(orbit_ca_indices)):
                right = orbit_ca_indices[right_index]
                if not len(right):
                    continue
                distances = torch.cdist(target[left], target[right])
                penalties.append(
                    torch.mean(
                        torch.square(
                            torch.relu(clash_distance - distances)
                        )
                    )
                )
                minimum_distances.append(distances.min())
        if not penalties:
            return zero, torch.full(
                (),
                float("inf"),
                dtype=target.dtype,
                device=target.device,
            )
        return (
            torch.stack(penalties).mean(),
            torch.stack(minimum_distances).min(),
        )

    def update_orbits_from_scaffold(
        self,
        scaffold_coordinates: torch.Tensor,
        *,
        progress: float,
        topology: BoundaryTopology,
        axis: CyclicAxis,
        principal_axes: tuple[torch.Tensor, ...],
        config: ScaffoldGuidanceConfig,
        apply_update: bool,
    ) -> torch.Tensor:
        """Jointly propose and atomically apply scaffold-driven orbit poses.

        Every orbit proposal is computed from the same immutable pose
        snapshot.  Candidate poses are then materialized together and are
        accepted only when the joint assembly objective improves.  This
        Jacobi-style update makes the result independent of declaration
        order and prevents one orbit from observing another orbit's partially
        committed state.
        """

        if not self.motifs:
            raise ValueError("Scaffold guidance requires a mobile motif orbit")
        if len(principal_axes) != len(self.motifs):
            raise ValueError(
                "principal_axes must contain one axis per mobile motif orbit"
            )
        if self.base_target.shape[0] != 1:
            raise ValueError(
                "Scaffold-derived motif guidance supports one pose batch"
            )
        scaffold = scaffold_coordinates.to(
            dtype=self.base_target.dtype,
            device=self.base_target.device,
        )
        if scaffold.shape != self.base_target.shape:
            raise ValueError(
                "Scaffold guidance coordinates must match the fixed target "
                "shape"
            )
        if not torch.isfinite(scaffold).all():
            raise ValueError(
                "Scaffold guidance coordinates contain NaN or Inf"
            )

        self.update_calls += 1
        self.last_update_applied = False
        windows = tuple(
            mobility_window_weight(
                progress,
                start_fraction=motif.start_fraction,
                end_fraction=motif.end_fraction,
            )
            for motif in self.motifs
        )
        window = max(windows, default=0.0)
        extra: dict[str, Any] = {
            "proposal_source": "scaffold_boundary",
            "proposal_only": not apply_update,
            "accepted": False,
            "applied": False,
            "atomic_joint_acceptance": True,
            "orbit_proposals": [],
            "objective_weights": {
                "junction": config.junction_weight,
                "clash": config.clash_weight,
                "tilt": config.tilt_weight,
                "prior": config.prior_weight,
            },
        }
        if window > 0.0:
            self.active_window_calls += 1
            baseline_target = self.materialize_target()[0]
            current_rotations = tuple(
                motif.state.rotation[0].clone() for motif in self.motifs
            )
            current_translations = tuple(
                motif.state.translation[0].clone() for motif in self.motifs
            )
            proposed_rotations = list(current_rotations)
            proposed_translations = list(current_translations)
            proposals = []

            for orbit_index, (
                motif,
                motif_window,
                principal_axis,
            ) in enumerate(zip(self.motifs, windows, principal_axes)):
                current_rotation = current_rotations[orbit_index]
                current_translation = current_translations[orbit_index]
                if motif_window <= 0.0:
                    proposals.append(None)
                    extra["orbit_proposals"].append(
                        {
                            "orbit_index": orbit_index,
                            "constraint_orbit_id": (
                                motif.constraint_orbit_id
                            ),
                            "component_id": motif.coupling_group_id,
                            "active": False,
                            "accepted": False,
                            "committed": False,
                        }
                    )
                    continue

                translation_basis = None
                rotation_basis = None
                maximum_step_rotation = (
                    motif.per_step_rotation_degrees * motif_window
                )
                maximum_total_rotation = motif.maximum_rotation_degrees
                rotation_step_size = (
                    motif.per_step_rotation_degrees
                    * motif.response
                    * motif_window
                )
                if motif.mobility_subspace in {"radial", "radial_axial"}:
                    axis_direction = axis.direction.to(
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    axis_direction = (
                        axis_direction
                        / torch.linalg.vector_norm(axis_direction)
                    )
                    axis_point = axis.point.to(
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    master_center = (
                        motif.template_master[0].mean(dim=0)
                        + current_translation
                    )
                    offset = master_center - axis_point
                    radial = offset - torch.dot(
                        offset,
                        axis_direction,
                    ) * axis_direction
                    radial_norm = torch.linalg.vector_norm(radial)
                    if float(radial_norm.item()) <= 1e-8:
                        raise ValueError(
                            "Radial mobility is undefined for a motif "
                            "centered on the symmetry axis"
                        )
                    radial = radial / radial_norm
                    translation_basis = radial[None, :]
                    if motif.mobility_subspace == "radial_axial":
                        translation_basis = torch.stack(
                            (radial, axis_direction),
                            dim=0,
                        )
                    rotation_basis = torch.empty(
                        (0, 3),
                        dtype=current_rotation.dtype,
                        device=current_rotation.device,
                    )
                    maximum_step_rotation = 0.0
                    maximum_total_rotation = 0.0
                    rotation_step_size = 0.0

                def energy_function(rotation, translation):
                    master = self._master_coordinates_for_pose(
                        motif,
                        rotation,
                        translation,
                    )
                    candidate_target = insert_master_orbit(
                        baseline_target,
                        master,
                        motif.group_atom_indices,
                        motif.group_transform_ids,
                        self.sym_transforms,
                    )
                    return scaffold_orbit_energy(
                        candidate_target,
                        scaffold[0],
                        topology,
                        axis,
                        principal_axis=principal_axis,
                        pose_rotation=rotation,
                        pose_translation=translation,
                        config=config,
                    )

                proposal = propose_bounded_se3_step(
                    current_rotation,
                    current_translation,
                    energy_function,
                    maximum_step_translation=(
                        motif.per_step_translation * motif_window
                    ),
                    maximum_step_rotation_degrees=maximum_step_rotation,
                    maximum_total_translation=motif.maximum_translation,
                    maximum_total_rotation_degrees=maximum_total_rotation,
                    translation_step_size=(
                        motif.per_step_translation
                        * motif.response
                        * motif_window
                    ),
                    rotation_step_size_degrees=rotation_step_size,
                    translation_basis=translation_basis,
                    rotation_basis=rotation_basis,
                )
                proposals.append(proposal)
                if proposal.accepted:
                    proposed_rotations[orbit_index] = proposal.rotation
                    proposed_translations[orbit_index] = proposal.translation
                with torch.no_grad():
                    local_initial = energy_function(
                        current_rotation,
                        current_translation,
                    ).detached_dict()
                    local_proposed = energy_function(
                        proposal.rotation,
                        proposal.translation,
                    ).detached_dict()
                tracked_terms = (
                    "total",
                    "junction",
                    "clash",
                    "tilt",
                    "prior",
                )
                extra["orbit_proposals"].append(
                    {
                        "orbit_index": orbit_index,
                        "constraint_orbit_id": motif.constraint_orbit_id,
                        "component_id": motif.coupling_group_id,
                        "objective_ids": list(motif.objective_ids),
                        "mobility_subspace": motif.mobility_subspace,
                        "active": True,
                        "accepted": proposal.accepted,
                        "committed": False,
                        "line_search_scale": proposal.line_search_scale,
                        "objective": {
                            "initial": local_initial,
                            "proposed": local_proposed,
                            "delta": {
                                term: local_proposed[term]
                                - local_initial[term]
                                for term in tracked_terms
                            },
                        },
                        "proposed_translation": [
                            float(value)
                            for value in proposal.translation.detach().cpu().tolist()
                        ],
                        "proposed_delta_translation": [
                            float(value)
                            for value in (
                                proposal.delta_translation.detach().cpu().tolist()
                            )
                        ],
                        "proposed_delta_rotation_degrees": math.degrees(
                            float(
                                _axis_angle(proposal.delta_rotation)[1]
                                .detach()
                                .cpu()
                                .item()
                            )
                        ),
                    }
                )

            candidate_target = baseline_target.clone()
            for motif, rotation, translation in zip(
                self.motifs,
                proposed_rotations,
                proposed_translations,
            ):
                master = self._master_coordinates_for_pose(
                    motif,
                    rotation,
                    translation,
                )
                candidate_target = insert_master_orbit(
                    candidate_target,
                    master,
                    motif.group_atom_indices,
                    motif.group_transform_ids,
                    self.sym_transforms,
                )

            with torch.no_grad():
                initial_total, initial_terms = self._joint_scaffold_energy(
                    baseline_target,
                    scaffold[0],
                    topology=topology,
                    axis=axis,
                    principal_axes=principal_axes,
                    rotations=current_rotations,
                    translations=current_translations,
                    config=config,
                )
                proposed_total, proposed_terms = self._joint_scaffold_energy(
                    candidate_target,
                    scaffold[0],
                    topology=topology,
                    axis=axis,
                    principal_axes=principal_axes,
                    rotations=tuple(proposed_rotations),
                    translations=tuple(proposed_translations),
                    config=config,
                )
            any_candidate = any(
                proposal is not None and proposal.accepted
                for proposal in proposals
            )
            joint_accepted = bool(
                any_candidate
                and float(proposed_total.item())
                < float(initial_total.item()) - 1e-12
            )
            extra.update(
                {
                    "accepted": joint_accepted,
                    "joint_decision": (
                        "accepted"
                        if joint_accepted
                        else "rejected"
                    ),
                    "initial_energy": initial_terms,
                    "proposed_energy": proposed_terms,
                    "joint_energy_delta": (
                        float(proposed_total.detach().cpu().item())
                        - float(initial_total.detach().cpu().item())
                    ),
                }
            )
            for record, proposal in zip(
                extra["orbit_proposals"],
                proposals,
            ):
                record["committed"] = bool(
                    apply_update
                    and joint_accepted
                    and proposal is not None
                    and proposal.accepted
                )
            if apply_update and joint_accepted:
                for motif, rotation, translation in zip(
                    self.motifs,
                    proposed_rotations,
                    proposed_translations,
                ):
                    motif.state = OrbitRigidPoseState(
                        rotation=rotation[None, ...],
                        translation=translation[None, ...],
                        last_proposal_rmsd=torch.zeros_like(
                            motif.state.last_proposal_rmsd
                        ),
                    )
                self.last_update_applied = True
                extra["applied"] = True

        target = self.materialize_target()
        self._diagnostic_trajectory.append(
            self._diagnostic_snapshot(
                progress=progress,
                window_weight=window,
                extra=extra,
            )
        )
        return target

    @staticmethod
    def _pose_diagnostics(motif: OrbitRigidMotif) -> dict[str, Any]:
        rotation_degrees = []
        for rotation in motif.state.rotation:
            _, angle = _axis_angle(rotation)
            rotation_degrees.append(math.degrees(float(angle.item())))
        translation_norms = torch.linalg.vector_norm(
            motif.state.translation,
            dim=-1,
        )
        return {
            "constraint_orbit_id": motif.constraint_orbit_id,
            "component_id": motif.coupling_group_id,
            "group_action_count": int(motif.group_transform_ids.numel()),
            "group_transform_ids": [
                int(value)
                for value in motif.group_transform_ids.detach().cpu().tolist()
            ],
            "translation_norms": [
                float(value)
                for value in translation_norms.detach().cpu().tolist()
            ],
            "rotation_degrees": rotation_degrees,
            "proposal_rmsd": [
                float(value)
                for value in (
                    motif.state.last_proposal_rmsd.detach().cpu().tolist()
                )
            ],
            "maximum_translation": motif.maximum_translation,
            "maximum_rotation_degrees": motif.maximum_rotation_degrees,
            "mobility_subspace": motif.mobility_subspace,
            "proposal_source": motif.proposal_source,
            "objective_ids": list(motif.objective_ids),
            "schedule": {
                "start_fraction": motif.start_fraction,
                "end_fraction": motif.end_fraction,
                "response": motif.response,
                "max_step_translation": motif.per_step_translation,
                "max_step_rotation_degrees": (
                    motif.per_step_rotation_degrees
                ),
            },
        }

    def _diagnostic_snapshot(
        self,
        *,
        progress: float,
        window_weight: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = {
            "progress": float(progress),
            "window_weight": float(window_weight),
            "orbits": [
                {
                    "orbit_index": orbit_index,
                    **self._pose_diagnostics(motif),
                }
                for orbit_index, motif in enumerate(self.motifs)
            ],
        }
        if extra:
            snapshot.update(extra)
        return snapshot

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-serializable pose and proposal diagnostics."""

        return {
            "update_calls": self.update_calls,
            "active_window_calls": self.active_window_calls,
            "orbits": [
                {
                    "orbit_index": orbit_index,
                    **self._pose_diagnostics(motif),
                }
                for orbit_index, motif in enumerate(self.motifs)
            ],
            "trajectory": list(self._diagnostic_trajectory),
        }
