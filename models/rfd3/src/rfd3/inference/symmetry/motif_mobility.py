"""Bounded rigid motion for complete motif symmetry orbits.

The controller is deliberately separate from atomwise diffusion.  One master
master pose is estimated from all symmetry-related observations, clamped,
and then expanded through the runtime group action.  Individual fragments or
copies never receive independent rigid motions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
from rfd3.inference.symmetry.constraint_orbit import (
    ConstraintOrbitLayout,
)
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceGuidanceConfig,
    GraphInterfacePatchState,
    GraphInterfaceTopology,
    apply_graph_interface_guidance,
    graph_interface_energy,
    graph_interface_proposal_acceptable,
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
            "mobility fractions must satisfy " "0 <= start_fraction < end_fraction <= 1"
        )
    if progress <= start_fraction or progress >= end_fraction:
        return 0.0
    unit = (progress - start_fraction) / (end_fraction - start_fraction)
    return math.sin(math.pi * unit) ** 2


def _invert_frame(points, rotation, translation):
    return torch.matmul(points - translation, rotation)


def _proper_rotation_from_cross_covariance(covariance):
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = (
        torch.eye(
            3,
            dtype=covariance.dtype,
            device=covariance.device,
        )
        .expand(covariance.shape[0], 3, 3)
        .clone()
    )
    correction[:, -1, -1] = torch.sign(torch.linalg.det(torch.matmul(u, vh)))
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
    axis[other[0]] = (rotation[largest, other[0]] + rotation[other[0], largest]) / (
        4.0 * axis[largest]
    )
    axis[other[1]] = (rotation[largest, other[1]] + rotation[other[1], largest]) / (
        4.0 * axis[largest]
    )
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
        self.last_joint_transaction_applied = False
        self.last_joint_packing_diagnostics: dict[str, Any] | None = None
        self._diagnostic_trajectory: list[dict[str, Any]] = []
        self._effective_pose_prior_scales: dict[str, tuple[float, float]] = {}

    def _pose_guidance_config(
        self,
        config: ScaffoldGuidanceConfig,
        motif: OrbitRigidMotif,
    ) -> ScaffoldGuidanceConfig:
        """Scale a soft pose prior to the orbit's declared search range.

        The hard translation/rotation caps remain authoritative.  The prior
        only discourages gratuitous motion inside those caps, so a component
        allowed to search 15 A / 45 degrees must not receive the same narrow
        1 A / 5 degree basin as a nearly locked component.
        """

        translation_scale = max(
            float(config.translation_prior_scale),
            float(motif.maximum_translation) / 3.0,
        )
        rotation_scale = max(
            float(config.rotation_prior_scale_degrees),
            float(motif.maximum_rotation_degrees) / 3.0,
        )
        self._effective_pose_prior_scales[motif.constraint_orbit_id] = (
            translation_scale,
            rotation_scale,
        )
        return replace(
            config,
            translation_prior_scale=translation_scale,
            rotation_prior_scale_degrees=rotation_scale,
        )

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
                "tilt_only",
                "radial_rotation",
                "radial_axial_rotation",
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
                    "A mobile constraint orbit has no executable proposal " "source"
                )
            if (
                orbit.mobility_subspace
                in {
                    "radial",
                    "radial_axial",
                    "tilt_only",
                    "radial_rotation",
                    "radial_axial_rotation",
                }
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
            master_indices = layout.groups[orbit.master_group_index].atom_indices
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
                    maximum_rotation_degrees=(orbit.maximum_rotation_degrees),
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
            raise ValueError("Scaffold-derived motif guidance supports one pose batch")
        center = motif.template_master[0].mean(dim=0)
        centered = motif.template_master[0] - center[None, :]
        return centered @ rotation.T + center[None, :] + translation[None, :]

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

    def _mobile_pose_snapshot(
        self,
    ) -> tuple[OrbitRigidPoseState, ...]:
        """Clone every mobile pose for an atomic multi-controller rollback."""

        return tuple(
            OrbitRigidPoseState(
                rotation=motif.state.rotation.clone(),
                translation=motif.state.translation.clone(),
                last_proposal_rmsd=motif.state.last_proposal_rmsd.clone(),
            )
            for motif in self.motifs
        )

    def _restore_mobile_pose_snapshot(
        self,
        snapshot: tuple[OrbitRigidPoseState, ...],
    ) -> None:
        if len(snapshot) != len(self.motifs):
            raise ValueError("Mobility rollback snapshot has the wrong size")
        for motif, state in zip(self.motifs, snapshot, strict=True):
            motif.state = OrbitRigidPoseState(
                rotation=state.rotation.clone(),
                translation=state.translation.clone(),
                last_proposal_rmsd=state.last_proposal_rmsd.clone(),
            )

    def _insert_mobile_target(
        self,
        coordinates: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Restore only mobile-orbit atoms into one scaffold snapshot."""

        if coordinates.shape != target.shape:
            raise ValueError(
                "Joint packing coordinates and target must have equal shape"
            )
        updated = coordinates.clone()
        for motif in self.motifs:
            indices = torch.unique(motif.group_atom_indices.flatten())
            updated[:, indices, :] = target[:, indices, :]
        return updated

    def _inverse_average_proposal(self, motif, raw_coordinates):
        canonical_copies = []
        for group_row, transform_id_tensor in enumerate(motif.group_transform_ids):
            transform_id = int(transform_id_tensor.item())
            rotation, translation = self.sym_transforms[transform_id]
            observed = raw_coordinates[
                :,
                motif.group_atom_indices[group_row],
                :,
            ]
            canonical_copies.append(_invert_frame(observed, rotation, translation))
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
                "Mobility proposal coordinates must match the fixed target " "shape"
            )
        if not torch.isfinite(raw_coordinates).all():
            raise ValueError("Mobility proposal coordinates contain NaN or Inf")
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
                    raise ValueError("Rigid motif pose fitting produced NaN or Inf")
                updated_rotations = []
                updated_translations = []
                for batch_index in range(raw_coordinates.shape[0]):
                    current_rotation = motif.state.rotation[batch_index]
                    relative_rotation = (
                        desired_rotation[batch_index] @ current_rotation.T
                    )
                    increment = _scaled_rotation(
                        relative_rotation,
                        math.radians(motif.per_step_rotation_degrees * motif_window),
                        motif.response * motif_window,
                    )
                    rotation = increment @ current_rotation
                    rotation = _scaled_rotation(
                        rotation,
                        math.radians(motif.maximum_rotation_degrees),
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
                        motif.state.translation[batch_index] + delta_translation
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
                "Joint scaffold energy inputs must match the mobile orbit " "count"
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
        for motif, principal_axis, rotation, translation in zip(
            self.motifs,
            principal_axes,
            rotations,
            translations,
        ):
            pose_config = replace(
                self._pose_guidance_config(config, motif),
                junction_weight=0.0,
                clash_weight=0.0,
            )
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
                        (config.tilt_weight * pose.tilt).detach().cpu().item()
                    ),
                    "prior": float(pose.prior.detach().cpu().item()),
                    "weighted_prior": float(
                        (config.prior_weight * pose.prior).detach().cpu().item()
                    ),
                    "tilt_degrees": float(pose.tilt_degrees.detach().cpu().item()),
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
                (config.junction_weight * geometry.junction).detach().cpu().item()
            ),
            "clash": float(combined_clash.detach().cpu().item()),
            "scaffold_clash": float(geometry.clash.detach().cpu().item()),
            "inter_orbit_clash": float(inter_orbit_clash.detach().cpu().item()),
            "weighted_clash": float(
                (config.clash_weight * combined_clash).detach().cpu().item()
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
            for value in (topology.fixed_ca_atom_indices.detach().cpu().tolist())
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
                    torch.mean(torch.square(torch.relu(clash_distance - distances)))
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
        pose_energy: Callable[[torch.Tensor], torch.Tensor] | None = None,
        proposal_response_scale: float = 1.0,
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
            raise ValueError("Scaffold-derived motif guidance supports one pose batch")
        scaffold = scaffold_coordinates.to(
            dtype=self.base_target.dtype,
            device=self.base_target.device,
        )
        if scaffold.shape != self.base_target.shape:
            raise ValueError(
                "Scaffold guidance coordinates must match the fixed target " "shape"
            )
        if not torch.isfinite(scaffold).all():
            raise ValueError("Scaffold guidance coordinates contain NaN or Inf")
        if not math.isfinite(proposal_response_scale) or proposal_response_scale <= 0:
            raise ValueError("proposal_response_scale must be finite and positive")

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
            "proposal_response_scale": float(proposal_response_scale),
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
                            "constraint_orbit_id": (motif.constraint_orbit_id),
                            "component_id": motif.coupling_group_id,
                            "active": False,
                            "accepted": False,
                            "committed": False,
                        }
                    )
                    continue

                pose_guidance_config = self._pose_guidance_config(
                    config,
                    motif,
                )

                translation_basis = None
                rotation_basis = None
                maximum_step_rotation = motif.per_step_rotation_degrees * motif_window
                maximum_total_rotation = motif.maximum_rotation_degrees
                effective_response = min(
                    1.0,
                    motif.response * proposal_response_scale,
                )
                rotation_step_size = (
                    motif.per_step_rotation_degrees
                    * effective_response
                    * motif_window
                )
                if motif.mobility_subspace == "tilt_only":
                    axis_direction = axis.direction.to(
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    axis_direction = axis_direction / torch.linalg.vector_norm(
                        axis_direction
                    )
                    # Pick the Cartesian direction least parallel to the
                    # symmetry axis, then build an orthonormal basis for the
                    # plane perpendicular to it.  Projecting the infinitesimal
                    # rotation vector into this plane permits tilt while
                    # forbidding axial twist.  Translation is explicitly
                    # projected into the empty subspace.
                    reference_index = int(
                        torch.argmin(torch.abs(axis_direction)).item()
                    )
                    reference = torch.zeros_like(axis_direction)
                    reference[reference_index] = 1.0
                    tilt_axis_1 = torch.linalg.cross(
                        axis_direction,
                        reference,
                    )
                    tilt_axis_1 = tilt_axis_1 / torch.linalg.vector_norm(
                        tilt_axis_1
                    )
                    tilt_axis_2 = torch.linalg.cross(
                        axis_direction,
                        tilt_axis_1,
                    )
                    tilt_axis_2 = tilt_axis_2 / torch.linalg.vector_norm(
                        tilt_axis_2
                    )
                    translation_basis = torch.empty(
                        (0, 3),
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    rotation_basis = torch.stack(
                        (tilt_axis_1, tilt_axis_2),
                        dim=0,
                    )
                    maximum_step_translation = 0.0
                    maximum_total_translation = 0.0
                    translation_step_size = 0.0
                if motif.mobility_subspace in {
                    "radial",
                    "radial_axial",
                    "radial_rotation",
                    "radial_axial_rotation",
                }:
                    axis_direction = axis.direction.to(
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    axis_direction = axis_direction / torch.linalg.vector_norm(
                        axis_direction
                    )
                    axis_point = axis.point.to(
                        dtype=current_translation.dtype,
                        device=current_translation.device,
                    )
                    master_center = (
                        motif.template_master[0].mean(dim=0) + current_translation
                    )
                    offset = master_center - axis_point
                    radial = (
                        offset
                        - torch.dot(
                            offset,
                            axis_direction,
                        )
                        * axis_direction
                    )
                    radial_norm = torch.linalg.vector_norm(radial)
                    if float(radial_norm.item()) <= 1e-8:
                        raise ValueError(
                            "Radial mobility is undefined for a motif "
                            "centered on the symmetry axis"
                        )
                    radial = radial / radial_norm
                    translation_basis = radial[None, :]
                    if motif.mobility_subspace in {
                        "radial_axial",
                        "radial_axial_rotation",
                    }:
                        translation_basis = torch.stack(
                            (radial, axis_direction),
                            dim=0,
                        )
                    if motif.mobility_subspace in {
                        "radial",
                        "radial_axial",
                    }:
                        rotation_basis = torch.empty(
                            (0, 3),
                            dtype=current_rotation.dtype,
                            device=current_rotation.device,
                        )
                        maximum_step_rotation = 0.0
                        maximum_total_rotation = 0.0
                        rotation_step_size = 0.0

                def energy_function(
                    rotation,
                    translation,
                    *,
                    active_motif=motif,
                    active_principal_axis=principal_axis,
                    active_config=pose_guidance_config,
                ):
                    master = self._master_coordinates_for_pose(
                        active_motif,
                        rotation,
                        translation,
                    )
                    candidate_target = insert_master_orbit(
                        baseline_target,
                        master,
                        active_motif.group_atom_indices,
                        active_motif.group_transform_ids,
                        self.sym_transforms,
                    )
                    energy = scaffold_orbit_energy(
                        candidate_target,
                        scaffold[0],
                        topology,
                        axis,
                        principal_axis=active_principal_axis,
                        pose_rotation=rotation,
                        pose_translation=translation,
                        config=active_config,
                    )
                    if pose_energy is not None:
                        packing = pose_energy(candidate_target)
                        if packing.ndim != 0 or not torch.isfinite(packing):
                            raise ValueError(
                                "Additional motif pose energy must be one "
                                "finite scalar"
                            )
                        # Keep the established diagnostics schema while making
                        # the actual SE(3) gradient packing-aware.  The outer
                        # atomic transaction records the packing contribution
                        # separately and is still the authoritative acceptor.
                        energy = replace(
                            energy,
                            total=energy.total + packing,
                        )
                    return energy

                proposal = propose_bounded_se3_step(
                    current_rotation,
                    current_translation,
                    energy_function,
                    maximum_step_translation=(
                        maximum_step_translation
                        if motif.mobility_subspace == "tilt_only"
                        else motif.per_step_translation * motif_window
                    ),
                    maximum_step_rotation_degrees=maximum_step_rotation,
                    maximum_total_translation=(
                        maximum_total_translation
                        if motif.mobility_subspace == "tilt_only"
                        else motif.maximum_translation
                    ),
                    maximum_total_rotation_degrees=maximum_total_rotation,
                    translation_step_size=(
                        translation_step_size
                        if motif.mobility_subspace == "tilt_only"
                        else motif.per_step_translation
                        * effective_response
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
                        "effective_response": effective_response,
                        "active": True,
                        "accepted": proposal.accepted,
                        "committed": False,
                        "line_search_scale": proposal.line_search_scale,
                        "objective": {
                            "initial": local_initial,
                            "proposed": local_proposed,
                            "delta": {
                                term: local_proposed[term] - local_initial[term]
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
                initial_pose_energy = torch.zeros_like(initial_total)
                proposed_pose_energy = torch.zeros_like(proposed_total)
                if pose_energy is not None:
                    initial_pose_energy = pose_energy(baseline_target)
                    proposed_pose_energy = pose_energy(candidate_target)
                    for name, value in (
                        ("initial", initial_pose_energy),
                        ("proposed", proposed_pose_energy),
                    ):
                        if value.ndim != 0 or not torch.isfinite(value):
                            raise ValueError(
                                "Additional joint motif pose energy must "
                                f"be one finite scalar ({name})"
                            )
                    # The local SE(3) gradient above already contains this
                    # packing term.  The atomic multi-orbit acceptance must
                    # compare that same objective; otherwise a
                    # packing-improving pose is silently rejected whenever
                    # the scaffold-only term rises by any amount.
                    initial_total = initial_total + initial_pose_energy
                    proposed_total = proposed_total + proposed_pose_energy
            any_candidate = any(
                proposal is not None and proposal.accepted for proposal in proposals
            )
            joint_accepted = bool(
                any_candidate
                and float(proposed_total.item()) < float(initial_total.item()) - 1e-12
            )
            extra.update(
                {
                    "accepted": joint_accepted,
                    "joint_decision": ("accepted" if joint_accepted else "rejected"),
                    "initial_energy": initial_terms,
                    "proposed_energy": proposed_terms,
                    "additional_pose_energy": {
                        "initial": float(initial_pose_energy.detach().cpu().item()),
                        "proposed": float(proposed_pose_energy.detach().cpu().item()),
                        "delta": float(
                            (proposed_pose_energy - initial_pose_energy)
                            .detach()
                            .cpu()
                            .item()
                        ),
                    },
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

    def update_orbits_with_interface_packing(
        self,
        scaffold_coordinates: torch.Tensor,
        features: dict[str, Any],
        *,
        progress: float,
        topology: BoundaryTopology,
        axis: CyclicAxis,
        principal_axes: tuple[torch.Tensor, ...],
        scaffold_config: ScaffoldGuidanceConfig,
        interface_topology: GraphInterfaceTopology,
        interface_config: GraphInterfaceGuidanceConfig,
        patch_state: GraphInterfacePatchState,
        projector: Callable[[torch.Tensor], torch.Tensor],
        apply_update: bool,
        capture_response_scale: float = 1.0,
        expand_response_scale: float = 1.0,
        polish_response_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Propose motif poses and generated packing as one transaction.

        The historical runtime updated the motif target before Euler and the
        generated interface after Euler.  Each controller could therefore
        accept a locally improving move that made the other controller's
        geometry worse.  This method snapshots both mutable states, proposes
        a symmetry-projected generated patch and bounded motif poses from the
        same scaffold, then accepts or rolls back the complete transaction.
        """

        if scaffold_coordinates.shape != self.base_target.shape:
            raise ValueError("Joint packing scaffold must match the motif target shape")
        if self.base_target.shape[0] != 1:
            raise ValueError("Joint packing mobility supports one pose batch")
        phase_response_scales = {
            "capture": float(capture_response_scale),
            "expand": float(expand_response_scale),
            "polish": float(polish_response_scale),
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in phase_response_scales.values()
        ):
            raise ValueError(
                "Joint packing phase response scales must be finite and positive"
            )
        self.last_joint_transaction_applied = False
        pose_snapshot = self._mobile_pose_snapshot()
        patch_snapshot = dict(patch_state.assignments)
        patch_locked_snapshot = bool(patch_state.locked)
        patch_lock_reason_snapshot = patch_state.lock_reason

        def rollback_mutable_state() -> None:
            self._restore_mobile_pose_snapshot(pose_snapshot)
            patch_state.assignments.clear()
            patch_state.assignments.update(patch_snapshot)
            patch_state.locked = patch_locked_snapshot
            patch_state.lock_reason = patch_lock_reason_snapshot
            self.last_update_applied = False

        baseline_target = self.materialize_target()
        baseline_coordinates = self._insert_mobile_target(
            scaffold_coordinates,
            baseline_target,
        )
        baseline_graph = graph_interface_energy(
            baseline_coordinates,
            interface_topology,
            interface_config,
            patch_assignments=patch_state.assignments,
        )
        baseline_rotations = tuple(
            motif.state.rotation[0].clone() for motif in self.motifs
        )
        baseline_translations = tuple(
            motif.state.translation[0].clone() for motif in self.motifs
        )
        baseline_scaffold_total, baseline_scaffold_terms = self._joint_scaffold_energy(
            baseline_target[0],
            baseline_coordinates[0],
            topology=topology,
            axis=axis,
            principal_axes=principal_axes,
            rotations=baseline_rotations,
            translations=baseline_translations,
            config=scaffold_config,
        )

        try:
            packed_coordinates, packing_step = apply_graph_interface_guidance(
                baseline_coordinates,
                features,
                interface_topology,
                progress=progress,
                config=interface_config,
                projector=projector,
                patch_state=patch_state,
            )
        except Exception:
            rollback_mutable_state()
            raise
        adaptive_phase = str(packing_step.get("adaptive_phase", "polish"))
        if adaptive_phase not in phase_response_scales:
            rollback_mutable_state()
            raise ValueError(
                f"Unknown joint packing adaptive phase {adaptive_phase!r}"
            )
        proposal_response_scale = phase_response_scales[adaptive_phase]

        def packing_aware_pose_energy(
            candidate_target: torch.Tensor,
        ) -> torch.Tensor:
            candidate_coordinates = self._insert_mobile_target(
                packed_coordinates,
                candidate_target[None, ...],
            )
            return graph_interface_energy(
                candidate_coordinates,
                interface_topology,
                interface_config,
                patch_assignments=patch_state.assignments,
            ).total

        try:
            self.update_orbits_from_scaffold(
                packed_coordinates,
                progress=progress,
                topology=topology,
                axis=axis,
                principal_axes=principal_axes,
                config=scaffold_config,
                # Commit provisionally.  The joint decision below is
                # authoritative and restores this snapshot on any failure or
                # proposal-only run.
                apply_update=True,
                pose_energy=packing_aware_pose_energy,
                proposal_response_scale=proposal_response_scale,
            )
        except Exception:
            rollback_mutable_state()
            raise
        motif_pose_changed = bool(self.last_update_applied)
        try:
            candidate_target = self.materialize_target()
            candidate_coordinates = self._insert_mobile_target(
                packed_coordinates,
                candidate_target,
            )
            candidate_graph = graph_interface_energy(
                candidate_coordinates,
                interface_topology,
                interface_config,
                patch_assignments=patch_state.assignments,
            )
            candidate_rotations = tuple(
                motif.state.rotation[0] for motif in self.motifs
            )
            candidate_translations = tuple(
                motif.state.translation[0] for motif in self.motifs
            )
            candidate_scaffold_total, candidate_scaffold_terms = (
                self._joint_scaffold_energy(
                    candidate_target[0],
                    candidate_coordinates[0],
                    topology=topology,
                    axis=axis,
                    principal_axes=principal_axes,
                    rotations=candidate_rotations,
                    translations=candidate_translations,
                    config=scaffold_config,
                )
            )
        except Exception:
            rollback_mutable_state()
            raise

        baseline_total = baseline_graph.total + baseline_scaffold_total
        candidate_total = candidate_graph.total + candidate_scaffold_total
        packing_improved = bool(
            torch.isfinite(candidate_graph.total)
            and float(candidate_graph.total.detach().cpu().item())
            < float(baseline_graph.total.detach().cpu().item()) - 1.0e-10
        )
        packing_contract_safe = graph_interface_proposal_acceptable(
            baseline_graph,
            candidate_graph,
            interface_config,
        )

        def minimum_not_worse(
            initial: torch.Tensor,
            candidate: torch.Tensor,
        ) -> bool:
            initial_value = float(initial.min().detach().cpu().item())
            candidate_value = float(candidate.min().detach().cpu().item())
            required = (
                interface_config.clash_ca_distance
                if initial_value >= interface_config.clash_ca_distance
                else initial_value
            )
            return candidate_value >= required - 1.0e-6

        edge_safe = minimum_not_worse(
            baseline_graph.minimum_distances,
            candidate_graph.minimum_distances,
        )
        global_safe = minimum_not_worse(
            baseline_graph.minimum_global_safety_distance.reshape(1),
            candidate_graph.minimum_global_safety_distance.reshape(1),
        )
        baseline_junction = float(baseline_graph.junction.detach().cpu().item())
        candidate_junction = float(candidate_graph.junction.detach().cpu().item())
        junction_limit = (
            interface_config.maximum_backbone_loss
            if baseline_junction <= interface_config.maximum_backbone_loss
            else baseline_junction
        )
        junction_safe = candidate_junction <= junction_limit + 1.0e-8
        combined_improved = bool(
            torch.isfinite(candidate_total)
            and float(candidate_total.detach().cpu().item())
            < float(baseline_total.detach().cpu().item()) - 1.0e-10
        )
        transaction_has_change = bool(
            packing_step.get("proposal_accepted", False) or motif_pose_changed
        )
        accepted = bool(
            transaction_has_change
            and packing_improved
            and combined_improved
            and packing_contract_safe
            and edge_safe
            and global_safe
            and junction_safe
        )
        committed = bool(accepted and apply_update)

        diagnostics = {
            "joint_packing_transaction": True,
            "accepted": accepted,
            "committed": committed,
            "proposal_only": not apply_update,
            "motif_pose_changed": motif_pose_changed,
            "adaptive_phase": adaptive_phase,
            "motif_pose_response_scale": proposal_response_scale,
            "generated_patch_changed": bool(
                packing_step.get("proposal_accepted", False)
            ),
            "packing_improved": packing_improved,
            "packing_contract_safe": packing_contract_safe,
            "combined_improved": combined_improved,
            "edge_safe": edge_safe,
            "global_safe": global_safe,
            "junction_safe": junction_safe,
            "baseline_total": float(baseline_total.detach().cpu().item()),
            "candidate_total": float(candidate_total.detach().cpu().item()),
            "baseline_packing": float(baseline_graph.total.detach().cpu().item()),
            "candidate_packing": float(candidate_graph.total.detach().cpu().item()),
            "baseline_scaffold": baseline_scaffold_terms,
            "candidate_scaffold": candidate_scaffold_terms,
            "packing_step": packing_step,
        }
        self.last_joint_packing_diagnostics = diagnostics

        if not committed:
            rollback_mutable_state()
            target = baseline_target
            coordinates = baseline_coordinates
        else:
            self.last_joint_transaction_applied = True
            target = candidate_target
            coordinates = candidate_coordinates

        if self._diagnostic_trajectory:
            self._diagnostic_trajectory[-1].update(
                {
                    "joint_packing_transaction": True,
                    "joint_packing_accepted": accepted,
                    "joint_packing_committed": committed,
                    "joint_packing_energy_delta": diagnostics["candidate_total"]
                    - diagnostics["baseline_total"],
                    "packing_energy_delta": diagnostics["candidate_packing"]
                    - diagnostics["baseline_packing"],
                }
            )
        return target, coordinates, diagnostics

    def _pose_diagnostics(self, motif: OrbitRigidMotif) -> dict[str, Any]:
        rotation_degrees = []
        for rotation in motif.state.rotation:
            _, angle = _axis_angle(rotation)
            rotation_degrees.append(math.degrees(float(angle.item())))
        translation_norms = torch.linalg.vector_norm(
            motif.state.translation,
            dim=-1,
        )
        template_master_centers = motif.template_master.mean(dim=1)
        prior_scales = self._effective_pose_prior_scales.get(
            motif.constraint_orbit_id
        )
        diagnostics = {
            "constraint_orbit_id": motif.constraint_orbit_id,
            "component_id": motif.coupling_group_id,
            "group_action_count": int(motif.group_transform_ids.numel()),
            "group_transform_ids": [
                int(value)
                for value in motif.group_transform_ids.detach().cpu().tolist()
            ],
            "translation_norms": [
                float(value) for value in translation_norms.detach().cpu().tolist()
            ],
            "translation_vectors": [
                [float(component) for component in vector]
                for vector in motif.state.translation.detach().cpu().tolist()
            ],
            "template_master_centers": [
                [float(component) for component in center]
                for center in template_master_centers.detach().cpu().tolist()
            ],
            "rotation_degrees": rotation_degrees,
            "proposal_rmsd": [
                float(value)
                for value in (motif.state.last_proposal_rmsd.detach().cpu().tolist())
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
                "max_step_rotation_degrees": (motif.per_step_rotation_degrees),
            },
        }
        if prior_scales is not None:
            diagnostics["effective_pose_prior"] = {
                "translation_scale": prior_scales[0],
                "rotation_scale_degrees": prior_scales[1],
                "normalization": "at_least_one_third_of_hard_bound",
            }
        return diagnostics

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
