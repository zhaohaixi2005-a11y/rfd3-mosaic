import inspect
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from jaxtyping import Float
from rfd3.inference.symmetry.constraint_runtime import (
    ConstraintProposalResult,
    MosaicConstraintRuntime,
)
from rfd3.inference.symmetry.cylindrical_projector import (
    CylindricalCoordinateProjector,
)
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceGuidanceConfig,
    GraphInterfacePatchState,
    apply_graph_interface_guidance,
    build_graph_interface_topology,
    build_symmetric_scaffold_interface_topology,
    graph_interface_energy,
    graph_interface_energy_diagnostics,
    graph_interface_quality_satisfied,
    resolve_graph_interface_patch_assignments,
)
from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector
from rfd3.inference.symmetry.local_neighbourhood import (
    LocalSymmetryRuntimeContext,
    build_local_symmetry_neighbourhood,
    crop_features_to_local_neighbourhood,
    expand_local_prediction_to_full_orbit,
    expand_local_token_prediction_to_full_orbit,
)
from rfd3.inference.symmetry.motif_mobility import (
    OrbitRigidMotifController,
)
from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    apply_scaffold_core_guidance,
    build_scaffold_core_topology,
    project_generated_polymer_continuity,
    robust_interface_capture_energy,
    scaffold_core_energy,
)
from rfd3.inference.symmetry.scaffold_guidance import (
    ScaffoldGuidanceConfig,
    build_boundary_topology,
    extract_symmetry_primary_axis,
    principal_axis_from_points,
)
from rfd3.inference.symmetry.symmetry_utils import (
    apply_symmetry_to_xyz_atomwise,
    build_symmetry_orbit_layout,
    expand_symmetry_coupled_displacements,
    project_symmetry_orbit_average,
    symmetry_orbit_mask_mismatch_count,
    symmetry_orbit_residual,
    symmetry_orbit_tolerance,
)
from rfd3.model.cfg_utils import strip_X

from foundry.common import exists
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


_AXIS_DEPENDENT_MOTIF_SUBSPACES = frozenset(
    {
        "radial",
        "radial_axial",
        "tilt_only",
        "radial_rotation",
        "radial_axial_rotation",
    }
)


def _scaffold_guidance_requires_primary_axis(
    symmetry_id: str | None,
    mobility_subspaces: tuple[str, ...],
) -> bool:
    """Resolve whether scaffold-driven mobility has one physical main axis.

    Cn and Dn have a declared primary cyclic axis. Polyhedral groups do not:
    their bounded SE(3) orbits remain fully executable, but a radial, axial or
    tilt-only subspace would be ambiguous and therefore fails closed.
    """

    normalized_id = str(symmetry_id or "").strip().upper()
    if (
        len(normalized_id) >= 2
        and normalized_id[0] in {"C", "D"}
        and normalized_id[1:].isdigit()
        and int(normalized_id[1:]) >= 2
    ):
        return True
    if normalized_id in {"T", "O", "I"}:
        requested = sorted(set(mobility_subspaces) & _AXIS_DEPENDENT_MOTIF_SUBSPACES)
        if requested:
            raise ValueError(
                f"{normalized_id} has no single global primary axis; "
                "axis-dependent motif mobility is undefined for subspaces: "
                + ", ".join(requested)
            )
        unsupported = sorted(set(mobility_subspaces) - {"bounded_se3"})
        if unsupported:
            raise ValueError(
                f"Unsupported {normalized_id} scaffold-driven motif mobility "
                "subspaces: " + ", ".join(unsupported)
            )
        return False
    raise ValueError(
        "Scaffold-driven motif mobility requires a runtime Cn, Dn, T, O or I "
        "symmetry_id"
    )


def _motif_mobility_proposal_schedule(
    *,
    total_steps: int,
    configured_interval: int,
    target_update_count: int,
    windows: tuple[tuple[float, float], ...],
    always_propose: bool = False,
) -> dict[str, Any]:
    """Resolve a trajectory-length-aware mobility proposal schedule."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if configured_interval <= 0:
        raise ValueError("configured_interval must be positive")
    if target_update_count < 0:
        raise ValueError("target_update_count cannot be negative")
    if not windows:
        raise ValueError("at least one mobility window is required")

    effective_interval = 1 if always_propose else configured_interval
    if target_update_count and not always_propose:
        target_interval = max(1, total_steps // target_update_count)
        effective_interval = min(configured_interval, target_interval)

    proposal_steps = tuple(range(0, total_steps, effective_interval))
    active_counts = []
    for start_fraction, end_fraction in windows:
        active_counts.append(
            sum(
                start_fraction < step / max(total_steps - 1, 1) < end_fraction
                for step in proposal_steps
            )
        )
    return {
        "total_diffusion_steps": total_steps,
        "declared_update_interval": configured_interval,
        "effective_update_interval": effective_interval,
        "target_update_count": target_update_count,
        "scheduled_proposal_count": len(proposal_steps),
        "scheduled_active_proposal_counts": active_counts,
        "normalization_applied": bool(
            target_update_count
            and not always_propose
            and effective_interval < configured_interval
        ),
        "proposal_every_model_step": always_propose,
    }


@dataclass(kw_only=True)
class SampleDiffusionConfig:
    kind: Literal["default", "symmetry"] = "default"

    # Standard EDM args
    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    solver: Literal["af3"] = "af3"

    # RFD3 / design args
    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0
    skip_few_diffusion_steps: bool = False
    allow_realignment: bool = False
    insert_motif_at_end: bool = True
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: float | None = None

    # Optional Interface-Seed scaffold guidance.  This is deliberately
    # disabled by default so standard RFD3 sampling is unchanged.
    interface_seed_compactness_weight: float = 0.0
    interface_seed_compactness_end_frac: float = 0.75
    interface_seed_compactness_max_step: float = 0.5
    # Graph-declared output-stage interface objectives.  Unlike the legacy
    # compactness option, this acts only across explicitly named neighbour
    # edges and applies a joint attractive/repulsive field to generated atoms.
    enable_graph_interface_guidance: bool = False
    # Explicit interface-seeded oligomer mode.  The supplied interface stays
    # exact; generated regions on cyclic neighbours receive the same mature
    # packing field used by graph-declared designed interfaces.
    enable_symmetric_scaffold_packing: bool = False
    # RFdiffusion-style balance.  The historical defaults are preserved:
    # no monomer-core field and full generated inter-chain attraction.
    scaffold_core_intra_chain_weight: float = 0.0
    scaffold_core_inter_chain_weight: float = 1.0
    scaffold_core_inter_chain_excess_penalty: float = 0.0
    # Supplied-interface-only rigid capture.  The compiler enables this for a
    # movable joint-rigid seed; it is never inferred from symmetry alone.
    enable_supplied_interface_robust_capture: bool = False
    supplied_interface_capture_weight: float = 1.0
    # Mosaic may request an independent kinematic safety projection for
    # generated peptide paths.  It does not imply intra-chain compaction or
    # inter-chain interface creation.
    enable_generated_polymer_continuity_guidance: bool = False
    generated_polymer_continuity_target_ca_distance: float = 3.8
    generated_polymer_continuity_tolerance: float = 0.5
    generated_polymer_continuity_iterations: int = 64
    # Compiler-owned, topology-neutral protection for two-anchor generated
    # runs.  The routing field uses relative endpoint-corridor ownership; it
    # neither creates an interface nor imposes a pore/compactness target.
    enable_generated_cross_chain_topology_guidance: bool = False
    generated_routing_ownership_weight: float = 1.0
    graph_interface_guidance_weight: float = 1.0
    graph_interface_guidance_coverage_weight: float = 1.0
    graph_interface_guidance_continuity_weight: float = 1.0
    graph_interface_guidance_orientation_weight: float = 0.25
    graph_interface_guidance_shape_weight: float = 0.5
    graph_interface_guidance_backbone_weight: float = 0.1
    graph_interface_guidance_interface_balance_weight: float = 0.5
    graph_interface_guidance_clash_weight: float = 8.0
    graph_interface_guidance_distance_weight: float = 0.25
    graph_interface_guidance_contact_prior_weight: float = 0.0
    graph_interface_guidance_contact_prior_guide_scale: float = 2.0
    graph_interface_guidance_contact_prior_decay_power: float = 2.0
    graph_interface_guidance_contact_prior_r_0: float = 8.0
    graph_interface_guidance_contact_prior_d_0: float = 2.0
    graph_interface_guidance_target_ca_distance: float = 8.0
    graph_interface_guidance_clash_ca_distance: float = 3.5
    graph_interface_guidance_pairs_per_edge: int = 8
    graph_interface_guidance_start_fraction: float = 0.05
    graph_interface_guidance_end_fraction: float = 0.80
    graph_interface_guidance_terminal_weight_floor: float = 0.8
    graph_interface_guidance_maximum_token_step: float = 0.25
    graph_interface_guidance_unsatisfied_step_fraction: float = 0.50
    graph_interface_guidance_final_polish_steps: int = 12
    graph_interface_guidance_token_smoothing_weight: float = 0.5
    graph_interface_guidance_token_smoothing_passes: int = 1
    graph_interface_guidance_continuity_softness: float = 0.75
    graph_interface_guidance_maximum_tangent_normal_cosine: float = 0.65
    graph_interface_guidance_backbone_ca_distance: float = 3.8
    graph_interface_guidance_backbone_ca_tolerance: float = 0.5
    graph_interface_guidance_patch_exclusivity_weight: float = 1.0
    graph_interface_guidance_patch_rigid_weight: float = 1.0
    graph_interface_guidance_patch_blend_radius: int = 2
    graph_interface_guidance_maximum_patch_rotation_degrees: float = 2.0
    graph_interface_guidance_patch_lock_fraction: float = 0.50
    graph_interface_guidance_line_search_steps: int = 5
    graph_interface_guidance_line_search_contraction: float = 0.5
    graph_interface_guidance_capture_ca_distance: float = 12.0
    graph_interface_guidance_maximum_orientation_loss: float = 0.05
    graph_interface_guidance_maximum_shape_loss: float = 0.08
    graph_interface_guidance_maximum_backbone_loss: float = 0.02
    graph_interface_guidance_maximum_patch_exclusivity_loss: float = 0.05
    preserve_fixed_motif_during_symmetry: bool = False
    require_motif_constraint_groups: bool = False
    motif_constraint_conflict_tolerance: float = 1e-4
    fixed_motif_finalization_mode: Literal[
        "official_reinsert_then_project", "motif_precedence"
    ] = "motif_precedence"
    symmetry_state_mode: Literal["legacy_asu", "orbit_average"] = "legacy_asu"
    symmetry_noise_mode: Literal["independent", "coupled"] = "independent"
    symmetry_execution_backend: Literal["explicit_all_copy", "local_neighbourhood"] = (
        "explicit_all_copy"
    )
    symmetry_neighbour_radius: int = 1
    symmetry_include_dihedral_mate: bool = True
    symmetry_orbit_max_error: float = 1e-3
    fixed_target_symmetry_rmsd_tolerance: float = 0.01
    fixed_target_symmetry_max_tolerance: float = 0.03
    enable_orbit_rigid_motif_mobility: bool = False
    motif_mobility_start_fraction: float = 0.10
    motif_mobility_end_fraction: float = 0.85
    motif_mobility_response: float = 0.25
    motif_mobility_per_step_translation: float = 0.25
    motif_mobility_per_step_rotation_degrees: float = 1.0
    motif_mobility_proposal_source: Literal["denoiser", "scaffold_boundary"] = (
        "denoiser"
    )
    motif_mobility_apply_updates: bool = True
    motif_mobility_update_interval: int = 5
    motif_mobility_target_update_count: int = 24
    # Movable rigid bodies use proportions of their declared active window,
    # never absolute diffusion-step indices.  These scales expose 100%, 50%
    # and 20% of the capture amplitude during capture, settle and polish,
    # respectively, for every declared base response.  The hard total
    # translation/rotation bounds remain authoritative throughout.
    motif_mobility_capture_fraction: float = 0.40
    motif_mobility_settle_fraction: float = 0.40
    motif_mobility_capture_response_scale: float = 5.0
    motif_mobility_expand_response_scale: float = 2.5
    motif_mobility_polish_response_scale: float = 1.0
    motif_mobility_target_max_tilt_degrees: float = 20.0
    motif_mobility_junction_weight: float = 1.0
    motif_mobility_clash_weight: float = 1.0
    motif_mobility_tilt_weight: float = 0.25
    motif_mobility_prior_weight: float = 0.05
    motif_mobility_junction_target_distance: float = 3.8
    motif_mobility_clash_distance: float = 3.0

    # Recycling
    n_recycle: int | None = None  # Override model default n_recycle for inference


class SampleDiffusionWithMotif(SampleDiffusionConfig):
    """Diffusion sampler that supports optional motif alignment."""

    def _construct_inference_noise_schedule(
        self, device: torch.device, partial_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        if self.num_timesteps < 2:
            raise ValueError(
                "num_timesteps must be at least two so the diffusion "
                "sampler performs an update"
            )
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        if partial_t is not None:
            # For now, partial t is a global parameter
            partial_t_value = float(partial_t.mean())
            noise_schedule = t_hat
            ranked_logger.info(
                "Using partial diffusion with t={}".format(partial_t_value)
            )

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t_value]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t_value}!"
                )
                raise ValueError(
                    "partial_t leaves no usable diffusion timesteps; "
                    f"schedule range is [{original_min:.6f}, "
                    f"{original_max:.6f}]"
                )
        else:
            noise_schedule = t_hat

        if len(noise_schedule) < 2:
            raise ValueError(
                "The effective diffusion schedule must contain at least "
                "two timesteps after partial_t filtering"
            )
        return noise_schedule

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )  # (D, L, 3)

        if self.s_jitter_origin > 0.0:
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    # If centering_affects_motif is True, the model's predictions from (step_num-1) might affect the motif
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    # If keeping the motif position wrt the origin fixed, we can't do translational augmentation
                    # We want to keep this position fixed in the interval where the model is not allowed to change it
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            # Handle chunked mode vs standard mode
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            if self.use_classifier_free_guidance and (
                self.cfg_t_max is None or c_t > self.cfg_t_max
            ):
                # CFG mode requires the reference (unconditional) features and
                # initializer outputs; RFD3.forward provides both only when CFG is on.
                assert ref_initializer_outputs is not None
                assert f_ref is not None
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                # unconditional forward pass
                outs_ref = diffusion_module(
                    X_noisy_L=X_noisy_L_stripped,  # modify X
                    t=t_hat.tile(D),
                    f=f_ref,  # modified f
                    n_recycle=self.n_recycle,
                    **ref_initializer_outputs,
                )

                X_denoised_L_stripped = outs_ref["X_L"]

                delta_L_ref = (
                    X_noisy_L_stripped - X_denoised_L_stripped
                ) / t_hat  # gradient of x wrt. t at x_t_hat

                # pad delta_L_ref with zeros to match delta_L (for the unindexed atoms)
                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                # apply CFG
                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0", 0) > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)
        self._exact_symmetry_orbit_layout = None
        valid_state_modes = {"legacy_asu", "orbit_average"}
        if self.symmetry_state_mode not in valid_state_modes:
            raise ValueError(
                "symmetry_state_mode must be one of " f"{sorted(valid_state_modes)}"
            )
        valid_finalization_modes = {
            "official_reinsert_then_project",
            "motif_precedence",
        }
        if self.fixed_motif_finalization_mode not in valid_finalization_modes:
            raise ValueError(
                "fixed_motif_finalization_mode must be one of "
                f"{sorted(valid_finalization_modes)}"
            )
        valid_noise_modes = {"independent", "coupled"}
        if self.symmetry_noise_mode not in valid_noise_modes:
            raise ValueError(
                "symmetry_noise_mode must be one of " f"{sorted(valid_noise_modes)}"
            )
        exact_state = self.symmetry_state_mode == "orbit_average"
        coupled_noise = self.symmetry_noise_mode == "coupled"
        if exact_state != coupled_noise:
            raise ValueError(
                "Exact orbit sampling requires both "
                "symmetry_state_mode=orbit_average and "
                "symmetry_noise_mode=coupled"
            )
        if exact_state and self.allow_realignment:
            raise ValueError(
                "allow_realignment=True is incompatible with exact symmetry "
                "orbits because the runtime symmetry operators are not "
                "conjugated into the augmented coordinate frame"
            )
        if self.enable_orbit_rigid_motif_mobility and not exact_state:
            raise ValueError(
                "Orbit-rigid motif mobility requires exact orbit-average "
                "state and coupled-noise modes"
            )
        if self.enable_graph_interface_guidance and not exact_state:
            raise ValueError(
                "Graph interface guidance requires exact orbit-average "
                "state and coupled-noise modes"
            )
        if self.enable_symmetric_scaffold_packing and not exact_state:
            raise ValueError(
                "Symmetric scaffold packing requires exact orbit-average "
                "state and coupled-noise modes"
            )
        scaffold_core_active = (
            float(self.scaffold_core_intra_chain_weight) > 0.0
            or float(self.scaffold_core_inter_chain_excess_penalty) > 0.0
            or bool(self.enable_generated_cross_chain_topology_guidance)
        )
        robust_capture_active = bool(self.enable_supplied_interface_robust_capture)
        polymer_continuity_active = bool(
            self.enable_generated_polymer_continuity_guidance
        )
        if scaffold_core_active and not exact_state:
            raise ValueError(
                "Scaffold intra/inter guidance requires exact orbit-average "
                "state and coupled-noise modes"
            )
        if robust_capture_active and not exact_state:
            raise ValueError(
                "Supplied-interface robust capture requires exact orbit-average "
                "state and coupled-noise modes"
            )
        if robust_capture_active and not self.enable_orbit_rigid_motif_mobility:
            raise ValueError(
                "Supplied-interface robust capture requires one declared "
                "joint-rigid mobile motif orbit"
            )
        if robust_capture_active and self.motif_mobility_proposal_source != (
            "scaffold_boundary"
        ):
            raise ValueError(
                "Supplied-interface robust capture requires "
                "motif_mobility_proposal_source=scaffold_boundary"
            )
        if float(self.supplied_interface_capture_weight) < 0.0:
            raise ValueError("supplied_interface_capture_weight cannot be negative")
        if polymer_continuity_active and not exact_state:
            raise ValueError(
                "Generated polymer continuity guidance requires exact "
                "orbit-average state and coupled-noise modes"
            )
        if self.enable_generated_cross_chain_topology_guidance and not exact_state:
            raise ValueError(
                "Generated cross-chain topology guidance requires exact "
                "orbit-average state and coupled-noise modes"
            )
        if float(self.generated_routing_ownership_weight) < 0.0:
            raise ValueError("generated_routing_ownership_weight cannot be negative")
        if float(self.generated_polymer_continuity_target_ca_distance) <= 0.0:
            raise ValueError(
                "generated_polymer_continuity_target_ca_distance must be positive"
            )
        if float(self.generated_polymer_continuity_tolerance) < 0.0:
            raise ValueError(
                "generated_polymer_continuity_tolerance cannot be negative"
            )
        if int(self.generated_polymer_continuity_iterations) < 1:
            raise ValueError("generated_polymer_continuity_iterations must be positive")
        if float(self.scaffold_core_intra_chain_weight) < 0.0:
            raise ValueError("scaffold_core_intra_chain_weight cannot be negative")
        if not 0.0 <= float(self.scaffold_core_inter_chain_weight) <= 2.0:
            raise ValueError("scaffold_core_inter_chain_weight must be between 0 and 2")
        if float(self.scaffold_core_inter_chain_excess_penalty) < 0.0:
            raise ValueError(
                "scaffold_core_inter_chain_excess_penalty cannot be negative"
            )
        if (
            self.enable_graph_interface_guidance
            and self.enable_symmetric_scaffold_packing
        ):
            raise ValueError(
                "Declare graph interface guidance or automatic symmetric "
                "scaffold packing, not both"
            )
        if (
            self.enable_graph_interface_guidance
            or self.enable_symmetric_scaffold_packing
        ) and float(self.interface_seed_compactness_weight) > 0.0:
            raise ValueError(
                "Interface packing guidance cannot be combined with the "
                "legacy all-chain interface_seed_compactness force"
            )
        if scaffold_core_active and float(self.interface_seed_compactness_weight) > 0.0:
            raise ValueError(
                "Scaffold intra/inter guidance cannot be combined with the "
                "legacy interface_seed_compactness force"
            )
        if (
            self.enable_orbit_rigid_motif_mobility
            and not self.preserve_fixed_motif_during_symmetry
        ):
            raise ValueError(
                "Orbit-rigid motif mobility requires "
                "preserve_fixed_motif_during_symmetry=True"
            )
        if not (
            0.0
            <= float(self.motif_mobility_start_fraction)
            < float(self.motif_mobility_end_fraction)
            <= 1.0
        ):
            raise ValueError(
                "Motif mobility fractions must satisfy " "0 <= start < end <= 1"
            )
        if not 0.0 < float(self.motif_mobility_response) <= 1.0:
            raise ValueError("motif_mobility_response must be in (0, 1]")
        if float(self.motif_mobility_per_step_translation) <= 0.0:
            raise ValueError("motif_mobility_per_step_translation must be positive")
        if float(self.motif_mobility_per_step_rotation_degrees) <= 0.0:
            raise ValueError(
                "motif_mobility_per_step_rotation_degrees must be positive"
            )
        valid_proposal_sources = {"denoiser", "scaffold_boundary"}
        if self.motif_mobility_proposal_source not in valid_proposal_sources:
            raise ValueError(
                "motif_mobility_proposal_source must be one of "
                f"{sorted(valid_proposal_sources)}"
            )
        if int(self.motif_mobility_update_interval) <= 0:
            raise ValueError("motif_mobility_update_interval must be positive")
        if int(self.motif_mobility_target_update_count) < 0:
            raise ValueError("motif_mobility_target_update_count cannot be negative")
        capture_fraction = float(self.motif_mobility_capture_fraction)
        settle_fraction = float(self.motif_mobility_settle_fraction)
        if not (
            0.0 < capture_fraction < 1.0
            and 0.0 < settle_fraction < 1.0
            and capture_fraction + settle_fraction < 1.0
        ):
            raise ValueError(
                "motif mobility capture/settle fractions must be positive "
                "and leave a nonzero polish phase"
            )
        for name, value in (
            (
                "motif_mobility_capture_response_scale",
                self.motif_mobility_capture_response_scale,
            ),
            (
                "motif_mobility_expand_response_scale",
                self.motif_mobility_expand_response_scale,
            ),
            (
                "motif_mobility_polish_response_scale",
                self.motif_mobility_polish_response_scale,
            ),
        ):
            if not 0.0 < float(value) <= 5.0:
                raise ValueError(f"{name} must be in (0, 5]")
        if (
            self.enable_orbit_rigid_motif_mobility
            and self.motif_mobility_proposal_source == "scaffold_boundary"
            and float(self.interface_seed_compactness_weight) > 0.0
        ):
            raise ValueError(
                "Scaffold-derived motif mobility cannot be combined with "
                "interface_seed_compactness_weight"
            )
        if self.motif_mobility_proposal_source == "scaffold_boundary":
            # Constructing the config here validates every user-controlled
            # weight and geometric target before model inference starts.
            self._scaffold_guidance_config()
        if (
            self.enable_graph_interface_guidance
            or self.enable_symmetric_scaffold_packing
        ):
            self._graph_interface_guidance_config()
        if scaffold_core_active:
            self._scaffold_core_guidance_config()
        if (
            (
                self.enable_graph_interface_guidance
                or self.enable_symmetric_scaffold_packing
            )
            and self.enable_orbit_rigid_motif_mobility
            and self.motif_mobility_proposal_source != "scaffold_boundary"
        ):
            raise ValueError(
                "Packing-guided motif mobility requires the unified "
                "scaffold_boundary proposal path; denoiser-only motif motion "
                "cannot run beside graph interface guidance"
            )
        if float(self.symmetry_orbit_max_error) <= 0.0:
            raise ValueError("symmetry_orbit_max_error must be positive")
        if float(self.fixed_target_symmetry_rmsd_tolerance) < 0.0:
            raise ValueError("fixed_target_symmetry_rmsd_tolerance cannot be negative")
        if float(self.fixed_target_symmetry_max_tolerance) < 0.0:
            raise ValueError("fixed_target_symmetry_max_tolerance cannot be negative")
        valid_backends = {"explicit_all_copy", "local_neighbourhood"}
        if self.symmetry_execution_backend not in valid_backends:
            raise ValueError(
                "symmetry_execution_backend must be one of " f"{sorted(valid_backends)}"
            )
        if int(self.symmetry_neighbour_radius) < 0:
            raise ValueError("symmetry_neighbour_radius cannot be negative")
        if self._uses_local_symmetry_neighbourhood:
            if not exact_state:
                raise ValueError(
                    "local_neighbourhood requires symmetry_state_mode="
                    "orbit_average and symmetry_noise_mode=coupled"
                )
            if not self.preserve_fixed_motif_during_symmetry:
                raise ValueError(
                    "local_neighbourhood requires "
                    "preserve_fixed_motif_during_symmetry=True"
                )
            if self.enable_orbit_rigid_motif_mobility:
                raise ValueError(
                    "local_neighbourhood does not yet support dynamic motif " "mobility"
                )

    @property
    def _uses_exact_symmetry_orbits(self) -> bool:
        return self.symmetry_state_mode == "orbit_average"

    @property
    def _uses_local_symmetry_neighbourhood(self) -> bool:
        return self.symmetry_execution_backend == "local_neighbourhood"

    def prepare_local_network_view(
        self,
        f: dict[str, Any],
        coordinates: torch.Tensor,
    ) -> LocalSymmetryRuntimeContext | None:
        """Build the bounded feature view before TokenInitializer runs."""

        if not self._uses_local_symmetry_neighbourhood:
            return None
        symmetry_id = f.get("symmetry_id")
        if not isinstance(symmetry_id, str) or not symmetry_id:
            raise ValueError(
                "local_neighbourhood requires the runtime symmetry_id feature"
            )
        layout = build_symmetry_orbit_layout(
            self._symmetry_features(f),
            like=coordinates,
        )
        neighbourhood = build_local_symmetry_neighbourhood(
            self._symmetry_features(f),
            symmetry_id,
            like=coordinates,
            neighbour_radius=int(self.symmetry_neighbour_radius),
            include_dihedral_mate=bool(self.symmetry_include_dihedral_mate),
            layout=layout,
        )
        feature_view = crop_features_to_local_neighbourhood(
            f,
            neighbourhood,
        )
        ranked_logger.info(
            "Local symmetry network view prepared: "
            f"symmetry={symmetry_id}, "
            f"copies={neighbourhood.copy_count}, "
            f"atoms={len(neighbourhood.atom_indices)}/"
            f"{neighbourhood.full_atom_count}, "
            f"tokens={len(feature_view.token_indices)}"
        )
        return LocalSymmetryRuntimeContext(
            layout=layout,
            neighbourhood=neighbourhood,
            feature_view=feature_view,
        )

    def _scaffold_guidance_config(self) -> ScaffoldGuidanceConfig:
        return ScaffoldGuidanceConfig(
            junction_weight=float(self.motif_mobility_junction_weight),
            clash_weight=float(self.motif_mobility_clash_weight),
            tilt_weight=float(self.motif_mobility_tilt_weight),
            prior_weight=float(self.motif_mobility_prior_weight),
            junction_target_distance=float(
                self.motif_mobility_junction_target_distance
            ),
            clash_distance=float(self.motif_mobility_clash_distance),
            maximum_tilt_degrees=float(self.motif_mobility_target_max_tilt_degrees),
        )

    def _graph_interface_guidance_config(
        self,
    ) -> GraphInterfaceGuidanceConfig:
        return GraphInterfaceGuidanceConfig(
            weight=float(self.graph_interface_guidance_weight),
            contact_prior_weight=float(
                self.graph_interface_guidance_contact_prior_weight
            ),
            contact_prior_guide_scale=float(
                self.graph_interface_guidance_contact_prior_guide_scale
            ),
            contact_prior_decay_power=float(
                self.graph_interface_guidance_contact_prior_decay_power
            ),
            contact_prior_r_0=float(self.graph_interface_guidance_contact_prior_r_0),
            contact_prior_d_0=float(self.graph_interface_guidance_contact_prior_d_0),
            coverage_weight=float(self.graph_interface_guidance_coverage_weight),
            continuity_weight=float(self.graph_interface_guidance_continuity_weight),
            orientation_weight=float(self.graph_interface_guidance_orientation_weight),
            shape_weight=float(self.graph_interface_guidance_shape_weight),
            backbone_weight=float(self.graph_interface_guidance_backbone_weight),
            interface_balance_weight=float(
                self.graph_interface_guidance_interface_balance_weight
            ),
            patch_exclusivity_weight=float(
                self.graph_interface_guidance_patch_exclusivity_weight
            ),
            clash_weight=float(self.graph_interface_guidance_clash_weight),
            distance_weight=float(self.graph_interface_guidance_distance_weight),
            target_ca_distance=float(self.graph_interface_guidance_target_ca_distance),
            clash_ca_distance=float(self.graph_interface_guidance_clash_ca_distance),
            pairs_per_edge=int(self.graph_interface_guidance_pairs_per_edge),
            start_fraction=float(self.graph_interface_guidance_start_fraction),
            end_fraction=float(self.graph_interface_guidance_end_fraction),
            terminal_weight_floor=float(
                self.graph_interface_guidance_terminal_weight_floor
            ),
            maximum_token_step=float(self.graph_interface_guidance_maximum_token_step),
            unsatisfied_step_fraction=float(
                self.graph_interface_guidance_unsatisfied_step_fraction
            ),
            final_polish_steps=int(self.graph_interface_guidance_final_polish_steps),
            token_smoothing_weight=float(
                self.graph_interface_guidance_token_smoothing_weight
            ),
            token_smoothing_passes=int(
                self.graph_interface_guidance_token_smoothing_passes
            ),
            continuity_softness=float(
                self.graph_interface_guidance_continuity_softness
            ),
            maximum_tangent_normal_cosine=float(
                self.graph_interface_guidance_maximum_tangent_normal_cosine
            ),
            backbone_ca_distance=float(
                self.graph_interface_guidance_backbone_ca_distance
            ),
            backbone_ca_tolerance=float(
                self.graph_interface_guidance_backbone_ca_tolerance
            ),
            patch_rigid_weight=float(self.graph_interface_guidance_patch_rigid_weight),
            patch_blend_radius=int(self.graph_interface_guidance_patch_blend_radius),
            maximum_patch_rotation_degrees=float(
                self.graph_interface_guidance_maximum_patch_rotation_degrees
            ),
            patch_lock_fraction=float(
                self.graph_interface_guidance_patch_lock_fraction
            ),
            line_search_steps=int(self.graph_interface_guidance_line_search_steps),
            line_search_contraction=float(
                self.graph_interface_guidance_line_search_contraction
            ),
            capture_ca_distance=float(
                self.graph_interface_guidance_capture_ca_distance
            ),
            maximum_orientation_loss=float(
                self.graph_interface_guidance_maximum_orientation_loss
            ),
            maximum_shape_loss=float(self.graph_interface_guidance_maximum_shape_loss),
            maximum_backbone_loss=float(
                self.graph_interface_guidance_maximum_backbone_loss
            ),
            maximum_patch_exclusivity_loss=float(
                self.graph_interface_guidance_maximum_patch_exclusivity_loss
            ),
        )

    def _scaffold_core_guidance_config(self) -> ScaffoldCoreGuidanceConfig:
        """Resolve the two public intra/inter controls into safe defaults."""

        return ScaffoldCoreGuidanceConfig(
            intra_chain_weight=float(self.scaffold_core_intra_chain_weight),
            inter_chain_weight=float(self.scaffold_core_inter_chain_weight),
            inter_chain_excess_penalty=float(
                self.scaffold_core_inter_chain_excess_penalty
            ),
            routing_ownership_weight=(
                float(self.generated_routing_ownership_weight)
                if self.enable_generated_cross_chain_topology_guidance
                else 0.0
            ),
        )

    @staticmethod
    def _declared_mobile_motif_orbit_count(
        f: dict[str, Any],
    ) -> int:
        mobility_modes = f.get("motif_constraint_orbit_mobility_mode")
        if mobility_modes is None:
            return 0
        mobility_modes = torch.as_tensor(mobility_modes, dtype=torch.long)
        if mobility_modes.ndim != 1:
            raise ValueError(
                "motif_constraint_orbit_mobility_mode must be one-dimensional"
            )
        if torch.any((mobility_modes != 0) & (mobility_modes != 1)):
            raise ValueError(
                "motif_constraint_orbit_mobility_mode may contain only 0 or 1"
            )
        return int(torch.count_nonzero(mobility_modes == 1).item())

    def _validate_motif_mobility_runtime(
        self,
        f: dict[str, Any],
        *,
        diffusion_batch_size: int,
        initializer_outputs: dict[str, Any],
    ) -> int:
        """Fail closed when the input and sampler mobility modes disagree."""

        mobile_orbit_count = self._declared_mobile_motif_orbit_count(f)
        if mobile_orbit_count and not self.enable_orbit_rigid_motif_mobility:
            raise ValueError(
                "The input declares orbit-rigid motif mobility but "
                "enable_orbit_rigid_motif_mobility=False"
            )
        if self.enable_orbit_rigid_motif_mobility and not mobile_orbit_count:
            raise ValueError(
                "Orbit-rigid motif mobility was enabled but the input "
                "declares no mobile motif constraint orbit"
            )
        if not self.enable_orbit_rigid_motif_mobility:
            return 0
        if diffusion_batch_size != 1:
            raise ValueError(
                "Dynamic motif conditioning currently requires "
                "diffusion_batch_size=1"
            )
        if not self._uses_exact_symmetry_orbits:
            raise ValueError(
                "Dynamic motif conditioning requires "
                "symmetry_state_mode=orbit_average"
            )
        if self.symmetry_noise_mode != "coupled":
            raise ValueError(
                "Dynamic motif conditioning requires " "symmetry_noise_mode=coupled"
            )
        if not self.preserve_fixed_motif_during_symmetry:
            raise ValueError(
                "Dynamic motif conditioning requires "
                "preserve_fixed_motif_during_symmetry=True"
            )
        if "chunked_pairwise_embedder" not in initializer_outputs:
            raise ValueError(
                "Dynamic motif conditioning currently requires the "
                "chunked/low-memory P_LL path"
            )
        if (
            self.motif_mobility_proposal_source == "denoiser"
            and not self.motif_mobility_apply_updates
        ):
            raise ValueError(
                "Proposal-only motif mobility requires "
                "motif_mobility_proposal_source=scaffold_boundary"
            )
        return mobile_orbit_count

    @staticmethod
    def _copy_motif_mobility_runtime_features(
        f: dict[str, Any],
    ) -> dict[str, Any]:
        """Detach mutable runtime conditioning from the input feature mapping."""

        if "motif_pos" not in f:
            raise ValueError("Dynamic motif conditioning requires f['motif_pos']")
        runtime_f = dict(f)
        runtime_f["motif_pos"] = torch.as_tensor(f["motif_pos"]).clone()
        return runtime_f

    @staticmethod
    def _synchronize_mobile_motif_conditioning(
        f: dict[str, Any],
        fixed_target: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> None:
        """Synchronize pair conditioning and hard group targets to one pose."""

        if fixed_target.ndim != 3 or fixed_target.shape[0] != 1:
            raise ValueError("Dynamic motif target must have shape [1, L, 3]")
        if not torch.isfinite(fixed_target).all():
            raise ValueError("Dynamic motif target contains NaN or Inf")
        fixed = torch.as_tensor(
            fixed_mask,
            dtype=torch.bool,
            device=fixed_target.device,
        )
        if fixed.ndim != 1 or fixed.shape[0] != fixed_target.shape[1]:
            raise ValueError("Dynamic motif fixed mask must have shape [L]")
        motif_pos = torch.as_tensor(
            f["motif_pos"],
            device=fixed_target.device,
        )
        if tuple(motif_pos.shape) != tuple(fixed_target.shape[1:]):
            raise ValueError(
                "f['motif_pos'] must have shape [L, 3] for dynamic "
                "motif conditioning"
            )
        updated_motif_pos = motif_pos.clone()
        updated_motif_pos[fixed] = fixed_target[0, fixed].to(
            dtype=updated_motif_pos.dtype
        )
        f["motif_pos"] = updated_motif_pos

        membership = f.get("motif_constraint_group_membership")
        if membership is None:
            f.pop("motif_constraint_target_coordinates", None)
            return
        membership = torch.as_tensor(
            membership,
            dtype=torch.bool,
            device=fixed_target.device,
        )
        if membership.ndim != 2 or membership.shape[1] != fixed_target.shape[1]:
            raise ValueError("motif_constraint_group_membership must have shape [G, L]")
        f["motif_constraint_target_coordinates"] = (
            fixed_target[:, None, :, :].expand(-1, membership.shape[0], -1, -1).clone()
        )

    @staticmethod
    def _symmetry_features(f: dict[str, Any]) -> dict[str, Any]:
        required = {
            "sym_transform",
            "sym_transform_id",
            "sym_entity_id",
            "is_sym_asu",
        }
        missing = required - set(f)
        if missing:
            raise ValueError(f"Symmetry sampling requires features {sorted(missing)}")
        return {
            key: value
            for key, value in f.items()
            if key.startswith("sym_") or key == "is_sym_asu"
        }

    def apply_symmetry_to_X_L(self, X_L, f):
        # check that we are doing symmetric inference

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        # update symmetric frames to correct for change in global frame
        symmetry_feats = self._symmetry_features(f)

        # apply symmetry frame shift to X_L
        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def apply_orbit_average_to_X_L(self, X_L, f):
        """Project coordinates using all copies in each runtime orbit."""

        return project_symmetry_orbit_average(
            X_L,
            self._symmetry_features(f),
            partial_diffusion=True,
            layout=self._exact_symmetry_orbit_layout,
        )

    def _project_symmetric_state(self, X_L, f):
        if self._uses_exact_symmetry_orbits:
            return self.apply_orbit_average_to_X_L(X_L, f)
        return self.apply_symmetry_to_X_L(X_L, f)

    def _joint_projector(
        self,
        f: dict[str, Any],
    ) -> UnifiedJointProjector:
        """Bind runtime features to the topology-neutral projection contract."""

        return UnifiedJointProjector(
            project_symmetry=lambda coordinates: (
                self._project_symmetric_state(coordinates, f)
            ),
            restore_constraints=lambda coordinates, target, mask: (
                self._restore_motif_constraint_groups(
                    coordinates,
                    target,
                    mask,
                    f,
                )
            ),
            validate_closure=lambda coordinates, label: (
                self._assert_symmetry_orbit_closed(
                    coordinates,
                    f,
                    label=label,
                )
            ),
        )

    def _cylindrical_projector(
        self,
        f: dict[str, Any],
        reference: torch.Tensor,
    ) -> CylindricalCoordinateProjector | None:
        keep_mask = f.get("cylindrical_keep_mask")
        if keep_mask is None:
            return None
        if self.allow_realignment:
            raise ValueError(
                "Cylindrical hard projection requires "
                "inference_sampler.allow_realignment=False so its declared "
                "symmetry axis remains in the runtime coordinate frame"
            )
        transforms = self._symmetry_features(f)["sym_transform"]
        translations = [
            torch.as_tensor(
                transform[1],
                dtype=reference.dtype,
                device=reference.device,
            )
            for _, transform in sorted(
                transforms.items(),
                key=lambda item: int(item[0]),
            )
        ]
        if not translations:
            raise ValueError(
                "Cylindrical hard projection requires runtime symmetry " "transforms"
            )
        # For a finite affine group the barycentre of the orbit of the
        # origin is a group-fixed point.  Its component along the axis is a
        # harmless gauge: cylindrical radius, azimuth and the reconstructed
        # axial coordinate are invariant to shifting the chosen centre along
        # that axis line.
        center = torch.stack(translations, dim=0).mean(dim=0)
        return CylindricalCoordinateProjector(
            reference=reference,
            keep_mask=torch.as_tensor(
                keep_mask,
                dtype=torch.bool,
                device=reference.device,
            ),
            axis=torch.as_tensor(
                f["cylindrical_axis"],
                dtype=reference.dtype,
                device=reference.device,
            ),
            center=center,
        )

    def _assert_symmetry_orbit_closed(
        self,
        X_L: torch.Tensor,
        f: dict[str, Any],
        *,
        label: str,
    ) -> None:
        if not self._uses_exact_symmetry_orbits:
            return
        if not torch.isfinite(X_L).all():
            raise ValueError(f"{label} contains NaN or Inf coordinates")
        rms, maximum = symmetry_orbit_residual(
            X_L,
            self._symmetry_features(f),
            layout=self._exact_symmetry_orbit_layout,
        )
        if not (torch.isfinite(rms).all() and torch.isfinite(maximum).all()):
            raise ValueError(f"{label} produced a non-finite symmetry residual")
        # A fixed 1e-3 A gate is appropriate once coordinates return to
        # molecular scale, but float32 cannot represent an exactly idempotent
        # rotation round-trip at the initial EDM noise scale (~2560 A).
        # Retain the absolute scientific gate while adding only a small
        # dtype-relative numerical floor.  This floor vanishes below the
        # configured tolerance as denoising approaches the final structure.
        configured_tolerance = float(self.symmetry_orbit_max_error)
        tolerance, numerical_floor = symmetry_orbit_tolerance(
            X_L,
            configured_tolerance=configured_tolerance,
        )
        if torch.any(maximum > tolerance):
            raise ValueError(
                f"{label} left the runtime symmetry orbit: "
                f"maximum residual={float(maximum.max().item()):.6f} A, "
                f"RMS residual={float(rms.max().item()):.6f} A, "
                f"tolerance={tolerance:.6f} A "
                f"(configured={configured_tolerance:.6f} A, "
                f"roundoff_floor={numerical_floor:.6f} A)"
            )

    def _prepare_static_symmetry_target(
        self,
        coordinates: torch.Tensor,
        fixed_mask: torch.Tensor,
        f: dict[str, Any],
        *,
        diffusion_batch_size: int,
    ) -> torch.Tensor:
        """Validate and project the immutable cross-chain target onto Cn/Dn."""

        if coordinates.shape[0] == 1 and diffusion_batch_size > 1:
            coordinates = coordinates.expand(
                diffusion_batch_size,
                -1,
                -1,
            )
        elif coordinates.shape[0] != diffusion_batch_size:
            raise ValueError(
                "Fixed target batch dimension must be one or match "
                "diffusion_batch_size"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("Fixed motif target contains NaN or Inf coordinates")
        symmetry_feats = self._symmetry_features(f)
        mismatch_count = symmetry_orbit_mask_mismatch_count(
            fixed_mask,
            symmetry_feats,
            layout=self._exact_symmetry_orbit_layout,
        )
        if mismatch_count:
            raise ValueError(
                "Fixed motif mask is not closed over the runtime symmetry "
                f"orbits ({mismatch_count} mismatched atom slots)"
            )

        rms, maximum = symmetry_orbit_residual(
            coordinates,
            symmetry_feats,
            atom_mask=fixed_mask,
            layout=self._exact_symmetry_orbit_layout,
        )
        if not (torch.isfinite(rms).all() and torch.isfinite(maximum).all()):
            raise ValueError(
                "Fixed motif target produced a non-finite symmetry residual"
            )
        rms_tolerance = float(self.fixed_target_symmetry_rmsd_tolerance)
        max_tolerance = float(self.fixed_target_symmetry_max_tolerance)
        if torch.any(rms > rms_tolerance) or torch.any(maximum > max_tolerance):
            raise ValueError(
                "Fixed motif target is incompatible with the runtime "
                "symmetry operators: "
                f"RMS={float(rms.max().item()):.6f} A "
                f"(limit {rms_tolerance:.6f}), "
                f"max={float(maximum.max().item()):.6f} A "
                f"(limit {max_tolerance:.6f})"
            )
        # Use the unique orbit-average target during sampling.  For a valid
        # target this is a sub-CIF-precision correction, and it prevents an
        # exact projector and a nearly symmetric hard constraint from
        # fighting at every step.
        projected = project_symmetry_orbit_average(
            coordinates,
            symmetry_feats,
            partial_diffusion=True,
            layout=self._exact_symmetry_orbit_layout,
        )
        if not torch.isfinite(projected).all():
            raise ValueError("Projected fixed motif target contains NaN or Inf")
        return projected

    def _get_exact_symmetric_initial_structure(
        self,
        *,
        c0: torch.Tensor,
        D: int,
        fixed_target: torch.Tensor,
        fixed_mask: torch.Tensor,
        f: dict[str, Any],
        constraint_runtime: MosaicConstraintRuntime | None = None,
    ) -> torch.Tensor:
        raw_noise = c0 * torch.normal(
            mean=0.0,
            std=1.0,
            size=fixed_target.shape,
            device=fixed_target.device,
        )
        noise = expand_symmetry_coupled_displacements(
            raw_noise,
            self._symmetry_features(f),
            layout=self._exact_symmetry_orbit_layout,
        )
        noise[..., fixed_mask, :] = 0.0
        X_L = fixed_target + noise
        if constraint_runtime is not None:
            return constraint_runtime.initialize_state(X_L)
        X_L = self.apply_orbit_average_to_X_L(X_L, f)
        X_L = self._restore_motif_constraint_groups(
            X_L,
            fixed_target,
            fixed_mask,
            f,
        )
        self._assert_symmetry_orbit_closed(
            X_L,
            f,
            label="Initial diffusion state",
        )
        return X_L

    def _apply_symmetry_preserving_fixed_motif(
        self,
        X_L: torch.Tensor,
        f: dict[str, Any],
        is_motif_atom_with_fixed_coord: torch.Tensor,
        *,
        fixed_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project generated coordinates while retaining the complete motif.

        Indexed motif fragments may belong to different symmetry copies.  A
        native atomwise symmetry projection can therefore change their
        cross-chain relative pose.  For Mosaic's opt-in hard-motif mode, save
        the complete motif in the current augmented frame, project the whole
        structure, and restore all fixed atoms as one coordinate set.
        """

        if not self.preserve_fixed_motif_during_symmetry:
            return self._project_symmetric_state(X_L, f)
        fixed = is_motif_atom_with_fixed_coord.bool()
        if not torch.any(fixed):
            return self._project_symmetric_state(X_L, f)
        if fixed_coordinates is None:
            fixed_coordinates = X_L
        return self._joint_projector(f).project(
            X_L,
            constraint_target=fixed_coordinates,
            constraint_mask=fixed,
            restore=True,
            label="Fixed-motif symmetry projection",
        )

    def _restore_motif_constraint_groups(
        self,
        X_L: torch.Tensor,
        fixed_coordinates: torch.Tensor,
        fixed_mask: torch.Tensor,
        f: dict[str, Any],
    ) -> torch.Tensor:
        """Restore complete motif groups without insertion-order dependence.

        ``motif_constraint_group_membership`` is an optional boolean tensor
        shaped ``[G, L]``.  Atoms may belong to multiple groups.  Overlapping
        groups are merged only when their target coordinates agree within the
        configured tolerance; conflicting hard constraints fail explicitly.
        Without group metadata, retain the legacy one-global-mask behavior.
        """

        membership = f.get("motif_constraint_group_membership")
        fixed = fixed_mask.bool()
        restored = X_L.clone()
        if membership is None:
            if self.require_motif_constraint_groups and torch.any(fixed):
                raise ValueError(
                    "Motif constraint groups are required but "
                    "motif_constraint_group_membership is absent"
                )
            restored[..., fixed, :] = fixed_coordinates[..., fixed, :]
            return restored

        membership = torch.as_tensor(
            membership,
            dtype=torch.bool,
            device=X_L.device,
        )
        if membership.ndim != 2 or membership.shape[1] != X_L.shape[-2]:
            raise ValueError("motif_constraint_group_membership must have shape [G, L]")
        assigned = membership.any(dim=0)
        if torch.any(assigned & ~fixed):
            raise ValueError(
                "Motif constraint groups may contain only fixed motif atoms"
            )
        if torch.any(fixed & ~assigned):
            raise ValueError("Every fixed motif atom must belong to a constraint group")

        group_targets = f.get("motif_constraint_target_coordinates")
        if group_targets is None:
            group_targets = fixed_coordinates[:, None, :, :].expand(
                -1,
                membership.shape[0],
                -1,
                -1,
            )
        else:
            group_targets = torch.as_tensor(
                group_targets,
                dtype=X_L.dtype,
                device=X_L.device,
            )
        expected_shape = (
            X_L.shape[0],
            membership.shape[0],
            X_L.shape[1],
            3,
        )
        if tuple(group_targets.shape) != expected_shape:
            raise ValueError(
                "motif_constraint_target_coordinates must have shape "
                f"{expected_shape}"
            )
        if not torch.isfinite(group_targets).all():
            raise ValueError("motif_constraint_target_coordinates contains NaN or Inf")

        weights = membership.to(dtype=X_L.dtype)[None, :, :, None]
        counts = weights.sum(dim=1)
        merged = (group_targets * weights).sum(dim=1) / counts.clamp_min(1.0)
        deviations = torch.linalg.vector_norm(
            group_targets - merged[:, None, :, :],
            dim=-1,
        )
        relevant_deviations = deviations[
            membership[None, :, :].expand(X_L.shape[0], -1, -1)
        ]
        tolerance = float(self.motif_constraint_conflict_tolerance)
        if tolerance < 0.0:
            raise ValueError("motif_constraint_conflict_tolerance cannot be negative")
        if relevant_deviations.numel() and torch.any(relevant_deviations > tolerance):
            maximum = float(relevant_deviations.max().item())
            raise ValueError(
                "Overlapping motif constraint groups disagree: maximum "
                f"coordinate deviation {maximum:.6f} exceeds {tolerance:.6f}"
            )
        restored[..., assigned, :] = merged[..., assigned, :]
        return restored

    def _apply_interface_seed_compactness(
        self,
        X_L: torch.Tensor,
        f: dict[str, Any],
        is_motif_atom_with_fixed_coord: torch.Tensor,
        *,
        step_num: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Gently compact generated residues around each chain's anchors.

        The update is token-rigid: every atom in a generated residue receives
        the same translation.  Fixed motif atoms never move.  Guidance fades
        linearly to zero before the final denoising steps so the network can
        repair local geometry without a late external force.
        """

        weight = float(self.interface_seed_compactness_weight)
        if weight <= 0.0 or num_steps <= 0:
            return X_L
        end_frac = float(self.interface_seed_compactness_end_frac)
        if not 0.0 < end_frac <= 1.0:
            raise ValueError("interface_seed_compactness_end_frac must be in (0, 1]")
        max_step = float(self.interface_seed_compactness_max_step)
        if max_step <= 0.0:
            raise ValueError("interface_seed_compactness_max_step must be positive")
        required = {"atom_to_token_map", "asym_id"}
        missing = required - set(f)
        if missing:
            raise ValueError(
                "Interface-Seed compactness guidance requires features "
                f"{sorted(missing)}"
            )

        progress = step_num / max(num_steps - 1, 1)
        if progress >= end_frac:
            return X_L
        schedule = 1.0 - progress / end_frac

        atom_to_token = f["atom_to_token_map"].long()
        atom_chain = f["asym_id"].long()[atom_to_token]
        fixed = is_motif_atom_with_fixed_coord.bool()
        guided = X_L.clone()

        for chain_id in torch.unique(atom_chain):
            chain_mask = atom_chain == chain_id
            anchor_mask = chain_mask & fixed
            generated_mask = chain_mask & ~fixed
            if not torch.any(anchor_mask) or not torch.any(generated_mask):
                continue
            anchor_center = X_L[:, anchor_mask, :].mean(dim=1)
            generated_tokens = torch.unique(atom_to_token[generated_mask])
            for token_id in generated_tokens:
                token_mask = generated_mask & (atom_to_token == token_id)
                token_center = X_L[:, token_mask, :].mean(dim=1)
                displacement = (anchor_center - token_center) * (weight * schedule)
                norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
                scale = torch.clamp(max_step / torch.clamp(norm, min=1e-8), max=1.0)
                guided[:, token_mask, :] += (displacement * scale)[:, None, :]
        return guided

    def _project_stepwise_updated_coordinates(
        self,
        X_L: torch.Tensor,
        f: dict[str, Any],
        is_motif_atom_with_fixed_coord: torch.Tensor,
        fixed_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce symmetry after the stochastic coordinate update.

        Projecting only the denoised prediction is insufficient because
        ``X_noisy_L`` contains independent atomwise noise.  The Euler update
        therefore need not remain symmetric even when ``X_denoised_L`` is
        symmetric.  Mosaic's hard-motif mode projects the actual updated
        coordinates and then restores complete cross-chain motif groups.
        """

        if self._uses_exact_symmetry_orbits:
            return self._joint_projector(f).project(
                X_L,
                constraint_target=fixed_coordinates,
                constraint_mask=is_motif_atom_with_fixed_coord,
                restore=self.preserve_fixed_motif_during_symmetry,
                label="Euler-updated diffusion state",
            )
        if not self.preserve_fixed_motif_during_symmetry:
            return X_L
        return self._apply_symmetry_preserving_fixed_motif(
            X_L,
            f,
            is_motif_atom_with_fixed_coord,
            fixed_coordinates=fixed_coordinates,
        )

    def _finalize_with_fixed_motif(
        self,
        X_L: torch.Tensor,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord: torch.Tensor,
        f: dict[str, Any],
    ) -> torch.Tensor:
        """Finalize symmetry while giving the complete fixed motif precedence.

        A fixed indexed motif may contain fragments on different protomers.
        Applying symmetry after motif insertion projects those fragments
        independently and can preserve each fragment while destroying their
        cross-chain interface.  Symmetrize the generated scaffold first, then
        insert and align the complete motif as one coordinate set.
        """

        if self.fixed_motif_finalization_mode == "official_reinsert_then_project":
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )
            X_L = self.apply_symmetry_to_X_L(X_L, f)
            return weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        if not self.preserve_fixed_motif_during_symmetry:
            X_L = self.apply_symmetry_to_X_L(X_L, f)
        X_L, _ = centre_random_augment_around_motif(
            X_L,
            coord_atom_lvl_to_be_noised,
            is_motif_atom_with_fixed_coord,
            reinsert_motif=self.insert_motif_at_end,
        )
        return weighted_rigid_align(
            coord_atom_lvl_to_be_noised,
            X_L,
            X_exists_L=is_motif_atom_with_fixed_coord,
        )

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        network_f: dict[str, Any] | None = None,
        local_symmetry_context: LocalSymmetryRuntimeContext | None = None,
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        **_,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = torch.as_tensor(
            f["is_motif_atom_with_fixed_coord"],
            dtype=torch.bool,
            device=coord_atom_lvl_to_be_noised.device,
        )
        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        denoiser_f = f if network_f is None else network_f
        if self._uses_local_symmetry_neighbourhood:
            if local_symmetry_context is None or network_f is None:
                raise ValueError(
                    "local_neighbourhood requires a prepared local network "
                    "feature view"
                )
            if "chunked_pairwise_embedder" not in initializer_outputs:
                raise ValueError(
                    "local_neighbourhood currently requires low_memory_mode=True"
                )
        elif local_symmetry_context is not None or network_f is not None:
            raise ValueError("A local network view was supplied to explicit_all_copy")
        mobile_orbit_count = self._validate_motif_mobility_runtime(
            f,
            diffusion_batch_size=D,
            initializer_outputs=initializer_outputs,
        )
        if mobile_orbit_count:
            f = self._copy_motif_mobility_runtime_features(f)
        fixed_target = coord_atom_lvl_to_be_noised.clone()
        motif_mobility_controller = None
        scaffold_guidance_topology = None
        scaffold_guidance_axis = None
        scaffold_guidance_principal_axes = None
        scaffold_guidance_config = None
        graph_interface_topology = None
        graph_interface_guidance_config = None
        graph_interface_patch_state = None
        graph_interface_diagnostics: list[dict[str, Any]] = []
        scaffold_core_topology = None
        scaffold_core_guidance_config = None
        scaffold_core_diagnostics: list[dict[str, Any]] = []
        polymer_continuity_diagnostics: list[dict[str, Any]] = []
        scaffold_core_active = (
            float(self.scaffold_core_intra_chain_weight) > 0.0
            or float(self.scaffold_core_inter_chain_excess_penalty) > 0.0
            or bool(self.enable_generated_cross_chain_topology_guidance)
        )
        robust_capture_active = bool(self.enable_supplied_interface_robust_capture)
        scaffold_core_topology_required = scaffold_core_active or robust_capture_active
        polymer_continuity_active = bool(
            self.enable_generated_polymer_continuity_guidance
        )
        joint_packing_mobility = False
        mobility_schedule_diagnostics = None
        effective_motif_mobility_update_interval = int(
            self.motif_mobility_update_interval
        )
        constraint_runtime = None
        if self.enable_graph_interface_guidance:
            graph_interface_topology = build_graph_interface_topology(
                f,
                is_motif_atom_with_fixed_coord,
            )
            if graph_interface_topology is None:
                raise ValueError(
                    "Graph interface guidance was enabled but the input "
                    "declares no required output-stage contact relation"
                )
            graph_interface_guidance_config = self._graph_interface_guidance_config()
            graph_interface_patch_state = GraphInterfacePatchState(assignments={})
            ranked_logger.info(
                "Graph interface guidance initialized: "
                f"edges={len(graph_interface_topology.edges)}"
            )
        elif self.enable_symmetric_scaffold_packing:
            graph_interface_topology = build_symmetric_scaffold_interface_topology(
                f,
                is_motif_atom_with_fixed_coord,
            )
            graph_interface_guidance_config = self._graph_interface_guidance_config()
            graph_interface_patch_state = GraphInterfacePatchState(assignments={})
            ranked_logger.info(
                "Automatic symmetric scaffold packing initialized: "
                f"edges={len(graph_interface_topology.edges)}"
            )
        if scaffold_core_topology_required or polymer_continuity_active:
            scaffold_core_topology = build_scaffold_core_topology(
                f,
                is_motif_atom_with_fixed_coord,
            )
        if scaffold_core_topology_required:
            scaffold_core_guidance_config = self._scaffold_core_guidance_config()
            ranked_logger.info(
                "Scaffold intra/inter guidance initialized: "
                f"chains={len(scaffold_core_topology.chains)}, "
                "intra="
                f"{scaffold_core_guidance_config.intra_chain_weight}, "
                "inter="
                f"{scaffold_core_guidance_config.inter_chain_weight}"
                ", routing="
                f"{scaffold_core_guidance_config.routing_ownership_weight}"
            )
        if self._uses_exact_symmetry_orbits:
            self._exact_symmetry_orbit_layout = (
                local_symmetry_context.layout
                if local_symmetry_context is not None
                else build_symmetry_orbit_layout(
                    self._symmetry_features(f),
                    like=fixed_target,
                )
            )
            if (
                torch.any(is_motif_atom_with_fixed_coord)
                and not self.preserve_fixed_motif_during_symmetry
            ):
                raise ValueError(
                    "Exact symmetry-orbit sampling with fixed motif atoms "
                    "requires preserve_fixed_motif_during_symmetry=True"
                )
            fixed_target = self._prepare_static_symmetry_target(
                fixed_target,
                is_motif_atom_with_fixed_coord,
                f,
                diffusion_batch_size=D,
            )
            if self.enable_orbit_rigid_motif_mobility:
                ranked_logger.info(
                    "Orbit-rigid motif mobility is active: Mosaic preserves "
                    "the complete internal motif geometry and exact symmetry "
                    "orbit while applying bounded master-pose updates"
                )
                motif_mobility_controller = OrbitRigidMotifController.from_features(
                    f,
                    fixed_target,
                    start_fraction=float(self.motif_mobility_start_fraction),
                    end_fraction=float(self.motif_mobility_end_fraction),
                    response=float(self.motif_mobility_response),
                    per_step_translation=float(
                        self.motif_mobility_per_step_translation
                    ),
                    per_step_rotation_degrees=float(
                        self.motif_mobility_per_step_rotation_degrees
                    ),
                    capture_fraction=float(self.motif_mobility_capture_fraction),
                    settle_fraction=float(self.motif_mobility_settle_fraction),
                    capture_response_scale=float(
                        self.motif_mobility_capture_response_scale
                    ),
                    settle_response_scale=float(
                        self.motif_mobility_expand_response_scale
                    ),
                    polish_response_scale=float(
                        self.motif_mobility_polish_response_scale
                    ),
                )
                if motif_mobility_controller is None:
                    raise ValueError(
                        "Orbit-rigid motif mobility was enabled but the "
                        "input declares no mobile motif constraint orbit"
                    )
                mobility_schedule_diagnostics = _motif_mobility_proposal_schedule(
                    total_steps=max(int(noise_schedule.numel()) - 1, 1),
                    configured_interval=int(self.motif_mobility_update_interval),
                    target_update_count=int(self.motif_mobility_target_update_count),
                    windows=tuple(
                        (
                            float(
                                getattr(
                                    motif,
                                    "start_fraction",
                                    self.motif_mobility_start_fraction,
                                )
                            ),
                            float(
                                getattr(
                                    motif,
                                    "end_fraction",
                                    self.motif_mobility_end_fraction,
                                )
                            ),
                        )
                        for motif in motif_mobility_controller.motifs
                    ),
                    always_propose=(self.motif_mobility_proposal_source == "denoiser"),
                )
                effective_motif_mobility_update_interval = int(
                    mobility_schedule_diagnostics["effective_update_interval"]
                )
                if self.motif_mobility_proposal_source == "scaffold_boundary":
                    scaffold_guidance_topology = build_boundary_topology(
                        f,
                        is_motif_atom_with_fixed_coord,
                    )
                    is_ca = torch.as_tensor(
                        f["is_ca"],
                        dtype=torch.bool,
                        device=fixed_target.device,
                    )
                    mobile_motifs = tuple(motif_mobility_controller.motifs)
                    uses_primary_axis = _scaffold_guidance_requires_primary_axis(
                        f.get("symmetry_id"),
                        tuple(str(motif.mobility_subspace) for motif in mobile_motifs),
                    )
                    if uses_primary_axis:
                        scaffold_guidance_axis = extract_symmetry_primary_axis(
                            f["sym_transform"],
                            symmetry_id=f.get("symmetry_id"),
                        )
                        scaffold_guidance_principal_axes = tuple(
                            principal_axis_from_points(
                                mobile_motif.template_master[
                                    0,
                                    is_ca[mobile_motif.master_atom_indices],
                                ]
                            )
                            for mobile_motif in mobile_motifs
                        )
                    else:
                        scaffold_guidance_axis = None
                        scaffold_guidance_principal_axes = tuple(
                            None for _ in mobile_motifs
                        )
                    scaffold_guidance_config = self._scaffold_guidance_config()
                    ranked_logger.info(
                        "Scaffold-derived motif guidance initialized: "
                        f"junctions={len(scaffold_guidance_topology.junction_pairs)}, "
                        "proposal_only="
                        f"{not self.motif_mobility_apply_updates}, "
                        "update_interval="
                        f"{effective_motif_mobility_update_interval} "
                        "(declared="
                        f"{self.motif_mobility_update_interval}, target="
                        f"{self.motif_mobility_target_update_count})"
                    )
                    # A monomer-core field must participate in the SE(3) pose
                    # gradient.  Keep the existing atomic graph transaction
                    # unchanged for legacy inter-only jobs; explicit intra
                    # jobs use the established scaffold transaction with an
                    # additional differentiable pose energy.
                    joint_packing_mobility = bool(
                        graph_interface_topology is not None
                        and scaffold_core_topology is None
                    )
                    if joint_packing_mobility:
                        ranked_logger.info(
                            "Unified packing-aware motif mobility enabled: "
                            "generated patches and all mobile orbit poses are "
                            "accepted or rolled back atomically"
                        )
            proposal_hook = None
            if motif_mobility_controller is not None:

                def proposal_hook(
                    proposal_coordinates: torch.Tensor,
                    progress: float,
                ) -> ConstraintProposalResult:
                    if self.motif_mobility_proposal_source == "scaffold_boundary":
                        if (
                            scaffold_guidance_topology is None
                            or scaffold_guidance_principal_axes is None
                            or scaffold_guidance_config is None
                        ):
                            raise RuntimeError("Scaffold guidance was not initialized")
                        if joint_packing_mobility:
                            if (
                                graph_interface_topology is None
                                or graph_interface_guidance_config is None
                                or graph_interface_patch_state is None
                                or constraint_runtime is None
                            ):
                                raise RuntimeError(
                                    "Unified packing mobility was not fully "
                                    "initialized"
                                )

                            def joint_projector(candidate: torch.Tensor):
                                return self._joint_projector(f).project(
                                    candidate,
                                    constraint_target=(constraint_runtime.fixed_target),
                                    constraint_mask=(is_motif_atom_with_fixed_coord),
                                    restore=True,
                                    label=("Unified packing mobility proposal"),
                                )

                            joint_update = motif_mobility_controller.update_orbits_with_interface_packing
                            (
                                target,
                                proposed_coordinates,
                                joint_diagnostics,
                            ) = joint_update(
                                proposal_coordinates,
                                f,
                                progress=progress,
                                topology=scaffold_guidance_topology,
                                axis=scaffold_guidance_axis,
                                principal_axes=(scaffold_guidance_principal_axes),
                                scaffold_config=scaffold_guidance_config,
                                interface_topology=(graph_interface_topology),
                                interface_config=(graph_interface_guidance_config),
                                patch_state=graph_interface_patch_state,
                                projector=joint_projector,
                                apply_update=bool(self.motif_mobility_apply_updates),
                                capture_response_scale=float(
                                    self.motif_mobility_capture_response_scale
                                ),
                                expand_response_scale=float(
                                    self.motif_mobility_expand_response_scale
                                ),
                                polish_response_scale=float(
                                    self.motif_mobility_polish_response_scale
                                ),
                            )
                            packing_step = dict(joint_diagnostics["packing_step"])
                            packing_step.update(
                                {
                                    "phase": "joint_packing_mobility",
                                    "progress": float(progress),
                                    "joint_transaction_accepted": (
                                        joint_diagnostics["accepted"]
                                    ),
                                    "joint_transaction_committed": (
                                        joint_diagnostics["committed"]
                                    ),
                                    "motif_pose_response_scale": (
                                        joint_diagnostics["motif_pose_response_scale"]
                                    ),
                                }
                            )
                            graph_interface_diagnostics.append(packing_step)
                            return ConstraintProposalResult(
                                target=target,
                                applied=bool(
                                    motif_mobility_controller.last_joint_transaction_applied
                                ),
                                coordinates=proposed_coordinates,
                            )
                        core_pose_energy = None
                        if scaffold_core_topology_required:
                            if scaffold_core_topology is None:
                                raise RuntimeError(
                                    "Scaffold core topology was not initialized"
                                )
                            if scaffold_core_guidance_config is None:
                                raise RuntimeError(
                                    "Scaffold core guidance config was not initialized"
                                )

                            def core_pose_energy(candidate_target):
                                generated_mask = (
                                    scaffold_core_topology.generated_atom_mask
                                )
                                # Candidate target contains the differentiable
                                # mobile seed pose; the generated scaffold must
                                # come from the current denoiser proposal.  A
                                # target-only energy would optimize against
                                # placeholder coordinates instead of the
                                # structure being sampled.
                                candidate_state = torch.where(
                                    generated_mask[:, None],
                                    proposal_coordinates[0],
                                    candidate_target,
                                )
                                core_total = scaffold_core_energy(
                                    candidate_state,
                                    scaffold_core_topology,
                                    scaffold_core_guidance_config,
                                ).total
                                if robust_capture_active:
                                    local_window = (
                                        float(progress)
                                        - float(self.motif_mobility_start_fraction)
                                    ) / max(
                                        float(self.motif_mobility_end_fraction)
                                        - float(self.motif_mobility_start_fraction),
                                        1e-8,
                                    )
                                    capture_fraction = float(
                                        self.motif_mobility_capture_fraction
                                    )
                                    if 0.0 <= local_window <= capture_fraction:
                                        capture_progress = min(
                                            1.0,
                                            max(0.0, local_window / capture_fraction),
                                        )
                                        capture_terms = [
                                            robust_interface_capture_energy(
                                                candidate_state,
                                                scaffold_core_topology,
                                                scaffold_core_guidance_config,
                                                motif.group_atom_indices,
                                                capture_progress=capture_progress,
                                            )
                                            for motif in motif_mobility_controller.motifs
                                        ]
                                        core_total = (
                                            core_total
                                            + float(
                                                self.supplied_interface_capture_weight
                                            )
                                            * torch.stack(capture_terms).mean()
                                        )
                                return core_total

                        scaffold_update_arguments = {
                            "progress": progress,
                            "topology": scaffold_guidance_topology,
                            "axis": scaffold_guidance_axis,
                            "principal_axes": scaffold_guidance_principal_axes,
                            "config": scaffold_guidance_config,
                            "apply_update": bool(self.motif_mobility_apply_updates),
                            # The engine reseeds torch once per Mosaic design.
                            # Reuse that independently replayable seed for
                            # early near-optimal pose selection; the controller
                            # derives distinct substreams by step and orbit.
                            "proposal_selection_seed": int(torch.initial_seed()),
                        }
                        # Preserve the exact legacy call signature when the
                        # new field is disabled.  Besides compatibility with
                        # external controllers, this proves that default-off
                        # intra guidance cannot alter established jobs.
                        if core_pose_energy is not None:
                            scaffold_update_arguments["pose_energy"] = core_pose_energy
                        target = motif_mobility_controller.update_orbits_from_scaffold(
                            proposal_coordinates,
                            **scaffold_update_arguments,
                        )
                    else:
                        target = motif_mobility_controller.update(
                            proposal_coordinates,
                            progress=progress,
                        )
                    return ConstraintProposalResult(
                        target=target,
                        applied=bool(motif_mobility_controller.last_update_applied),
                    )

            conditioning_synchronizer = None
            if motif_mobility_controller is not None:

                def conditioning_synchronizer(target):
                    return self._synchronize_mobile_motif_conditioning(
                        f,
                        target,
                        is_motif_atom_with_fixed_coord,
                    )

            constraint_runtime = MosaicConstraintRuntime(
                projector=self._joint_projector(f),
                fixed_target=fixed_target,
                fixed_mask=is_motif_atom_with_fixed_coord,
                cylindrical_projector=self._cylindrical_projector(
                    f,
                    fixed_target,
                ),
                proposal_source=self.motif_mobility_proposal_source,
                proposal_interval=effective_motif_mobility_update_interval,
                proposal_hook=proposal_hook,
                synchronize_conditioning=conditioning_synchronizer,
            )
            if motif_mobility_controller is not None:
                constraint_runtime.synchronize_initial_conditioning()
            fixed_target = constraint_runtime.fixed_target
            X_L = self._get_exact_symmetric_initial_structure(
                c0=noise_schedule[0],
                D=D,
                fixed_target=fixed_target,
                fixed_mask=is_motif_atom_with_fixed_coord,
                f=f,
                constraint_runtime=constraint_runtime,
            )
        else:
            self._exact_symmetry_orbit_layout = None
            X_L = self._get_initial_structure(
                c0=noise_schedule[0],
                D=D,
                L=L,
                coord_atom_lvl_to_be_noised=fixed_target,
                is_motif_atom_with_fixed_coord=(is_motif_atom_with_fixed_coord),
            )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        # symmetrize X_L until the step gamma = gamma_min_sym
        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            if self.symmetry_noise_mode == "coupled":
                epsilon_L = expand_symmetry_coupled_displacements(
                    epsilon_L,
                    self._symmetry_features(f),
                    layout=self._exact_symmetry_orbit_layout,
                )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )

            X_noisy_L = X_L + epsilon_L
            self._assert_symmetry_orbit_closed(
                X_noisy_L,
                f,
                label=f"Noisy diffusion state at step {step_num}",
            )

            # Denoise either the complete assembly or the bounded local view.
            denoiser_X_noisy_L = X_noisy_L
            if local_symmetry_context is not None:
                denoiser_X_noisy_L = X_noisy_L[
                    :,
                    local_symmetry_context.neighbourhood.atom_indices,
                    :,
                ]
            # Handle chunked mode vs standard mode (same as default sampler)
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=denoiser_X_noisy_L,
                    t=t_hat.tile(D),
                    f=denoiser_f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=denoiser_X_noisy_L,
                    t=t_hat.tile(D),
                    f=denoiser_f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )
            if local_symmetry_context is not None:
                if "X_L" not in outs:
                    raise ValueError(
                        "local_neighbourhood requires dictionary model output "
                        "with X_L"
                    )
                outs["X_L"] = expand_local_prediction_to_full_orbit(
                    outs["X_L"],
                    X_noisy_L,
                    local_symmetry_context.neighbourhood,
                    layout=local_symmetry_context.layout,
                )
                if exists(outs.get("sequence_logits_I")):
                    outs["sequence_logits_I"] = (
                        expand_local_token_prediction_to_full_orbit(
                            outs["sequence_logits_I"],
                            f,
                            local_symmetry_context.feature_view,
                        )
                    )
                    outs["sequence_indices_I"] = diffusion_module.sequence_head.decode(
                        outs["sequence_logits_I"]
                    )
            if "X_L" in outs and constraint_runtime is not None:
                outs["X_L"] = constraint_runtime.process_model_prediction(
                    outs["X_L"],
                    step_num=step_num,
                    total_steps=len(noise_schedule) - 1,
                )
                fixed_target = constraint_runtime.fixed_target
            elif "X_L" in outs and c_t > gamma_min_sym:
                outs["X_L"] = self._apply_symmetry_preserving_fixed_motif(
                    outs["X_L"],
                    f,
                    is_motif_atom_with_fixed_coord,
                    fixed_coordinates=X_noisy_L,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            # NOTE: no classifier-free guidance for symmetry

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L
            # Independent noise in X_noisy_L means this Euler update is not
            # guaranteed to be symmetric even when X_denoised_L was
            # projected. Project the actual state that advances to the next
            # denoising step, then restore the complete interface groups.
            if constraint_runtime is not None:
                X_L = constraint_runtime.project_state_update(
                    X_L,
                    step_num=step_num,
                )
            else:
                X_L = self._project_stepwise_updated_coordinates(
                    X_L,
                    f,
                    is_motif_atom_with_fixed_coord,
                    X_noisy_L,
                )
            if self.interface_seed_compactness_weight > 0.0:
                X_L = self._apply_interface_seed_compactness(
                    X_L,
                    f,
                    is_motif_atom_with_fixed_coord,
                    step_num=step_num,
                    num_steps=len(noise_schedule) - 1,
                )
                # Token-level translations are computed independently for
                # each chain.  Re-project afterwards to prevent accumulated
                # numerical or mask-induced deviations from native symmetry.
                if constraint_runtime is not None:
                    X_L = constraint_runtime.project_post_guidance(
                        X_L,
                        step_num=step_num,
                    )
                else:
                    X_L = self._apply_symmetry_preserving_fixed_motif(
                        X_L,
                        f,
                        is_motif_atom_with_fixed_coord,
                    )
            if graph_interface_topology is not None and not joint_packing_mobility:
                if graph_interface_guidance_config is None:
                    raise RuntimeError(
                        "Graph interface guidance config was not initialized"
                    )
                if constraint_runtime is not None:

                    def graph_projector(
                        candidate,
                        *,
                        active_step_num=step_num,
                    ):
                        return constraint_runtime.project_post_guidance(
                            candidate,
                            step_num=active_step_num,
                        )
                else:

                    def graph_projector(
                        candidate,
                        *,
                        active_fixed_target=fixed_target,
                    ):
                        return self._project_stepwise_updated_coordinates(
                            candidate,
                            f,
                            is_motif_atom_with_fixed_coord,
                            active_fixed_target,
                        )

                if graph_interface_patch_state is None:
                    raise RuntimeError(
                        "Graph interface patch state was not initialized"
                    )
                progress = step_num / max(len(noise_schedule) - 2, 1)
                X_L, interface_step = apply_graph_interface_guidance(
                    X_L,
                    f,
                    graph_interface_topology,
                    progress=progress,
                    config=graph_interface_guidance_config,
                    projector=graph_projector,
                    patch_state=graph_interface_patch_state,
                )
                interface_step["step_num"] = step_num
                graph_interface_diagnostics.append(interface_step)

            if scaffold_core_active:
                if scaffold_core_topology is None:
                    raise RuntimeError("Scaffold core topology was not initialized")
                if scaffold_core_guidance_config is None:
                    raise RuntimeError(
                        "Scaffold core guidance config was not initialized"
                    )
                if constraint_runtime is not None:

                    def core_projector(
                        candidate,
                        *,
                        active_step_num=step_num,
                    ):
                        return constraint_runtime.project_post_guidance(
                            candidate,
                            step_num=active_step_num,
                        )
                else:

                    def core_projector(
                        candidate,
                        *,
                        active_fixed_target=fixed_target,
                    ):
                        return self._project_stepwise_updated_coordinates(
                            candidate,
                            f,
                            is_motif_atom_with_fixed_coord,
                            active_fixed_target,
                        )

                progress = step_num / max(len(noise_schedule) - 2, 1)
                X_L, core_step = apply_scaffold_core_guidance(
                    X_L,
                    scaffold_core_topology,
                    progress=progress,
                    config=scaffold_core_guidance_config,
                    projector=core_projector,
                )
                core_step["step_num"] = step_num
                scaffold_core_diagnostics.append(core_step)

            if polymer_continuity_active:
                if scaffold_core_topology is None:
                    raise RuntimeError(
                        "Polymer continuity topology was not initialized"
                    )
                if constraint_runtime is not None:

                    def continuity_projector(
                        candidate,
                        *,
                        active_step_num=step_num,
                    ):
                        return constraint_runtime.project_post_guidance(
                            candidate,
                            step_num=active_step_num,
                        )
                else:

                    def continuity_projector(
                        candidate,
                        *,
                        active_fixed_target=fixed_target,
                    ):
                        return self._project_stepwise_updated_coordinates(
                            candidate,
                            f,
                            is_motif_atom_with_fixed_coord,
                            active_fixed_target,
                        )

                X_L, continuity_step = project_generated_polymer_continuity(
                    X_L,
                    scaffold_core_topology,
                    target_ca_distance=float(
                        self.generated_polymer_continuity_target_ca_distance
                    ),
                    tolerance=float(self.generated_polymer_continuity_tolerance),
                    iterations=int(self.generated_polymer_continuity_iterations),
                    projector=continuity_projector,
                )
                continuity_step["step_num"] = step_num
                polymer_continuity_diagnostics.append(continuity_step)

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        final_graph_interface_energy = None
        final_graph_interface_quality_satisfied = None
        if graph_interface_topology is not None and scaffold_core_topology is None:
            if graph_interface_guidance_config is None:
                raise RuntimeError(
                    "Graph interface guidance config was not initialized"
                )
            # If diffusion never formed a patch strong enough for a
            # quality-triggered identity lock, choose the best reciprocal
            # contiguous window in the actual final denoised state.  The
            # deterministic polish then improves that one physical patch
            # instead of hopping between sequence windows.
            if (
                graph_interface_patch_state is not None
                and not graph_interface_patch_state.locked
            ):
                graph_interface_patch_state.assignments = (
                    resolve_graph_interface_patch_assignments(
                        X_L,
                        graph_interface_topology,
                        graph_interface_guidance_config,
                    )
                )
                graph_interface_patch_state.locked = True
                graph_interface_patch_state.lock_reason = "final_polish"
            # A bounded deterministic polish closes the gap between a useful
            # interface seen mid-trajectory and the structure that is
            # actually written.  Every correction is followed by the same
            # exact joint projector used in the diffusion loop, so fixed
            # motifs and group symmetry remain authoritative.
            for polish_index in range(
                graph_interface_guidance_config.final_polish_steps
            ):
                current_energy = graph_interface_energy(
                    X_L,
                    graph_interface_topology,
                    graph_interface_guidance_config,
                    patch_assignments=(
                        graph_interface_patch_state.assignments
                        if graph_interface_patch_state is not None
                        else None
                    ),
                )
                if graph_interface_quality_satisfied(
                    current_energy,
                    clash_ca_distance=(
                        graph_interface_guidance_config.clash_ca_distance
                    ),
                    config=graph_interface_guidance_config,
                ):
                    break
                polish_step_num = len(noise_schedule) - 1 + polish_index
                if constraint_runtime is not None:

                    def graph_projector(
                        candidate,
                        *,
                        active_step_num=polish_step_num,
                    ):
                        return constraint_runtime.project_post_guidance(
                            candidate,
                            step_num=active_step_num,
                        )
                else:

                    def graph_projector(
                        candidate,
                        *,
                        active_fixed_target=fixed_target,
                    ):
                        return self._project_stepwise_updated_coordinates(
                            candidate,
                            f,
                            is_motif_atom_with_fixed_coord,
                            active_fixed_target,
                        )

                X_L, interface_step = apply_graph_interface_guidance(
                    X_L,
                    f,
                    graph_interface_topology,
                    progress=1.0,
                    config=graph_interface_guidance_config,
                    projector=graph_projector,
                    patch_state=graph_interface_patch_state,
                )
                interface_step.update(
                    {
                        "phase": "final_polish",
                        "polish_index": polish_index,
                        "step_num": polish_step_num,
                    }
                )
                graph_interface_diagnostics.append(interface_step)

        if polymer_continuity_active:
            if scaffold_core_topology is None:
                raise RuntimeError("Polymer continuity topology was not initialized")
            if constraint_runtime is not None:

                def final_continuity_projector(candidate):
                    return constraint_runtime.project_post_guidance(
                        candidate,
                        step_num=max(len(noise_schedule) - 2, 0),
                    )
            else:

                def final_continuity_projector(candidate):
                    return self._project_stepwise_updated_coordinates(
                        candidate,
                        f,
                        is_motif_atom_with_fixed_coord,
                        fixed_target,
                    )

            X_L, final_continuity_step = project_generated_polymer_continuity(
                X_L,
                scaffold_core_topology,
                target_ca_distance=float(
                    self.generated_polymer_continuity_target_ca_distance
                ),
                tolerance=float(self.generated_polymer_continuity_tolerance),
                iterations=int(self.generated_polymer_continuity_iterations),
                projector=final_continuity_projector,
            )
            final_continuity_step["phase"] = "final_projection"
            polymer_continuity_diagnostics.append(final_continuity_step)

        if constraint_runtime is not None:
            X_L = constraint_runtime.finalize(X_L)
        elif torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            X_L = self._finalize_with_fixed_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                f,
            )

        if graph_interface_topology is not None:
            if graph_interface_guidance_config is None:
                raise RuntimeError(
                    "Graph interface guidance config was not initialized"
                )
            final_graph_interface_energy = graph_interface_energy(
                X_L,
                graph_interface_topology,
                replace(
                    graph_interface_guidance_config,
                    contact_prior_weight=0.0,
                ),
                patch_assignments=(
                    graph_interface_patch_state.assignments
                    if graph_interface_patch_state is not None
                    else None
                ),
            )
            final_graph_interface_quality_satisfied = graph_interface_quality_satisfied(
                final_graph_interface_energy,
                clash_ca_distance=(graph_interface_guidance_config.clash_ca_distance),
                config=graph_interface_guidance_config,
            )

        final_scaffold_core_energy = None
        if scaffold_core_active:
            if scaffold_core_topology is None:
                raise RuntimeError("Scaffold core topology was not initialized")
            if scaffold_core_guidance_config is None:
                raise RuntimeError("Scaffold core guidance config was not initialized")
            final_scaffold_core_energy = scaffold_core_energy(
                X_L[0],
                scaffold_core_topology,
                scaffold_core_guidance_config,
            )

        result = dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )
        if constraint_runtime is not None:
            result["constraint_runtime_diagnostics"] = constraint_runtime.diagnostics()
        if motif_mobility_controller is not None:
            if constraint_runtime is None:
                raise RuntimeError(
                    "Motif mobility completed without a constraint runtime"
                )
            mobility_diagnostics = motif_mobility_controller.diagnostics()
            mobility_diagnostics["symmetry_id"] = str(f.get("symmetry_id") or "")
            mobility_diagnostics["runtime_group_action_count"] = len(
                f.get("sym_transform") or {}
            )
            mobility_diagnostics["conditioning_refresh_count"] = (
                constraint_runtime.conditioning_refresh_count
            )
            mobility_diagnostics["constraint_runtime"] = result[
                "constraint_runtime_diagnostics"
            ]
            mobility_diagnostics["mobile_orbit_count"] = mobile_orbit_count
            mobility_diagnostics["proposal_source"] = (
                self.motif_mobility_proposal_source
            )
            mobility_diagnostics["apply_updates"] = bool(
                self.motif_mobility_apply_updates
            )
            mobility_diagnostics["update_interval"] = (
                effective_motif_mobility_update_interval
            )
            if mobility_schedule_diagnostics is not None:
                mobility_diagnostics["proposal_schedule"] = dict(
                    mobility_schedule_diagnostics
                )
            if scaffold_guidance_config is not None:
                mobility_diagnostics["scaffold_guidance_config"] = {
                    key: value for key, value in vars(scaffold_guidance_config).items()
                }
            axis_point = getattr(scaffold_guidance_axis, "point", None)
            axis_direction = getattr(
                scaffold_guidance_axis,
                "direction",
                None,
            )
            axis_transform_ids = getattr(
                scaffold_guidance_axis,
                "transform_ids",
                None,
            )
            if (
                isinstance(axis_point, torch.Tensor)
                and axis_point.numel() == 3
                and torch.isfinite(axis_point).all()
                and isinstance(axis_direction, torch.Tensor)
                and axis_direction.numel() == 3
                and torch.isfinite(axis_direction).all()
                and isinstance(axis_transform_ids, (list, tuple))
            ):
                mobility_diagnostics["symmetry_axis"] = {
                    "point": [
                        float(value)
                        for value in axis_point.detach().cpu().reshape(-1).tolist()
                    ],
                    "direction": [
                        float(value)
                        for value in axis_direction.detach().cpu().reshape(-1).tolist()
                    ],
                    "transform_ids": [int(value) for value in axis_transform_ids],
                }
            result["motif_mobility_diagnostics"] = mobility_diagnostics
            final_orbits = mobility_diagnostics["orbits"]
            maximum_translation = max(
                (
                    max(orbit["translation_norms"], default=0.0)
                    for orbit in final_orbits
                ),
                default=0.0,
            )
            maximum_rotation = max(
                (max(orbit["rotation_degrees"], default=0.0) for orbit in final_orbits),
                default=0.0,
            )
            ranked_logger.info(
                "Orbit-rigid motif mobility completed: "
                f"orbits={mobile_orbit_count}, "
                f"updates={mobility_diagnostics['update_calls']}, "
                "active_window_updates="
                f"{mobility_diagnostics['active_window_calls']}, "
                "conditioning_refreshes="
                f"{constraint_runtime.conditioning_refresh_count}, "
                f"max_translation={maximum_translation:.6f} A, "
                f"max_rotation={maximum_rotation:.6f} deg"
            )
        if graph_interface_topology is not None:
            if final_graph_interface_energy is None:
                raise RuntimeError("Final graph interface energy was not evaluated")
            final_proxy = graph_interface_energy_diagnostics(
                final_graph_interface_energy
            )
            result["graph_interface_guidance_diagnostics"] = {
                "schema_version": 9,
                "runtime_active": True,
                "edge_count": len(graph_interface_topology.edges),
                "edge_ids": [edge.edge_id for edge in graph_interface_topology.edges],
                "source_interface_ids": [
                    edge.source_interface_id for edge in graph_interface_topology.edges
                ],
                "capacity_preflight": list(graph_interface_topology.capacity_preflight),
                "config": vars(graph_interface_guidance_config),
                "steps": graph_interface_diagnostics,
                "applied_steps": sum(
                    bool(step.get("applied")) for step in graph_interface_diagnostics
                ),
                "final_polish_steps": sum(
                    step.get("phase") == "final_polish"
                    for step in graph_interface_diagnostics
                ),
                "final_proxy_targets_satisfied": (
                    final_graph_interface_quality_satisfied
                ),
                "final_proxy": final_proxy,
            }
        if scaffold_core_active:
            if scaffold_core_topology is None:
                raise RuntimeError("Scaffold core topology was not initialized")
            if (
                scaffold_core_guidance_config is None
                or final_scaffold_core_energy is None
            ):
                raise RuntimeError("Final scaffold core energy was not evaluated")
            result["scaffold_core_guidance_diagnostics"] = {
                "schema_version": 1,
                "runtime_active": True,
                "chain_count": len(scaffold_core_topology.chains),
                "config": vars(scaffold_core_guidance_config),
                "steps": scaffold_core_diagnostics,
                "applied_steps": sum(
                    bool(step.get("applied")) for step in scaffold_core_diagnostics
                ),
                "final_metrics": final_scaffold_core_energy.detached_dict(),
            }
        if polymer_continuity_active:
            result["generated_polymer_continuity_diagnostics"] = {
                "schema_version": 1,
                "runtime_active": True,
                "target_ca_distance": float(
                    self.generated_polymer_continuity_target_ca_distance
                ),
                "tolerance": float(self.generated_polymer_continuity_tolerance),
                "configured_iterations": int(
                    self.generated_polymer_continuity_iterations
                ),
                "steps": polymer_continuity_diagnostics,
                "all_steps_within_tolerance": all(
                    bool(step["within_tolerance"])
                    for step in polymer_continuity_diagnostics
                ),
            }
        return result


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        if kind == "default":
            unsupported = []
            if kwargs.get("symmetry_state_mode", "legacy_asu") != "legacy_asu":
                unsupported.append("symmetry_state_mode")
            if kwargs.get("symmetry_noise_mode", "independent") != "independent":
                unsupported.append("symmetry_noise_mode")
            if (
                kwargs.get(
                    "symmetry_execution_backend",
                    "explicit_all_copy",
                )
                != "explicit_all_copy"
            ):
                unsupported.append("symmetry_execution_backend")
            for flag in (
                "preserve_fixed_motif_during_symmetry",
                "require_motif_constraint_groups",
                "enable_orbit_rigid_motif_mobility",
                "enable_graph_interface_guidance",
                "enable_symmetric_scaffold_packing",
                "enable_generated_polymer_continuity_guidance",
                "enable_supplied_interface_robust_capture",
            ):
                if kwargs.get(flag, False):
                    unsupported.append(flag)
            if float(kwargs.get("scaffold_core_intra_chain_weight", 0.0)) > 0.0:
                unsupported.append("scaffold_core_intra_chain_weight")
            if float(kwargs.get("scaffold_core_inter_chain_weight", 1.0)) != 1.0:
                unsupported.append("scaffold_core_inter_chain_weight")
            if (
                float(
                    kwargs.get(
                        "scaffold_core_inter_chain_excess_penalty",
                        0.0,
                    )
                )
                > 0.0
            ):
                unsupported.append("scaffold_core_inter_chain_excess_penalty")
            if float(kwargs.get("interface_seed_compactness_weight", 0.0)) > 0.0:
                unsupported.append("interface_seed_compactness_weight")
            if unsupported:
                raise ValueError(
                    "Symmetry-only inference options require "
                    "inference_sampler.kind=symmetry: " + ", ".join(sorted(unsupported))
                )
        try:
            SamplerCls = self._registry[kind]
            # remove kwargs that the sampler cannot take
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names


def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_coord: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    # ... Centering
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of motif atoms
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of diffused atoms

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    # ... Center
    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R
