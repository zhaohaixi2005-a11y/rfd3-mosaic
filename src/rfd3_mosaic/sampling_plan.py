"""Compile public sampling declarations into backend-neutral runtime intent."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from pydantic import Field

from rfd3_mosaic.schema.design import (
    UserDesignSpec,
    UserFixedOrientationSpec,
)
from rfd3_mosaic.schema.specs import StrictModel


class DiffusionSamplingPlan(StrictModel):
    """Randomness and execution choices inside the RFD3 timestep loop."""

    timesteps: int
    designs: int
    replicates_per_pose: int
    seed: int
    preset: str
    low_memory_mode: bool
    execution_backend: str
    neighbour_radius: int
    scaffold_packing: str
    screening_mode: str
    screening_protocol: str
    retain_all_outputs: bool


class StaticPosePlan(StrictModel):
    """One rigid pose chosen before diffusion begins."""

    group_id: str = "motif_group"
    seed: int
    radius_minimum: float
    radius_maximum: float
    axial_minimum: float
    axial_maximum: float
    radial_direction: tuple[float, float, float]
    orientation_method: str
    rotation_deg: tuple[float, float, float] | None = None
    maximum_tilt_deg: float | None = None


class SamplingPlan(StrictModel):
    """Deterministic separation of initial-pose and diffusion sampling."""

    schema_version: int = 1
    initial_pose: StaticPosePlan | None = None
    component_initial_poses: tuple[StaticPosePlan, ...] = Field(default_factory=tuple)
    diffusion: DiffusionSamplingPlan
    runtime_mobility: tuple[dict[str, object], ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class DesignSamplingAssignment:
    """One reproducible assembly-pose and diffusion pairing."""

    design_index: int
    pose_index: int
    replicate_index: int
    pose_seed: int | None
    diffusion_seed: int


def pose_plan_is_stochastic(plan: SamplingPlan) -> bool:
    """Return whether recompiling with another seed can change coordinates."""

    poses = (
        (plan.initial_pose,)
        if plan.initial_pose is not None
        else plan.component_initial_poses
    )
    return any(
        pose.orientation_method in {"uniform_so3", "principal_axis_cone"}
        or pose.radius_minimum != pose.radius_maximum
        or pose.axial_minimum != pose.axial_maximum
        for pose in poses
    )


def design_sampling_assignments(
    plan: SamplingPlan,
) -> tuple[DesignSamplingAssignment, ...]:
    """Expand ``designs`` without confusing pose and diffusion randomness.

    A pose-capable design receives one independently seeded assembly pose per
    ``replicates_per_pose`` outputs.  A fixed design retains exactly one pose
    and varies only the diffusion trajectory.  Sequential seed derivation is
    intentional: it is transparent to users, deterministic, and exactly
    replayable without storing hidden RNG state.
    """

    diffusion = plan.diffusion
    stochastic_pose = pose_plan_is_stochastic(plan)
    replicas = diffusion.replicates_per_pose if stochastic_pose else diffusion.designs
    pose_count = ceil(diffusion.designs / replicas)
    component_seeded_pose = False
    if plan.initial_pose is not None:
        base_pose_seed = plan.initial_pose.seed
    elif plan.component_initial_poses:
        component_seeded_pose = True
        base_pose_seed = min(pose.seed for pose in plan.component_initial_poses)
    else:
        base_pose_seed = None

    assignments: list[DesignSamplingAssignment] = []
    for design_index in range(diffusion.designs):
        pose_index = design_index // replicas if stochastic_pose else 0
        assignments.append(
            DesignSamplingAssignment(
                design_index=design_index,
                pose_index=pose_index,
                replicate_index=design_index % replicas,
                pose_seed=(
                    None
                    if component_seeded_pose and pose_index == 0
                    else base_pose_seed + pose_index
                    if stochastic_pose and base_pose_seed is not None
                    else None
                ),
                diffusion_seed=diffusion.seed + design_index,
            )
        )
    assert pose_count == len({item.pose_index for item in assignments})
    return tuple(assignments)


def compile_sampling_plan(design: UserDesignSpec) -> SamplingPlan:
    """Compile a public declaration without sampling any coordinates.

    The plan records ranges and seeds.  Coordinate realization remains the
    assembly compiler's responsibility, so planning is side-effect free and
    reproducible.
    """

    sampling = design.sampling

    def compile_pose(
        initial,
        *,
        group_id: str,
    ) -> StaticPosePlan:
        fixed = isinstance(initial.orientation, UserFixedOrientationSpec)
        return StaticPosePlan(
            group_id=group_id,
            seed=initial.seed,
            radius_minimum=initial.radius.minimum,
            radius_maximum=initial.radius.maximum,
            axial_minimum=initial.axial_offset.minimum,
            axial_maximum=initial.axial_offset.maximum,
            radial_direction=initial.radial_direction,
            orientation_method=initial.orientation.method,
            rotation_deg=(initial.orientation.rotation_deg if fixed else None),
            maximum_tilt_deg=getattr(
                initial.orientation,
                "maximum_tilt_deg",
                None,
            ),
        )

    initial = sampling.initial_pose
    pose_plan = (
        compile_pose(initial, group_id="motif_group") if initial is not None else None
    )
    component_pose_plans = tuple(
        compile_pose(pose, group_id=component_id)
        for component_id, pose in sampling.initial_poses.items()
    )
    return SamplingPlan(
        initial_pose=pose_plan,
        component_initial_poses=component_pose_plans,
        diffusion=DiffusionSamplingPlan(
            timesteps=sampling.timesteps,
            designs=sampling.designs,
            replicates_per_pose=sampling.replicates_per_pose,
            seed=sampling.seed,
            preset=sampling.preset,
            low_memory_mode=sampling.low_memory_mode,
            execution_backend=sampling.execution_backend,
            neighbour_radius=sampling.neighbour_radius,
            scaffold_packing=sampling.scaffold_packing,
            screening_mode=sampling.screening.mode,
            screening_protocol=sampling.screening.protocol,
            retain_all_outputs=sampling.screening.retain_all_outputs,
        ),
    )


def assembly_initialization_payload(
    plan: SamplingPlan,
) -> tuple[int | None, dict[str, object]]:
    """Lower static pose plans into the existing assembly IR fields."""

    poses = (
        (plan.initial_pose,)
        if plan.initial_pose is not None
        else plan.component_initial_poses
    )
    if not poses:
        return None, {}

    def interval(minimum: float, maximum: float) -> dict[str, float]:
        return {
            "mean": (minimum + maximum) / 2.0,
            "range": (maximum - minimum) / 2.0,
        }

    def pose_payload(
        pose: StaticPosePlan,
        *,
        include_seed: bool,
    ) -> dict[str, object]:
        orientation: dict[str, object] = {"method": pose.orientation_method}
        if pose.rotation_deg is not None:
            orientation["rotation_deg"] = pose.rotation_deg
        if pose.maximum_tilt_deg is not None:
            orientation["maximum_tilt_deg"] = pose.maximum_tilt_deg
        payload: dict[str, object] = {
            "center_method": "interface_heavy_atom_com",
            "orientation": orientation,
            "placement": {
                "radius": interval(
                    pose.radius_minimum,
                    pose.radius_maximum,
                ),
                "axial_offset": interval(
                    pose.axial_minimum,
                    pose.axial_maximum,
                ),
                "radial_direction": pose.radial_direction,
            },
        }
        if include_seed:
            payload["random_seed"] = pose.seed
        return payload

    component_mode = plan.initial_pose is None
    return (
        None if component_mode else poses[0].seed,
        {
            pose.group_id: pose_payload(
                pose,
                include_seed=component_mode,
            )
            for pose in poses
        },
    )


__all__ = [
    "DesignSamplingAssignment",
    "DiffusionSamplingPlan",
    "SamplingPlan",
    "StaticPosePlan",
    "assembly_initialization_payload",
    "compile_sampling_plan",
    "design_sampling_assignments",
    "pose_plan_is_stochastic",
]
