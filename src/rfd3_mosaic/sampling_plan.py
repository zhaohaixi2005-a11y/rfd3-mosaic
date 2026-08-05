"""Compile public sampling declarations into backend-neutral runtime intent."""

from __future__ import annotations

from pydantic import Field

from rfd3_mosaic.schema.design import (
    UserDesignSpec,
    UserFixedOrientationSpec,
)
from rfd3_mosaic.schema.specs import StrictModel


class DiffusionSamplingPlan(StrictModel):
    """Randomness and execution choices inside the RFD3 timestep loop."""

    timesteps: int
    seed: int
    preset: str
    low_memory_mode: bool
    execution_backend: str
    neighbour_radius: int


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


class SamplingPlan(StrictModel):
    """Deterministic separation of initial-pose and diffusion sampling."""

    schema_version: int = 1
    initial_pose: StaticPosePlan | None = None
    diffusion: DiffusionSamplingPlan
    runtime_mobility: tuple[dict[str, object], ...] = Field(
        default_factory=tuple
    )


def compile_sampling_plan(design: UserDesignSpec) -> SamplingPlan:
    """Compile a public declaration without sampling any coordinates.

    The plan records ranges and seeds.  Coordinate realization remains the
    assembly compiler's responsibility, so planning is side-effect free and
    reproducible.
    """

    sampling = design.sampling
    initial = sampling.initial_pose
    pose_plan: StaticPosePlan | None = None
    if initial is not None:
        fixed = isinstance(initial.orientation, UserFixedOrientationSpec)
        pose_plan = StaticPosePlan(
            seed=initial.seed,
            radius_minimum=initial.radius.minimum,
            radius_maximum=initial.radius.maximum,
            axial_minimum=initial.axial_offset.minimum,
            axial_maximum=initial.axial_offset.maximum,
            radial_direction=initial.radial_direction,
            orientation_method=initial.orientation.method,
            rotation_deg=(
                initial.orientation.rotation_deg if fixed else None
            ),
        )
    return SamplingPlan(
        initial_pose=pose_plan,
        diffusion=DiffusionSamplingPlan(
            timesteps=sampling.timesteps,
            seed=sampling.seed,
            preset=sampling.preset,
            low_memory_mode=sampling.low_memory_mode,
            execution_backend=sampling.execution_backend,
            neighbour_radius=sampling.neighbour_radius,
        ),
    )


def assembly_initialization_payload(
    plan: SamplingPlan,
) -> tuple[int | None, dict[str, object]]:
    """Lower one static pose plan into the existing assembly IR fields."""

    pose = plan.initial_pose
    if pose is None:
        return None, {}

    def interval(minimum: float, maximum: float) -> dict[str, float]:
        return {
            "mean": (minimum + maximum) / 2.0,
            "range": (maximum - minimum) / 2.0,
        }

    orientation: dict[str, object] = {"method": pose.orientation_method}
    if pose.rotation_deg is not None:
        orientation["rotation_deg"] = pose.rotation_deg
    return pose.seed, {
        pose.group_id: {
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
    }


__all__ = [
    "DiffusionSamplingPlan",
    "SamplingPlan",
    "StaticPosePlan",
    "assembly_initialization_payload",
    "compile_sampling_plan",
]
