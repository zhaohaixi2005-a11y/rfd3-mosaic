"""Compile physical user preferences into auditable runtime settings.

The public surface deliberately contains a handful of qualitative choices.
This module is the only place that translates them into numerical sampler
settings, so a preset can be calibrated without changing user designs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rfd3_mosaic.schema.design import (
    CavityPreference,
    ComponentMotionPreference,
    DiversityPreference,
    FixedArrangementPolicy,
    InterfaceAreaPreference,
    PackingPreference,
    UserDesignSpec,
)
from rfd3_mosaic.schema.specs import StrictModel


class ResolvedDiversityPlan(StrictModel):
    global_pose_samples: int
    scope: Literal["independent_seed_pose_search"] = "independent_seed_pose_search"


class ResolvedDesignPreferences(StrictModel):
    """Frozen values consumed by search, sampler, provenance and reports."""

    schema_version: Literal[1] = 1
    preset_version: Literal[
        "packing_preferences_v1",
        "packing_preferences_v2",
    ] = "packing_preferences_v2"
    packing: PackingPreference
    cavity: CavityPreference
    diversity: DiversityPreference
    interface_area: InterfaceAreaPreference
    component_motion: ComponentMotionPreference
    mobility_subspace: str | None
    initial_radius_scale: float
    diversity_plan: ResolvedDiversityPlan
    sampler_overrides: dict[str, float | int] = Field(default_factory=dict)
    hard_contracts: tuple[str, ...] = (
        "exact_fixed_geometry",
        "exact_symmetry",
        "chain_continuity",
        "global_clash_rejection",
    )

    def hydra_overrides(self) -> tuple[str, ...]:
        return tuple(
            f"++inference_sampler.{name}={value}"
            for name, value in sorted(self.sampler_overrides.items())
        )


_PACKING: dict[PackingPreference, dict[str, float]] = {
    PackingPreference.LOOSE: {
        "graph_interface_guidance_contact_prior_weight": 0.06,
        "graph_interface_guidance_weight": 0.80,
        "graph_interface_guidance_coverage_weight": 0.80,
        "graph_interface_guidance_continuity_weight": 0.85,
        "graph_interface_guidance_orientation_weight": 0.20,
        "graph_interface_guidance_shape_weight": 0.35,
        "graph_interface_guidance_interface_balance_weight": 0.40,
        "graph_interface_guidance_distance_weight": 0.20,
        "graph_interface_guidance_maximum_token_step": 0.20,
    },
    PackingPreference.BALANCED: {
        "graph_interface_guidance_contact_prior_weight": 0.10,
        "graph_interface_guidance_weight": 1.00,
        "graph_interface_guidance_coverage_weight": 1.00,
        "graph_interface_guidance_continuity_weight": 1.00,
        "graph_interface_guidance_orientation_weight": 0.25,
        "graph_interface_guidance_shape_weight": 0.50,
        "graph_interface_guidance_interface_balance_weight": 0.50,
        "graph_interface_guidance_distance_weight": 0.25,
        "graph_interface_guidance_maximum_token_step": 0.25,
    },
    PackingPreference.TIGHT: {
        "graph_interface_guidance_contact_prior_weight": 0.15,
        "graph_interface_guidance_weight": 1.15,
        "graph_interface_guidance_coverage_weight": 1.25,
        "graph_interface_guidance_continuity_weight": 1.30,
        "graph_interface_guidance_orientation_weight": 0.40,
        "graph_interface_guidance_shape_weight": 0.80,
        "graph_interface_guidance_interface_balance_weight": 0.70,
        "graph_interface_guidance_distance_weight": 0.35,
        "graph_interface_guidance_maximum_token_step": 0.20,
    },
}

_AREA: dict[InterfaceAreaPreference, tuple[int, float, float]] = {
    InterfaceAreaPreference.SMALL: (4, 0.85, 0.90),
    InterfaceAreaPreference.AUTO: (8, 1.00, 1.00),
    InterfaceAreaPreference.LARGE: (12, 1.25, 1.15),
}

_RADIUS_SCALE = {
    CavityPreference.COMPACT: 0.88,
    CavityPreference.AUTO: 1.00,
    CavityPreference.OPEN: 1.12,
}

_DIVERSITY = {
    DiversityPreference.LOW: ResolvedDiversityPlan(
        global_pose_samples=4,
    ),
    DiversityPreference.MEDIUM: ResolvedDiversityPlan(
        global_pose_samples=8,
    ),
    DiversityPreference.HIGH: ResolvedDiversityPlan(
        global_pose_samples=16,
    ),
}


def compile_design_preferences(
    design: UserDesignSpec,
) -> ResolvedDesignPreferences:
    """Resolve one public design without weakening any safety threshold."""

    preferences = design.preferences
    motion = preferences.component_motion
    if motion is None:
        mobile_subspaces = {
            component.pose.subspace or "bounded_se3"
            for component in design.components.values()
            if component.pose.mode == "bounded_mobile"
        }
        # Legacy/simple fixed_xyz declarations may explicitly make one whole
        # supplied seed orbit mobile without declaring expert components.
        # Treat that as the same physical user intent for reporting and
        # sampler preset resolution; the fixed atoms remain one rigid body.
        mobile_subspaces.update(
            str(constraint.pose.subspace or "bounded_se3")
            for constraint in design.constraints
            if getattr(constraint, "kind", None) == "fixed_xyz"
            and getattr(constraint, "pose", None) is not None
            and constraint.pose.mode == "bounded_mobile"
        )
        if "bounded_se3" in mobile_subspaces:
            motion = ComponentMotionPreference.FREE
        elif mobile_subspaces:
            motion = ComponentMotionPreference.GUIDED
        else:
            motion = (
                ComponentMotionPreference.GUIDED
                if design.fixed_arrangement
                == FixedArrangementPolicy.OPTIMIZE_COMPONENTS
                else ComponentMotionPreference.LOCKED
            )
    mobility_subspace = {
        ComponentMotionPreference.LOCKED: None,
        ComponentMotionPreference.GUIDED: "radial_axial_rotation",
        ComponentMotionPreference.FREE: "bounded_se3",
    }[motion]

    overrides: dict[str, float | int] = dict(_PACKING[preferences.packing])
    pairs, coverage_scale, continuity_scale = _AREA[preferences.interface_area]
    overrides["graph_interface_guidance_pairs_per_edge"] = pairs
    overrides["graph_interface_guidance_coverage_weight"] *= coverage_scale
    overrides["graph_interface_guidance_continuity_weight"] *= continuity_scale

    # Safety terms are compiler-owned and never reduced by a public preset.
    overrides.update(
        {
            "graph_interface_guidance_clash_weight": 8.0,
            "graph_interface_guidance_backbone_weight": 0.1,
            "graph_interface_guidance_patch_exclusivity_weight": 1.0,
        }
    )
    if design.guidance is not None:
        expert = design.guidance.model_dump(exclude_none=True)
        intra_chain_weight = expert.pop("intra_chain_weight", None)
        inter_chain_weight = expert.pop("inter_chain_weight", None)
        inter_chain_excess_penalty = expert.pop(
            "inter_chain_excess_penalty",
            None,
        )
        overrides.update(
            {
                f"graph_interface_guidance_{name}": value
                for name, value in expert.items()
            }
        )
        if intra_chain_weight is not None:
            overrides["scaffold_core_intra_chain_weight"] = intra_chain_weight
        if inter_chain_weight is not None:
            overrides["scaffold_core_inter_chain_weight"] = inter_chain_weight
            # RFdiffusion-style ``inter`` controls broad contact capture. It
            # must not scale Mosaic continuity, orientation, or safety terms.
            overrides[
                "graph_interface_guidance_contact_prior_weight"
            ] = inter_chain_weight
        if inter_chain_excess_penalty is not None:
            overrides["scaffold_core_inter_chain_excess_penalty"] = (
                inter_chain_excess_penalty
            )
    return ResolvedDesignPreferences(
        packing=preferences.packing,
        cavity=preferences.cavity,
        diversity=preferences.diversity,
        interface_area=preferences.interface_area,
        component_motion=motion,
        mobility_subspace=mobility_subspace,
        initial_radius_scale=_RADIUS_SCALE[preferences.cavity],
        diversity_plan=_DIVERSITY[preferences.diversity],
        sampler_overrides=overrides,
    )


def resolved_preferences_payload(design: UserDesignSpec) -> dict[str, Any]:
    return compile_design_preferences(design).model_dump(mode="json")


__all__ = [
    "ResolvedDesignPreferences",
    "ResolvedDiversityPlan",
    "compile_design_preferences",
    "resolved_preferences_payload",
]
