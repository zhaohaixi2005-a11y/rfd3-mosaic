"""Lifecycle owner for symmetry-coupled Mosaic constraints.

The RFD3 sampler owns the EDM integration and model invocation.  This module
owns the ordered constraint lifecycle around those operations so initial,
predicted, Euler-updated and final coordinates cannot silently use different
projection or target-refresh rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector


ProposalSource = Literal["denoiser", "scaffold_boundary"]


@dataclass(frozen=True)
class ConstraintProposalResult:
    """One optional hard-target update proposed during a diffusion step."""

    target: torch.Tensor
    applied: bool


ProposalHook = Callable[
    [torch.Tensor, float],
    ConstraintProposalResult,
]
ConditioningSynchronizer = Callable[[torch.Tensor], None]


@dataclass
class MosaicConstraintRuntime:
    """Execute one ordered hard-constraint lifecycle for an exact assembly.

    Mobility is optional.  When present, its proposal hook receives either the
    raw denoiser prediction or a hard-constraint-restored scaffold snapshot.
    A changed target is committed before the prediction is projected, and its
    conditioning is refreshed through the same runtime transaction.
    """

    projector: UnifiedJointProjector
    fixed_target: torch.Tensor
    fixed_mask: torch.Tensor
    proposal_source: ProposalSource = "denoiser"
    proposal_interval: int = 1
    proposal_hook: ProposalHook | None = None
    synchronize_conditioning: ConditioningSynchronizer | None = None
    conditioning_refresh_count: int = 0
    _phase_counts: dict[str, int] = field(
        default_factory=lambda: {
            "initialize": 0,
            "model_prediction": 0,
            "proposal": 0,
            "proposal_applied": 0,
            "state_update": 0,
            "post_guidance": 0,
            "finalize": 0,
        },
        init=False,
        repr=False,
    )
    _state: Literal["created", "running", "finalized"] = field(
        default="created",
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.fixed_target = self._validated_target(self.fixed_target).clone()
        mask = torch.as_tensor(
            self.fixed_mask,
            dtype=torch.bool,
            device=self.fixed_target.device,
        )
        if mask.ndim != 1 or mask.shape[0] != self.fixed_target.shape[1]:
            raise ValueError(
                "Constraint runtime fixed mask must have shape [L]"
            )
        self.fixed_mask = mask
        if self.proposal_source not in {"denoiser", "scaffold_boundary"}:
            raise ValueError(
                "Constraint runtime proposal source must be denoiser or "
                "scaffold_boundary"
            )
        if int(self.proposal_interval) <= 0:
            raise ValueError(
                "Constraint runtime proposal interval must be positive"
            )
        self.proposal_interval = int(self.proposal_interval)

    def _validated_target(self, target: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(target)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(
                "Constraint runtime target must have shape [D, L, 3]"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                "Constraint runtime target contains NaN or Inf"
            )
        if hasattr(self, "fixed_target") and tuple(value.shape) != tuple(
            self.fixed_target.shape
        ):
            raise ValueError(
                "Constraint runtime target shape changed during sampling"
            )
        if hasattr(self, "fixed_target") and value.device != (
            self.fixed_target.device
        ):
            raise ValueError(
                "Constraint runtime target device changed during sampling"
            )
        return value

    def _project(
        self,
        coordinates: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        if tuple(coordinates.shape) != tuple(self.fixed_target.shape):
            raise ValueError(
                "Constraint runtime coordinates must match target shape"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError(
                f"{label} contains NaN or Inf coordinates"
            )
        return self.projector.project(
            coordinates,
            constraint_target=self.fixed_target,
            constraint_mask=self.fixed_mask,
            restore=True,
            label=label,
        )

    def synchronize_initial_conditioning(self) -> None:
        """Publish the initial exact target through the runtime boundary."""

        if self.synchronize_conditioning is None:
            return
        if self._state != "created" or self.conditioning_refresh_count:
            raise RuntimeError(
                "Initial conditioning can be synchronized only once before "
                "runtime initialization"
            )
        self.synchronize_conditioning(self.fixed_target)
        self.conditioning_refresh_count += 1

    def _require_running(self, phase: str) -> None:
        if self._state != "running":
            raise RuntimeError(
                f"Constraint runtime cannot execute {phase} while "
                f"{self._state}"
            )

    def initialize_state(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self._state != "created":
            raise RuntimeError(
                "Constraint runtime can initialize only one diffusion state"
            )
        initialized = self._project(
            coordinates,
            label="Initial diffusion state",
        )
        self._phase_counts["initialize"] += 1
        self._state = "running"
        return initialized

    def process_model_prediction(
        self,
        coordinates: torch.Tensor,
        *,
        step_num: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Optionally update the target, then project one model prediction."""

        self._require_running("model_prediction")
        if step_num < 0 or total_steps <= 0 or step_num >= total_steps:
            raise ValueError(
                "Constraint runtime received an invalid diffusion step"
            )
        self._phase_counts["model_prediction"] += 1
        should_propose = (
            self.proposal_hook is not None
            and (
                self.proposal_source == "denoiser"
                or step_num % self.proposal_interval == 0
            )
        )
        if should_propose:
            progress = step_num / max(total_steps - 1, 1)
            proposal_coordinates = coordinates
            if self.proposal_source == "scaffold_boundary":
                proposal_coordinates = self._project(
                    coordinates,
                    label=(
                        "Scaffold-guidance model prediction at step "
                        f"{step_num}"
                    ),
                )
            with torch.autocast(
                device_type=coordinates.device.type,
                enabled=False,
            ):
                proposal = self.proposal_hook(
                    proposal_coordinates.detach().to(torch.float32),
                    progress,
                )
            if not isinstance(proposal, ConstraintProposalResult):
                raise TypeError(
                    "Constraint proposal hook must return "
                    "ConstraintProposalResult"
                )
            target = self._validated_target(proposal.target)
            self._phase_counts["proposal"] += 1
            if proposal.applied:
                self.fixed_target = target.clone()
                self._phase_counts["proposal_applied"] += 1
                if self.synchronize_conditioning is not None:
                    self.synchronize_conditioning(self.fixed_target)
                    self.conditioning_refresh_count += 1

        return self._project(
            coordinates,
            label=f"Denoised model prediction at step {step_num}",
        )

    def project_state_update(
        self,
        coordinates: torch.Tensor,
        *,
        step_num: int,
    ) -> torch.Tensor:
        self._require_running("state_update")
        self._phase_counts["state_update"] += 1
        return self._project(
            coordinates,
            label=f"Euler-updated diffusion state at step {step_num}",
        )

    def project_post_guidance(
        self,
        coordinates: torch.Tensor,
        *,
        step_num: int,
    ) -> torch.Tensor:
        self._require_running("post_guidance")
        self._phase_counts["post_guidance"] += 1
        return self._project(
            coordinates,
            label=f"Post-guidance diffusion state at step {step_num}",
        )

    def finalize(self, coordinates: torch.Tensor) -> torch.Tensor:
        self._require_running("finalize")
        self._phase_counts["finalize"] += 1
        finalized = self._project(
            coordinates,
            label="Final diffusion state",
        )
        self._state = "finalized"
        return finalized

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "proposal_source": self.proposal_source,
            "proposal_interval": self.proposal_interval,
            "state": self._state,
            "conditioning_refresh_count": (
                self.conditioning_refresh_count
            ),
            "phase_counts": dict(self._phase_counts),
        }


__all__ = [
    "ConstraintProposalResult",
    "MosaicConstraintRuntime",
]
