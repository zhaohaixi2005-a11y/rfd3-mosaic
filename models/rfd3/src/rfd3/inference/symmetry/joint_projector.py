"""One ordered projection contract for symmetry and motif constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


SymmetryProjector = Callable[[torch.Tensor], torch.Tensor]
ConstraintRestorer = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
ClosureValidator = Callable[[torch.Tensor, str], None]


@dataclass(frozen=True)
class UnifiedJointProjector:
    """Apply symmetry, hard constraints and validation in one fixed order.

    The sampler owns the concrete symmetry representation and constraint
    targets.  This object owns the sequencing contract so static motifs,
    mobile-orbit targets and Euler-updated states cannot silently use
    different projection orders.
    """

    project_symmetry: SymmetryProjector
    restore_constraints: ConstraintRestorer
    validate_closure: ClosureValidator

    def project(
        self,
        coordinates: torch.Tensor,
        *,
        constraint_target: torch.Tensor,
        constraint_mask: torch.Tensor,
        restore: bool,
        label: str,
    ) -> torch.Tensor:
        projected = self.project_symmetry(coordinates)
        mask = torch.as_tensor(
            constraint_mask,
            dtype=torch.bool,
            device=coordinates.device,
        )
        if restore and torch.any(mask):
            projected = self.restore_constraints(
                projected,
                constraint_target,
                mask,
            )
        self.validate_closure(projected, label)
        return projected


__all__ = ["UnifiedJointProjector"]
