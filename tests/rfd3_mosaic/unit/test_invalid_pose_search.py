"""A rejected trial cannot abort search through its undefined objective."""

import pytest
import torch
from rfd3.inference.symmetry.scaffold_guidance import propose_bounded_se3_step


def search(objective, validator):
    return propose_bounded_se3_step(
        torch.eye(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        objective,
        maximum_step_translation=0.25,
        maximum_step_rotation_degrees=1.0,
        maximum_total_translation=1.0,
        maximum_total_rotation_degrees=5.0,
        line_search_scales=(1.0, 0.5),
        candidate_validator=validator,
    )


def test_invalid_trial_is_not_scored_and_smaller_valid_trial_can_succeed():
    scored = []

    def objective(r, t):
        x = float(t[0].detach())
        scored.append(x)
        if x > 0.15:
            raise ValueError("block_contact")
        return (
            (t[0] - 1).square()
            + t[1:].square().sum()
            + (r - torch.eye(3, dtype=r.dtype)).square().sum()
        )

    proposal = search(
        objective,
        lambda r, t: {"accepted": float(t[0]) <= 0.15, "reason": "block_contact"},
    )
    assert proposal.accepted
    assert float(proposal.translation[0]) == pytest.approx(0.125)
    assert all(x <= 0.15 for x in scored)
    assert proposal.line_search_trials[0]["first_rejection_reason"] == "geometry_guard"
    assert proposal.line_search_trials[0]["energy"] is None


def test_valid_trial_unexpected_objective_error_still_propagates():
    def objective(r, t):
        if not t.requires_grad:
            raise ValueError("unexpected scoring bug")
        return (
            (t[0] - 1).square()
            + t[1:].square().sum()
            + (r - torch.eye(3, dtype=r.dtype)).square().sum()
        )

    with pytest.raises(ValueError, match="unexpected scoring bug"):
        search(objective, lambda r, t: {"accepted": True})
