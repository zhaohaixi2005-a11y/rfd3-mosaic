"""Undefined trial objectives must not abort valid bounded pose searches."""

from dataclasses import replace

import pytest
import test_motif_mobility as mobility_fixture
import torch
from rfd3.inference.symmetry.scaffold_guidance import propose_bounded_se3_step


def test_combined_only_invalid_pose_is_not_scored_and_joint_backtracks():
    target, scaffold, topology, axis, config, controller = (
        mobility_fixture.MotifMobilityTestCase._two_orbit_scaffold_guidance_case()
    )
    indices = [int(m.master_atom_indices[0]) for m in controller.motifs]
    origin = target[0, indices, 0].sum()
    scored = []
    rejected = []
    for motif in controller.motifs:
        motif.mobility_subspace = "bounded_se3"
        motif.per_step_rotation_degrees = 0.0

    def displacement(value):
        return value[indices, 0].sum() - origin

    def energy(value):
        delta = displacement(value)
        scored.append(float(delta.detach()))
        if float(delta.detach()) > 0.18:
            raise ValueError("Combined-only invalid reference reached scoring")
        return (delta - 0.1).square()

    def validate(value):
        delta = float(displacement(value[0]))
        accepted = delta <= 0.18
        if not accepted:
            rejected.append(delta)
        return {"accepted": accepted, "reason": "Combined-only reference invalid"}

    result = controller.update_orbits_from_scaffold(
        scaffold,
        progress=0.5,
        topology=topology,
        axis=axis,
        principal_axes=(None, None),
        config=replace(config, junction_weight=0.0),
        apply_update=True,
        pose_energy=energy,
        candidate_validator=validate,
    )

    assert controller.last_update_applied
    snapshot = controller.diagnostics()["trajectory"][-1]
    # Both local updates exist; it is their combination that needs backtracking.
    assert all(record["accepted"] for record in snapshot["orbit_proposals"])
    rejected_joint = [
        trial
        for trial in snapshot["joint_line_search_trials"]
        if trial.get("first_rejection_reason") == "geometry_guard"
    ]
    assert rejected_joint, snapshot["joint_line_search_trials"]
    assert all(
        trial["total"] is None and not trial["energy_evaluated"]
        for trial in rejected_joint
    )
    assert max(scored) <= 0.18
    assert rejected
    assert 0 < float(displacement(result[0])) <= 0.18


@pytest.mark.parametrize("kind", ["all_invalid", "baseline_error"])
def test_rejected_trials_keep_baseline_without_hiding_baseline_errors(kind):
    calls = []

    def objective(rotation, translation):
        calls.append(float(translation[0].detach()))
        if kind == "baseline_error":
            raise ValueError("Invalid baseline must propagate")
        if float(translation.norm().detach()) > 0:
            raise ValueError("A rejected trial reached scoring")
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        return (translation[0] - 1).square() + (rotation - identity).square().sum()

    def run():
        return propose_bounded_se3_step(
            torch.eye(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            objective,
            maximum_step_translation=0.25,
            maximum_step_rotation_degrees=1.0,
            maximum_total_translation=1.0,
            maximum_total_rotation_degrees=5.0,
            line_search_scales=(1.0, 0.5),
            candidate_validator=lambda r, t: {
                "accepted": False,
                "reason": "Expected rejection",
            },
        )

    if kind == "baseline_error":
        with pytest.raises(ValueError, match="Invalid baseline must propagate"):
            run()
    else:
        proposal = run()
        assert not proposal.accepted
        assert torch.equal(proposal.rotation, torch.eye(3, dtype=torch.float64))
        assert torch.equal(proposal.translation, torch.zeros(3, dtype=torch.float64))
        assert all(value == 0 for value in calls)
        assert all(
            not trial["energy_evaluated"] for trial in proposal.line_search_trials
        )
