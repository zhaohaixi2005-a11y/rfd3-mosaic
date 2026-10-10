"""Final stored proposals publish atomically across the actual runtime boundary."""

from unittest.mock import patch

import pytest
import torch
from rfd3.inference.symmetry.constraint_runtime import (
    ConstraintProposalResult,
    MosaicConstraintRuntime,
)
from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry


def projector(restore=None, closure=None, symmetry=None):
    return UnifiedJointProjector(
        symmetry or (lambda x: x),
        restore or (lambda x, t, m: torch.where(m[None, :, None], t, x)),
        closure or (lambda x, label: None),
    )


def test_proposal_uses_private_membership_targets_until_conditioning_publish():
    old = torch.zeros(1, 3, 3)
    fixed = torch.tensor([True, True, False])
    target = old + 0.125
    f = {
        "motif_pos": old[0],
        "motif_constraint_group_membership": torch.tensor(
            [[True, False, False], [False, True, False]]
        ),
    }
    sampler = SampleDiffusionWithSymmetry(
        gamma_0=0.6, require_motif_constraint_groups=True
    )
    sampler._synchronize_mobile_motif_conditioning(f, old, fixed)

    def make(target):
        private = dict(f)
        sampler._synchronize_mobile_motif_conditioning(private, target, fixed)
        return projector(
            restore=lambda x, t, m: sampler._restore_motif_constraint_groups(
                x, t, m, private
            )
        )

    runtime = MosaicConstraintRuntime(
        projector=projector(
            restore=lambda x, t, m: sampler._restore_motif_constraint_groups(x, t, m, f)
        ),
        fixed_target=old,
        fixed_mask=fixed,
        proposal_hook=lambda x, p: ConstraintProposalResult(
            target, True, coordinates=target.clone()
        ),
        proposal_projector_factory=make,
        synchronize_conditioning=lambda t: sampler._synchronize_mobile_motif_conditioning(
            f, t, fixed
        ),
    )
    runtime.initialize_state(old)
    output = runtime.process_model_prediction(old, step_num=0, total_steps=2)
    assert torch.equal(output[:, fixed], target[:, fixed])
    assert torch.equal(f["motif_pos"][fixed], target[0, fixed])


@pytest.mark.parametrize(
    "failure", ["none", "closure", "cylindrical", "final_guard", "sync", "commit"]
)
def test_final_state_has_no_reprojection_and_all_failure_phases_roll_back(failure):
    old = torch.zeros(1, 3, 3)
    fixed = torch.tensor([True, True, False])
    target = old + 0.125
    final = target.clone()
    final[:, ~fixed, 0] = 2.0
    events = []
    features = {"target": old.clone()}
    history = []

    def symmetry(x):
        events.append("projection")
        return x + 1e-5

    def closure(x, label):
        if failure == "closure":
            raise ValueError("injected closure failure")

    def validate(x, t):
        assert torch.equal(x, final)
        if failure == "final_guard":
            raise ValueError("injected guard failure")

    def sync(t):
        events.append("sync")
        features["target"] = t.clone()
        if failure == "sync":
            raise ValueError("injected sync failure")

    def commit():
        events.append("commit")
        history.append("new reference")
        if failure == "commit":
            raise ValueError("injected commit failure")

    def rollback():
        events.append("rollback")
        history.clear()
        features["target"] = old.clone()

    runtime = MosaicConstraintRuntime(
        projector=projector(symmetry=symmetry, closure=closure),
        fixed_target=old,
        fixed_mask=fixed,
        synchronize_conditioning=sync,
        proposal_hook=lambda x, p: ConstraintProposalResult(
            target,
            True,
            coordinates=final,
            coordinates_final=True,
            validate_final=validate,
            commit=commit,
            rollback=rollback,
        ),
    )
    runtime._state = "running"
    if failure == "cylindrical":

        class Cylinder:
            def maximum_error(self, x):
                return 0.01

        runtime.cylindrical_projector = Cylinder()
    if failure == "none":
        output = runtime.process_model_prediction(old, step_num=0, total_steps=2)
        assert torch.equal(output, final)
        assert events == ["sync", "commit"]
        assert history == ["new reference"]
        assert runtime.conditioning_refresh_count == 1
    else:
        with pytest.raises(ValueError):
            runtime.process_model_prediction(old, step_num=0, total_steps=2)
        assert "projection" not in events
        assert events[-1] == "rollback"
        assert not history
        assert torch.equal(features["target"], old)
        assert torch.equal(runtime.fixed_target, old)
        assert runtime.conditioning_refresh_count == 0
        assert runtime.diagnostics()["phase_counts"]["proposal_applied"] == 0


@pytest.mark.parametrize("failure", ["target", "coordinates", "allocation", "attempt_count", "publication", "interrupt"])
def test_entire_proposal_boundary_rolls_back_and_can_retry(failure):
    old = torch.zeros(1, 3, 3)
    fixed = torch.tensor([True, True, False])
    target = old + 0.125
    state = {"features": old, "reference": "old", "rollbacks": 0}
    publications = []
    inject = True

    def sync(value):
        publications.append("sync")
        state["features"] = value

    def commit():
        publications.append("commit")
        state["reference"] = "new"
        if inject and failure == "interrupt":
            raise KeyboardInterrupt("injected interruption")

    def rollback():
        state.update(features=old, reference="old", rollbacks=state["rollbacks"] + 1)

    def propose(coordinates, progress):
        proposed_target = target
        proposed_coordinates = target
        if inject and failure == "target":
            proposed_target = target[:, :1]
        if inject and failure == "coordinates":
            proposed_coordinates = target[:, :1]
        return ConstraintProposalResult(
            proposed_target, True, coordinates=proposed_coordinates,
            coordinates_final=True, validate_final=lambda x, t: None,
            commit=commit, rollback=rollback,
        )

    runtime = MosaicConstraintRuntime(
        projector=projector(), fixed_target=old, fixed_mask=fixed,
        proposal_hook=propose, synchronize_conditioning=sync,
    )
    runtime.initialize_state(old)
    initial_target = runtime.fixed_target
    initial_counts = dict(runtime.diagnostics()["phase_counts"])

    class FailPublicationOnce(dict):
        def __setitem__(self, key, value):
            failed_key = "proposal" if failure == "attempt_count" else "proposal_applied"
            if inject and key == failed_key and value == 1:
                raise MemoryError("injected publication failure")
            super().__setitem__(key, value)

    if failure in {"publication", "attempt_count"}:
        runtime._phase_counts = FailPublicationOnce(runtime._phase_counts)
    native_clone = torch.Tensor.clone

    def clone(value, *args, **kwargs):
        if inject and failure == "allocation" and value is target:
            raise MemoryError("injected allocation failure")
        return native_clone(value, *args, **kwargs)

    expected = KeyboardInterrupt if failure == "interrupt" else (ValueError, MemoryError)
    with patch.object(torch.Tensor, "clone", clone), pytest.raises(expected):
        runtime.process_model_prediction(old, step_num=0, total_steps=2)
    assert state["rollbacks"] == 1
    assert state["features"] is old
    assert state["reference"] == "old"
    assert runtime.fixed_target is initial_target
    assert runtime.conditioning_refresh_count == 0
    assert runtime.diagnostics()["phase_counts"]["proposal_applied"] == 0
    assert runtime.diagnostics()["phase_counts"] == initial_counts
    if failure in {"allocation", "target", "coordinates", "attempt_count"}:
        assert not publications

    inject = False
    result = runtime.process_model_prediction(old, step_num=0, total_steps=2)
    assert torch.equal(result, target)
    assert torch.equal(runtime.fixed_target, target)
    assert runtime.fixed_target.data_ptr() != target.data_ptr()
    assert state["reference"] == "new"
    assert runtime.conditioning_refresh_count == 1
    assert runtime.diagnostics()["phase_counts"]["proposal_applied"] == 1


def test_unapplied_proposal_projection_failure_rolls_back_producer_state():
    old = torch.zeros(1, 3, 3)
    state = []

    def fail_projection(value):
        raise ValueError("injected fallback projection failure")

    def propose(coordinates, progress):
        state.append("attempt")
        return ConstraintProposalResult(old, False, rollback=state.clear)

    runtime = MosaicConstraintRuntime(
        projector=projector(symmetry=fail_projection), fixed_target=old,
        fixed_mask=torch.tensor([True, True, False]), proposal_hook=propose,
    )
    runtime._state = "running"
    with pytest.raises(ValueError, match="fallback projection"):
        runtime.process_model_prediction(old, step_num=0, total_steps=2)
    assert state == []
    assert torch.equal(runtime.fixed_target, old)
    assert runtime.conditioning_refresh_count == 0
    assert runtime.diagnostics()["phase_counts"]["proposal_applied"] == 0
