"""Faults across the real sampler publication boundary must permit clean retry.

The network/controller are deterministic CPU fixtures. Projection, conditioning,
reference preparation, sampler commit/rollback closures and runtime are real.
Real nonempty patch state is deliberately mutated by the fixture controller;
this exercises publication ownership, not graph-packing optimization quality.
"""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from rfd3.inference.symmetry.constraint_runtime import MosaicConstraintRuntime
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfacePatchAssignment,
    GraphInterfacePatchState,
)
from rfd3.model.inference_sampler import SampleDiffusionWithSymmetry
from test_reference_transport import transport_fixture
from test_symmetry_motif_finalization import _RecordingScaffoldController

from rfd3_mosaic.validation.reference_transport import replay_reference_transport


class InjectedPublicationFailure(RuntimeError):
    pass


def _closure(function):
    """Observe actual sampler-owned objects, without replacing its callbacks."""
    return {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ())
    }


def _same_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _same_tree(left, right)
    elif hasattr(expected, "__dict__"):
        assert type(actual) is type(expected)
        _same_tree(actual.__dict__, expected.__dict__)
    else:
        assert actual == expected


def _sampler_case(monkeypatch, *, partial):
    contract, _xyz, plan, features, topology = transport_fixture()
    original = features["motif_pos"][None].clone()
    fixed = ~topology.generated_atom_mask
    features.update(
        {
            "symmetry_id": "C2",
            "sym_entity_id": torch.zeros(176, dtype=torch.long),
            "sym_transform_id": torch.arange(2).repeat_interleave(88),
            "is_sym_asu": torch.arange(176) < 88,
            "sym_orbit_slot": torch.arange(88).repeat(2),
            "sym_orbit_slot_verified": torch.tensor(True),
            "sym_transform": {
                str(i): (torch.tensor(m[:3, :3]), torch.tensor(m[:3, 3]))
                for i, m in enumerate(np.array(plan["registry_transforms"]))
            },
            "motif_constraint_group_membership": fixed[None],
            "motif_constraint_target_coordinates": original[:, None].clone(),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "is_motif_atom_with_fixed_coord": fixed,
            "ref_element": torch.zeros(176, dtype=torch.long),
            "token_bonds": torch.zeros((44, 44)),
        }
    )
    if partial:
        features["partial_t"] = 2.0
    else:
        for key in list(features):
            if key.startswith("mosaic_"):
                features.pop(key)

    class Controller(_RecordingScaffoldController):
        def update_orbits_from_scaffold(self, coordinates, *, progress, **kwargs):
            proposed = self.fixed_target.clone()
            proposed[:, :, 2] += 0.125
            if partial:
                trial = torch.where(fixed[None, :, None], proposed, coordinates)
                report = kwargs["candidate_validator"](trial)
                assert report["accepted"], report
            self.fixed_target = proposed
            self.calls.append({"active": True, "progress": float(progress)})
            self.last_update_applied = True
            return proposed

    controller = Controller(original)
    controller.motifs[0].group_atom_indices = tuple(
        torch.tensor(
            features.get(
                "mosaic_transport_fixed_atom_indices",
                [i for i in range(176) if fixed[i]],
            )
        )[group["atom_indices"]]
        for group in plan["groups"].values()
    )
    controller.motifs[0].master_atom_indices = controller.motifs[0].group_atom_indices[
        0
    ]
    controller.motifs[0].template_master = original[
        :, controller.motifs[0].master_atom_indices
    ]
    monkeypatch.setattr(
        "rfd3.model.inference_sampler.OrbitRigidMotifController.from_features",
        controller.bind_runtime_frames,
    )

    class Denoiser(torch.nn.Module):
        def forward(self, X_noisy_L, f, **kwargs):
            if partial:
                coordinates = torch.tensor(
                    [
                        record["reference_backbone"]
                        for record in f["mosaic_scaffold_contract"]["residues"]
                    ],
                    dtype=original.dtype,
                ).reshape(1, -1, 3)
            else:
                coordinates = f["motif_pos"][None].clone()
            return {"X_L": coordinates}

    sampler = SampleDiffusionWithSymmetry(
        gamma_0=0.6,
        num_timesteps=6,
        preserve_fixed_motif_during_symmetry=True,
        require_motif_constraint_groups=True,
        symmetry_state_mode="orbit_average",
        symmetry_noise_mode="coupled",
        enable_orbit_rigid_motif_mobility=True,
        motif_mobility_proposal_source="scaffold_boundary",
        motif_mobility_apply_updates=True,
        motif_mobility_target_update_count=0,
        motif_mobility_update_interval=1,
    )
    monkeypatch.setattr(
        sampler,
        "_construct_inference_noise_schedule",
        lambda **kw: torch.tensor(
            [1.0, 0.5, 0.2, 0.0],
            dtype=original.dtype,
        ),
    )

    def run():
        with torch.no_grad():
            return sampler.sample_diffusion_like_af3(
                f=features,
                diffusion_module=Denoiser(),
                diffusion_batch_size=1,
                coord_atom_lvl_to_be_noised=original,
                initializer_outputs={"chunked_pairwise_embedder": object()},
                ref_initializer_outputs=None,
                f_ref=None,
            )

    return run, features, original, fixed, contract, plan


@pytest.mark.parametrize(
    "partial,failure_stage",
    [
        (True, "postcommit"),
        (False, "postcommit"),
        (True, "private_second_copy"),
        (True, "hook_raise"),
        (False, "hook_raise"),
    ],
    ids=[
        "partial-postcommit",
        "fullnoise-postcommit",
        "partial-private-allocation",
        "partial-hook-failure",
        "fullnoise-hook-failure",
    ],
)
def test_real_sampler_postcommit_failure_restores_every_state_then_retries(
    monkeypatch, partial, failure_stage
):
    run, inputs, original, fixed, contract, plan = _sampler_case(
        monkeypatch, partial=partial
    )
    native_process = MosaicConstraintRuntime.process_model_prediction
    observations = []

    def fault_once_and_retry(runtime, coordinates, *, step_num, total_steps):
        if step_num != 1:
            return native_process(
                runtime, coordinates, step_num=step_num, total_steps=total_steps
            )
        hook = runtime.proposal_hook
        # Seed real mutable sampler state without enabling the graph optimizer:
        # existing graph optimization has separate behavioral unit coverage.
        cells = dict(zip(hook.__code__.co_freevars, hook.__closure__ or ()))
        seeded_patch = GraphInterfacePatchState(
            assignments={
                "prior-edge": GraphInterfacePatchAssignment((4, 5), (26, 27)),
            },
            locked=True,
            lock_reason="prior_committed_assignment",
        )
        cells["graph_interface_patch_state"].cell_contents = seeded_patch
        existing_diagnostic = {"phase": "prior_committed_packing", "marker": 17}
        cells["graph_interface_diagnostics"].cell_contents.append(existing_diagnostic)
        live = _closure(hook)
        old_controller = live["motif_mobility_controller"]
        old_frames = old_controller.sym_transforms
        old_frame_tensors = dict(old_frames)
        controller_before = copy.deepcopy(old_controller.__dict__)
        old_features = live["f"]
        features_before = copy.deepcopy(old_features)
        old_bindings = {
            name: live.get(name)
            for name in (
                "motif_mobility_controller",
                "graph_interface_patch_state",
                "graph_interface_diagnostics",
                "scaffold_core_topology",
            )
        }
        patch_before = copy.deepcopy(seeded_patch.__dict__)
        diagnostics_before = copy.deepcopy(live["graph_interface_diagnostics"])
        native_update = type(old_controller).update_orbits_from_scaffold
        hook_fault_armed = failure_stage == "hook_raise"

        def update_and_mutate_patch(controller, *args, **kwargs):
            nonlocal hook_fault_armed
            proposed = native_update(controller, *args, **kwargs)
            active = _closure(hook)
            active_patch = active["graph_interface_patch_state"]
            active_patch.assignments.clear()
            active_patch.assignments["trial-edge"] = GraphInterfacePatchAssignment(
                (6, 7), (28, 29)
            )
            active_patch.locked = False
            active_patch.lock_reason = "tentative_reassignment"
            active["graph_interface_diagnostics"].append(
                {"phase": "tentative_packing", "marker": 99}
            )
            if hook_fault_armed:
                hook_fault_armed = False
                raise InjectedPublicationFailure("injected sampler controller hook")
            return proposed

        monkeypatch.setattr(
            type(old_controller), "update_orbits_from_scaffold", update_and_mutate_patch
        )
        reference = live.get("reference_transport")
        if partial:
            assert reference is not None
            assert len(reference.accepted) == 1  # preserve a prior successful commit
            reference_before = (
                reference.reference,
                reference.transforms,
                reference.motions,
                copy.deepcopy(reference.accepted),
                copy.deepcopy(reference.proposal_attempts),
            )
        else:
            assert reference is None
        target_before = runtime.fixed_target
        target_values = target_before.clone()
        refresh_before = runtime.conditioning_refresh_count
        applied_before = runtime.diagnostics()["phase_counts"]["proposal_applied"]

        def injected_hook(candidate, progress):
            if failure_stage == "private_second_copy":
                native_deepcopy = copy.deepcopy
                controller_copy_finished = False

                def fail_second_copy(value, *args, **kwargs):
                    nonlocal controller_copy_finished
                    if (
                        controller_copy_finished
                        and value is old_bindings["graph_interface_patch_state"]
                    ):
                        raise InjectedPublicationFailure(
                            "injected sampler private allocation"
                        )
                    result = native_deepcopy(value, *args, **kwargs)
                    if value is old_controller:
                        controller_copy_finished = True
                    return result

                with monkeypatch.context() as fault:
                    fault.setattr(copy, "deepcopy", fail_second_copy)
                    return hook(candidate, progress)
            proposal = hook(candidate, progress)
            assert proposal.applied
            # Partial reference updates are deferred. Fullnoise may retain its
            # existing in-place proposal semantics, provided failures restore
            # every affected controller and conditioning field.
            if partial:
                assert _closure(hook)["motif_mobility_controller"] is old_controller
                _same_tree(old_controller.__dict__, controller_before)
            original_commit = proposal.commit
            if partial:
                assert original_commit is not None
                assert proposal.rollback is not None

            def fail_after_complete_commit():
                if original_commit is not None:
                    original_commit()
                if partial:
                    assert (
                        _closure(hook)["motif_mobility_controller"]
                        is not old_controller
                    )
                    assert len(reference.accepted) == len(reference_before[3]) + 1
                    assert reference.reference is not reference_before[0]
                else:
                    assert not torch.equal(
                        old_controller.fixed_target, controller_before["fixed_target"]
                    )
                published = _closure(hook)
                assert set(published["graph_interface_patch_state"].assignments) == {
                    "trial-edge"
                }
                assert not published["graph_interface_patch_state"].locked
                assert (
                    published["graph_interface_patch_state"].lock_reason
                    == "tentative_reassignment"
                )
                assert published["graph_interface_diagnostics"][-1]["marker"] == 99
                raise InjectedPublicationFailure("injected sampler postcommit failure")

            return replace(proposal, commit=fail_after_complete_commit)

        runtime.proposal_hook = injected_hook
        try:
            with pytest.raises(InjectedPublicationFailure, match="injected sampler"):
                native_process(
                    runtime, coordinates, step_num=step_num, total_steps=total_steps
                )
        finally:
            runtime.proposal_hook = hook
        restored = _closure(hook)
        for name, value in old_bindings.items():
            assert restored.get(name) is value, name
        assert restored["graph_interface_patch_state"] is seeded_patch
        _same_tree(seeded_patch.__dict__, patch_before)
        _same_tree(restored["graph_interface_diagnostics"], diagnostics_before)
        assert restored["graph_interface_diagnostics"][0] is existing_diagnostic
        _same_tree(old_controller.__dict__, controller_before)
        assert old_controller.sym_transforms is old_frames
        for key, (rotation, translation) in old_frame_tensors.items():
            assert old_controller.sym_transforms[key][0] is rotation
            assert old_controller.sym_transforms[key][1] is translation
        _same_tree(old_features, features_before)
        assert runtime.fixed_target is target_before
        assert torch.equal(runtime.fixed_target, target_values)
        assert runtime.conditioning_refresh_count == refresh_before
        assert (
            runtime.diagnostics()["phase_counts"]["proposal_applied"] == applied_before
        )
        if partial:
            assert reference.reference is reference_before[0]
            assert reference.transforms is reference_before[1]
            assert reference.motions is reference_before[2]
            _same_tree(reference.accepted, reference_before[3])
            _same_tree(reference.proposal_attempts, reference_before[4])

        output = native_process(
            runtime, coordinates, step_num=step_num, total_steps=total_steps
        )
        assert runtime.conditioning_refresh_count == refresh_before + 1
        assert (
            runtime.diagnostics()["phase_counts"]["proposal_applied"]
            == applied_before + 1
        )
        assert not torch.equal(runtime.fixed_target[:, fixed], target_values[:, fixed])
        if partial:
            assert len(reference.accepted) == len(reference_before[3]) + 1
        assert _closure(hook)["motif_mobility_controller"].sym_transforms is old_frames
        successful = _closure(hook)
        assert set(successful["graph_interface_patch_state"].assignments) == {
            "trial-edge"
        }
        assert successful["graph_interface_diagnostics"][0] is existing_diagnostic
        assert (
            len(successful["graph_interface_diagnostics"])
            == len(diagnostics_before) + 1
        )
        observations.append("rolled_back_then_retried")
        return output

    monkeypatch.setattr(
        MosaicConstraintRuntime, "process_model_prediction", fault_once_and_retry
    )
    result = run()
    assert observations == ["rolled_back_then_retried"]
    assert torch.isfinite(result["X_L"]).all()
    assert torch.equal(inputs["motif_pos"], original[0])
    runtime_report = result["constraint_runtime_diagnostics"]
    phases = runtime_report["phase_counts"]
    assert phases["model_prediction"] == phases["state_update"] == 3
    assert phases["proposal"] == phases["proposal_applied"] == 3
    assert runtime_report["conditioning_refresh_count"] == 4  # initial + 3 commits
    assert result["motif_mobility_diagnostics"]["update_calls"] == phases["proposal"]
    if partial:
        diagnostics = result["scaffold_contract_diagnostics"]
        assert diagnostics["contract_met"]
        assert len(diagnostics["reference_transport"]["accepted_transitions"]) == 3
        assert (
            len(diagnostics["reference_transport"]["proposal_attempts"])
            == phases["proposal"]
        )
        replay_reference_transport(contract, plan, diagnostics["reference_transport"])
