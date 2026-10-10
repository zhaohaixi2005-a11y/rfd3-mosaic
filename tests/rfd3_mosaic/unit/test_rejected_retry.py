import json

import pytest
from rfd3.inference.symmetry.constraint_runtime import MosaicConstraintRuntime
from test_sampler_transaction_rollback import _closure, _sampler_case


@pytest.mark.parametrize("path", ["unapplied", "prepare_rejected"])
def test_actual_partial_unapplied_fallback_failure(monkeypatch, path):
    run, *_ = _sampler_case(monkeypatch, partial=True)
    native_process = MosaicConstraintRuntime.process_model_prediction
    seen = {}

    def wrapped(runtime, coordinates, *, step_num, total_steps):
        if step_num != 1:
            return native_process(
                runtime, coordinates, step_num=step_num, total_steps=total_steps
            )
        live = _closure(runtime.proposal_hook)
        reference = live["reference_transport"]
        controller_type = type(live["motif_mobility_controller"])
        old_attempts = len(reference.proposal_attempts)
        old_counts = dict(runtime._phase_counts)
        old_rejected = len(reference.rejected)

        def noapply(controller, coordinates, *, progress, **kwargs):
            controller.last_update_applied = False
            return controller.fixed_target

        with monkeypatch.context() as m:
            if path == "unapplied":
                m.setattr(controller_type, "update_orbits_from_scaffold", noapply)
            else:
                native_prepare = reference.prepare
                prepare_calls = 0

                def reject_prepare(*args, **kwargs):
                    nonlocal prepare_calls
                    prepare_calls += 1
                    if prepare_calls == 1:
                        return native_prepare(*args, **kwargs)
                    raise ValueError("injected prepare rejection")

                m.setattr(reference, "prepare", reject_prepare)
            original_project = runtime._project
            project_calls = 0

            def fail_project(*args, **kwargs):
                nonlocal project_calls
                project_calls += 1
                if project_calls == 1:
                    return original_project(*args, **kwargs)
                raise RuntimeError("injected fallback projection failure")

            m.setattr(runtime, "_project", fail_project)
            try:
                native_process(
                    runtime, coordinates, step_num=step_num, total_steps=total_steps
                )
            except RuntimeError as error:
                assert str(error) == "injected fallback projection failure"
            else:
                raise AssertionError("fault did not fire")
        seen.update(
            path=path,
            rejected_before=old_rejected,
            rejected_after_failure=len(reference.rejected),
            attempts_before=old_attempts,
            attempts_after_failure=len(reference.proposal_attempts),
            counts_before=old_counts,
            counts_after_failure=dict(runtime._phase_counts),
        )
        result = native_process(
            runtime, coordinates, step_num=step_num, total_steps=total_steps
        )
        seen.update(
            attempts_after_retry=len(reference.proposal_attempts),
            counts_after_retry=dict(runtime._phase_counts),
        )
        return result

    monkeypatch.setattr(MosaicConstraintRuntime, "process_model_prediction", wrapped)
    run()
    print(json.dumps(seen, sort_keys=True))
    assert seen["counts_after_failure"] == seen["counts_before"]
    assert seen["attempts_after_failure"] == seen["attempts_before"]

    assert seen["rejected_after_failure"] == seen["rejected_before"]

    assert seen["attempts_after_retry"] == seen["attempts_before"] + 1
    assert seen["attempts_after_retry"] == seen["counts_after_retry"]["proposal"]


class StopAfterBoundary(Exception):
    pass


@pytest.mark.parametrize("path", ["unapplied", "prepare_rejected"])
@pytest.mark.parametrize("recording_failure", [False, True])
def test_normal_rejections_keep_history_but_recording_fault_restores(
    monkeypatch, path, recording_failure
):
    run, *_ = _sampler_case(monkeypatch, partial=True)
    native_process = MosaicConstraintRuntime.process_model_prediction

    def wrapped(runtime, coordinates, *, step_num, total_steps):
        if step_num != 1:
            return native_process(
                runtime, coordinates, step_num=step_num, total_steps=total_steps
            )
        live = _closure(runtime.proposal_hook)
        reference = live["reference_transport"]
        controller_type = type(live["motif_mobility_controller"])
        prior = {"reason": "prior rejection must survive"}
        reference.rejected.append(prior)
        counts = dict(runtime._phase_counts)
        attempts = list(reference.proposal_attempts)
        rejections = list(reference.rejected)
        accepted = list(reference.accepted)
        target = runtime.fixed_target
        oldref = reference.reference

        def noapply(controller, coordinates, *, progress, **kwargs):
            controller.last_update_applied = False
            return controller.fixed_target

        with monkeypatch.context() as m:
            if path == "unapplied":
                m.setattr(controller_type, "update_orbits_from_scaffold", noapply)
            else:
                original = reference.prepare
                calls = 0

                def reject(*a, **kw):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return original(*a, **kw)
                    raise ValueError("injected prepare rejection")

                m.setattr(reference, "prepare", reject)
            if recording_failure:
                record = reference.record_proposal_attempt

                def fail_record(*a, **kw):
                    record(*a, **kw)
                    raise RuntimeError("injected record append failure")

                m.setattr(reference, "record_proposal_attempt", fail_record)
                with pytest.raises(
                    RuntimeError, match="injected record append failure"
                ):
                    native_process(
                        runtime, coordinates, step_num=step_num, total_steps=total_steps
                    )
                assert reference.proposal_attempts == attempts
                assert reference.rejected == rejections
                assert runtime._phase_counts == counts
            else:
                native_process(
                    runtime, coordinates, step_num=step_num, total_steps=total_steps
                )
                assert len(reference.proposal_attempts) == len(attempts) + 1
                assert reference.proposal_attempts[:-1] == attempts
                assert reference.rejected[: len(rejections)] == rejections
                assert len(reference.rejected) == len(rejections) + int(
                    path == "prepare_rejected"
                )
                assert runtime._phase_counts["proposal"] == counts["proposal"] + 1
                assert (
                    runtime._phase_counts["proposal_applied"]
                    == counts["proposal_applied"]
                )
        assert reference.rejected[0] is prior
        assert reference.reference is oldref
        assert reference.accepted == accepted
        assert runtime.fixed_target is target
        raise StopAfterBoundary

    monkeypatch.setattr(MosaicConstraintRuntime, "process_model_prediction", wrapped)
    with pytest.raises(StopAfterBoundary):
        run()
