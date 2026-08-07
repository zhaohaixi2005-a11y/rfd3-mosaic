import unittest

import torch

from rfd3.inference.symmetry.constraint_runtime import (
    ConstraintProposalResult,
    MosaicConstraintRuntime,
)
from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector


class ConstraintRuntimeTestCase(unittest.TestCase):
    @staticmethod
    def _projector(labels: list[str]) -> UnifiedJointProjector:
        def project_symmetry(coordinates):
            return coordinates + 1.0

        def restore_constraints(coordinates, target, mask):
            restored = coordinates.clone()
            restored[:, mask, :] = target[:, mask, :]
            return restored

        def validate_closure(_, label):
            labels.append(label)

        return UnifiedJointProjector(
            project_symmetry=project_symmetry,
            restore_constraints=restore_constraints,
            validate_closure=validate_closure,
        )

    def test_owns_every_exact_projection_phase(self) -> None:
        labels: list[str] = []
        target = torch.zeros((1, 3, 3))
        fixed = torch.tensor([True, False, True])
        runtime = MosaicConstraintRuntime(
            projector=self._projector(labels),
            fixed_target=target,
            fixed_mask=fixed,
        )

        initialized = runtime.initialize_state(torch.ones_like(target))
        predicted = runtime.process_model_prediction(
            initialized,
            step_num=0,
            total_steps=1,
        )
        updated = runtime.project_state_update(predicted, step_num=0)
        guided = runtime.project_post_guidance(updated, step_num=0)
        finalized = runtime.finalize(guided)

        self.assertTrue(torch.all(finalized[:, fixed, :] == 0.0))
        self.assertEqual(
            labels,
            [
                "Initial diffusion state",
                "Denoised model prediction at step 0",
                "Euler-updated diffusion state at step 0",
                "Post-guidance diffusion state at step 0",
                "Final diffusion state",
            ],
        )
        self.assertEqual(
            runtime.diagnostics()["phase_counts"],
            {
                "initialize": 1,
                "model_prediction": 1,
                "proposal": 0,
                "proposal_applied": 0,
                "state_update": 1,
                "post_guidance": 1,
                "finalize": 1,
            },
        )
        self.assertEqual(runtime.diagnostics()["state"], "finalized")

    def test_rejects_out_of_order_lifecycle_calls(self) -> None:
        target = torch.zeros((1, 1, 3))
        runtime = MosaicConstraintRuntime(
            projector=self._projector([]),
            fixed_target=target,
            fixed_mask=torch.tensor([True]),
        )

        with self.assertRaisesRegex(RuntimeError, "while created"):
            runtime.process_model_prediction(
                target,
                step_num=0,
                total_steps=1,
            )
        runtime.initialize_state(target)
        runtime.finalize(target)
        with self.assertRaisesRegex(RuntimeError, "while finalized"):
            runtime.project_state_update(target, step_num=0)

    def test_schedules_and_commits_target_updates_transactionally(self) -> None:
        labels: list[str] = []
        target = torch.zeros((1, 2, 3))
        synchronized = []
        progress_values = []

        def proposal(_, progress):
            progress_values.append(progress)
            return ConstraintProposalResult(
                target=runtime.fixed_target + 2.0,
                applied=True,
            )

        runtime = MosaicConstraintRuntime(
            projector=self._projector(labels),
            fixed_target=target,
            fixed_mask=torch.ones(2, dtype=torch.bool),
            proposal_source="scaffold_boundary",
            proposal_interval=2,
            proposal_hook=proposal,
            synchronize_conditioning=(
                lambda value: synchronized.append(value.clone())
            ),
        )
        runtime.synchronize_initial_conditioning()
        runtime.initialize_state(target)
        output = target
        for step in range(4):
            output = runtime.process_model_prediction(
                output,
                step_num=step,
                total_steps=4,
            )

        self.assertEqual(progress_values, [0.0, 2.0 / 3.0])
        self.assertEqual(len(synchronized), 3)
        self.assertTrue(torch.all(runtime.fixed_target == 4.0))
        self.assertTrue(torch.all(output == 4.0))
        diagnostics = runtime.diagnostics()
        self.assertEqual(diagnostics["conditioning_refresh_count"], 3)
        self.assertEqual(diagnostics["phase_counts"]["proposal"], 2)
        self.assertEqual(
            diagnostics["phase_counts"]["proposal_applied"],
            2,
        )

    def test_rejected_proposal_cannot_change_the_hard_target(self) -> None:
        target = torch.zeros((1, 2, 3))

        def reject(_, __):
            return ConstraintProposalResult(
                target=torch.full_like(target, 100.0),
                applied=False,
            )

        runtime = MosaicConstraintRuntime(
            projector=self._projector([]),
            fixed_target=target,
            fixed_mask=torch.ones(2, dtype=torch.bool),
            proposal_hook=reject,
        )
        runtime.initialize_state(target)
        output = runtime.process_model_prediction(
            torch.ones_like(target),
            step_num=0,
            total_steps=1,
        )

        self.assertTrue(torch.all(runtime.fixed_target == 0.0))
        self.assertTrue(torch.all(output == 0.0))
        self.assertEqual(
            runtime.diagnostics()["phase_counts"]["proposal_applied"],
            0,
        )

    def test_rejects_nonfinite_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            MosaicConstraintRuntime(
                projector=self._projector([]),
                fixed_target=torch.tensor([[[float("nan"), 0.0, 0.0]]]),
                fixed_mask=torch.tensor([True]),
            )


if __name__ == "__main__":
    unittest.main()
