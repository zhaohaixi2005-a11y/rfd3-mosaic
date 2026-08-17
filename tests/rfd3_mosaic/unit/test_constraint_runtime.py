import unittest

import torch

from rfd3.inference.symmetry.constraint_runtime import (
    ConstraintProposalResult,
    MosaicConstraintRuntime,
)
from rfd3.inference.symmetry.cylindrical_projector import (
    CylindricalCoordinateProjector,
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
        self.assertEqual(runtime.diagnostics()["schema_version"], 2)
        self.assertEqual(
            runtime.diagnostics()["final_fixed_target_rmsd"],
            0.0,
        )
        self.assertEqual(
            runtime.diagnostics()["final_fixed_target_maximum_error"],
            0.0,
        )

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

    def test_joint_proposal_commits_target_and_scaffold_coordinates(self) -> None:
        target = torch.zeros((1, 2, 3))

        def propose(_, __):
            coordinates = torch.full_like(target, 7.0)
            return ConstraintProposalResult(
                target=torch.full_like(target, 3.0),
                coordinates=coordinates,
                applied=True,
            )

        runtime = MosaicConstraintRuntime(
            projector=self._projector([]),
            fixed_target=target,
            fixed_mask=torch.tensor([True, False]),
            proposal_hook=propose,
        )
        runtime.initialize_state(target)
        output = runtime.process_model_prediction(
            torch.ones_like(target),
            step_num=0,
            total_steps=1,
        )

        self.assertTrue(torch.all(output[:, 0, :] == 3.0))
        # The proposal coordinate is first symmetrized by the test projector.
        self.assertTrue(torch.all(output[:, 1, :] == 8.0))
        self.assertTrue(torch.all(runtime.fixed_target == 3.0))

    def test_rejected_joint_proposal_cannot_leak_scaffold_coordinates(
        self,
    ) -> None:
        target = torch.zeros((1, 2, 3))

        def reject(_, __):
            return ConstraintProposalResult(
                target=torch.full_like(target, 3.0),
                coordinates=torch.full_like(target, 100.0),
                applied=False,
            )

        runtime = MosaicConstraintRuntime(
            projector=self._projector([]),
            fixed_target=target,
            fixed_mask=torch.tensor([True, False]),
            proposal_hook=reject,
        )
        runtime.initialize_state(target)
        output = runtime.process_model_prediction(
            torch.ones_like(target),
            step_num=0,
            total_steps=1,
        )

        self.assertTrue(torch.all(output[:, 0, :] == 0.0))
        self.assertTrue(torch.all(output[:, 1, :] == 2.0))

    def test_joint_proposal_rejects_coordinate_shape_change(self) -> None:
        target = torch.zeros((1, 2, 3))

        def invalid(_, __):
            return ConstraintProposalResult(
                target=target,
                coordinates=torch.zeros((1, 3, 3)),
                applied=True,
            )

        runtime = MosaicConstraintRuntime(
            projector=self._projector([]),
            fixed_target=target,
            fixed_mask=torch.tensor([True, False]),
            proposal_hook=invalid,
        )
        runtime.initialize_state(target)
        with self.assertRaisesRegex(ValueError, "must match"):
            runtime.process_model_prediction(
                target,
                step_num=0,
                total_steps=1,
            )

    def test_rejects_nonfinite_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            MosaicConstraintRuntime(
                projector=self._projector([]),
                fixed_target=torch.tensor([[[float("nan"), 0.0, 0.0]]]),
                fixed_mask=torch.tensor([True]),
            )

    def test_finalize_rejects_a_broken_constraint_restorer(self) -> None:
        projector = UnifiedJointProjector(
            project_symmetry=lambda coordinates: coordinates,
            restore_constraints=lambda coordinates, target, mask: coordinates,
            validate_closure=lambda coordinates, label: None,
        )
        target = torch.zeros((1, 1, 3))
        runtime = MosaicConstraintRuntime(
            projector=projector,
            fixed_target=target,
            fixed_mask=torch.tensor([True]),
        )
        runtime.initialize_state(target)
        with self.assertRaisesRegex(
            RuntimeError,
            "did not restore the runtime fixed target",
        ):
            runtime.finalize(torch.ones_like(target))

    def test_cylindrical_dofs_are_projected_at_every_runtime_phase(self) -> None:
        labels: list[str] = []
        reference = torch.tensor(
            [[[2.0, 0.0, 3.0], [0.0, 4.0, -1.0], [1.0, 1.0, 1.0]]]
        )
        keep = torch.tensor(
            [
                [True, False, False],
                [False, True, True],
                [False, False, False],
            ]
        )
        cylindrical = CylindricalCoordinateProjector(
            reference=reference,
            keep_mask=keep,
            axis=torch.tensor([0.0, 0.0, 1.0]),
            center=torch.zeros(3),
        )
        runtime = MosaicConstraintRuntime(
            projector=self._projector(labels),
            fixed_target=reference,
            fixed_mask=torch.zeros(3, dtype=torch.bool),
            cylindrical_projector=cylindrical,
        )
        state = torch.tensor(
            [[[0.0, 8.0, 9.0], [6.0, 0.0, 7.0], [5.0, 5.0, 5.0]]]
        )
        initialized = runtime.initialize_state(state)
        # Token 0 keeps radius=2 but retains its post-symmetry azimuth/axial.
        self.assertAlmostEqual(
            torch.linalg.vector_norm(initialized[0, 0, :2]).item(),
            2.0,
            places=5,
        )
        # Token 1 keeps the reference +Y azimuth and z=-1 while retaining
        # the sampled radial magnitude after symmetry projection.
        self.assertAlmostEqual(initialized[0, 1, 0].item(), 0.0, places=5)
        self.assertGreater(initialized[0, 1, 1].item(), 0.0)
        self.assertAlmostEqual(initialized[0, 1, 2].item(), -1.0, places=5)
        output = runtime.process_model_prediction(
            initialized,
            step_num=0,
            total_steps=1,
        )
        output = runtime.project_state_update(output, step_num=0)
        output = runtime.project_post_guidance(output, step_num=0)
        output = runtime.finalize(output)
        self.assertLessEqual(cylindrical.maximum_error(output), 1.0e-6)
        self.assertTrue(
            runtime.diagnostics()["cylindrical_projector_active"]
        )
        self.assertEqual(len(labels), 10)

    def test_cylindrical_projection_composes_independent_dofs(self) -> None:
        reference = torch.tensor([[[3.0, 0.0, 2.0]]])
        sampled = torch.tensor([[[0.0, 5.0, 9.0]]])

        radius_only = CylindricalCoordinateProjector(
            reference=reference,
            keep_mask=torch.tensor([[True, False, False]]),
            axis=torch.tensor([0.0, 0.0, 2.0]),
            center=torch.zeros(3),
        ).project(sampled)
        torch.testing.assert_close(
            radius_only,
            torch.tensor([[[0.0, 3.0, 9.0]]]),
        )

        azimuth_only = CylindricalCoordinateProjector(
            reference=reference,
            keep_mask=torch.tensor([[False, True, False]]),
            axis=torch.tensor([0.0, 0.0, 1.0]),
            center=torch.zeros(3),
        ).project(sampled)
        torch.testing.assert_close(
            azimuth_only,
            torch.tensor([[[5.0, 0.0, 9.0]]]),
        )

        axial_only = CylindricalCoordinateProjector(
            reference=reference,
            keep_mask=torch.tensor([[False, False, True]]),
            axis=torch.tensor([0.0, 0.0, 1.0]),
            center=torch.zeros(3),
        ).project(sampled)
        torch.testing.assert_close(
            axial_only,
            torch.tensor([[[0.0, 5.0, 2.0]]]),
        )

    def test_cylindrical_azimuth_rejects_axis_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "lies on the symmetry axis"):
            CylindricalCoordinateProjector(
                reference=torch.tensor([[[0.0, 0.0, 2.0]]]),
                keep_mask=torch.tensor([[False, True, False]]),
                axis=torch.tensor([0.0, 0.0, 1.0]),
                center=torch.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
