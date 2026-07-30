import math
import unittest

import torch

from rfd3.inference.symmetry.motif_mobility import (
    OrbitRigidMotifController,
    fit_centered_rigid_pose,
    mobility_window_weight,
)


def _z_rotation(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _apply(points, rotation, translation):
    return points @ rotation.T + translation


class MotifMobilityTestCase(unittest.TestCase):
    def test_schedule_freezes_before_and_after_window(self) -> None:
        self.assertEqual(
            mobility_window_weight(
                0.05,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )
        self.assertGreater(
            mobility_window_weight(
                0.50,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )
        self.assertEqual(
            mobility_window_weight(
                0.90,
                start_fraction=0.10,
                end_fraction=0.85,
            ),
            0.0,
        )

    def test_rigid_fit_recovers_known_pose_without_reflection(self) -> None:
        template = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        rotation = _z_rotation(17.0)
        center = template.mean(dim=1, keepdim=True)
        translation = torch.tensor(
            [[[1.0, -0.5, 0.25]]],
            dtype=torch.float64,
        )
        proposal = (
            (template - center) @ rotation.T
            + center
            + translation
        )

        fitted_rotation, fitted_translation, rmsd = (
            fit_centered_rigid_pose(template, proposal)
        )

        self.assertTrue(
            torch.allclose(
                fitted_rotation[0],
                rotation,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                fitted_translation,
                translation[:, 0, :],
                atol=1e-6,
            )
        )
        self.assertLess(float(rmsd.max()), 1e-8)
        self.assertGreater(float(torch.linalg.det(fitted_rotation[0])), 0.0)

    @staticmethod
    def _controller_case():
        template = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        transforms = {
            str(transform_id): (
                _z_rotation(120.0 * transform_id),
                torch.zeros(3, dtype=torch.float64),
            )
            for transform_id in range(3)
        }
        target = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            target[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(
                    template,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
            )
        features = {
            "sym_transform": transforms,
            "motif_constraint_group_orbit_index": torch.tensor(
                [0, 0, 0]
            ),
            "motif_constraint_group_orbit_transform_id": torch.tensor(
                [0, 1, 2]
            ),
            "motif_constraint_group_atom_indices": torch.tensor(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                ]
            ),
            "motif_constraint_group_atom_mask": torch.ones(
                (3, 4),
                dtype=torch.bool,
            ),
            "motif_constraint_orbit_master_group_index": torch.tensor(
                [0]
            ),
            "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
            "motif_constraint_orbit_bounds": torch.tensor(
                [[2.0, 10.0]]
            ),
        }
        controller = OrbitRigidMotifController.from_features(
            features,
            target,
            start_fraction=0.0,
            end_fraction=1.0,
            response=1.0,
            per_step_translation=0.25,
            per_step_rotation_degrees=1.0,
        )
        assert controller is not None
        return template, target, transforms, controller

    def test_controller_moves_one_master_pose_with_bounded_c3_copies(
        self,
    ) -> None:
        template, _, transforms, controller = self._controller_case()
        desired_rotation = _z_rotation(30.0)
        center = template.mean(dim=1, keepdim=True)
        desired_master = (
            (template - center) @ desired_rotation.T
            + center
            + torch.tensor(
                [[[5.0, 0.0, 0.0]]],
                dtype=torch.float64,
            )
        )
        raw = torch.empty((1, 12, 3), dtype=torch.float64)
        for transform_id in range(3):
            raw[:, 4 * transform_id : 4 * (transform_id + 1)] = (
                _apply(
                    desired_master,
                    transforms[str(transform_id)][0],
                    transforms[str(transform_id)][1],
                )
            )

        target = controller.update(raw, progress=0.5)
        state = controller.motifs[0].state
        angle = torch.acos(
            torch.clamp(
                (torch.trace(state.rotation[0]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        self.assertLessEqual(
            math.degrees(float(angle)),
            1.0 + 1e-5,
        )
        self.assertAlmostEqual(
            math.degrees(float(angle)),
            1.0,
            places=5,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            0.25 + 1e-6,
        )
        self.assertTrue(
            torch.allclose(
                state.translation[0],
                torch.tensor(
                    [0.25, 0.0, 0.0],
                    dtype=torch.float64,
                ),
                atol=1e-6,
            )
        )

        master = target[:, :4]
        reference_distances = torch.cdist(master, master)
        for transform_id in range(3):
            group = target[
                :,
                4 * transform_id : 4 * (transform_id + 1),
            ]
            inverse = (
                group - transforms[str(transform_id)][1]
            ) @ transforms[str(transform_id)][0]
            self.assertTrue(torch.allclose(inverse, master, atol=1e-6))
            self.assertTrue(
                torch.allclose(
                    torch.cdist(group, group),
                    reference_distances,
                    atol=1e-6,
                )
            )

        for _ in range(20):
            target = controller.update(raw, progress=0.5)
        state = controller.motifs[0].state
        cumulative_angle = torch.acos(
            torch.clamp(
                (torch.trace(state.rotation[0]) - 1.0) / 2.0,
                -1.0,
                1.0,
            )
        )
        self.assertLessEqual(
            math.degrees(float(cumulative_angle)),
            10.0 + 1e-5,
        )
        self.assertAlmostEqual(
            math.degrees(float(cumulative_angle)),
            10.0,
            places=5,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            2.0 + 1e-6,
        )
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(state.translation[0])),
            2.0,
            places=6,
        )

        state.rotation[0] = _z_rotation(3.0)
        state.translation[0] = torch.tensor(
            [0.5, 0.0, 0.0],
            dtype=torch.float64,
        )
        frozen_rotation = state.rotation.clone()
        frozen_translation = state.translation.clone()
        controller.update(raw, progress=1.0)
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.rotation,
                frozen_rotation,
            )
        )
        self.assertTrue(
            torch.equal(
                controller.motifs[0].state.translation,
                frozen_translation,
            )
        )

    def test_controller_rejects_nonfinite_proposal(self) -> None:
        _, target, _, controller = self._controller_case()
        target[:, 0, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            controller.update(target, progress=0.5)


if __name__ == "__main__":
    unittest.main()
