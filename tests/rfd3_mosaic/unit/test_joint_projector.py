import unittest

import torch

from rfd3.inference.symmetry.joint_projector import UnifiedJointProjector


class UnifiedJointProjectorTestCase(unittest.TestCase):
    def test_projects_then_restores_then_validates(self) -> None:
        events = []

        def project(coordinates):
            events.append("project")
            return coordinates + 10.0

        def restore(coordinates, target, mask):
            events.append("restore")
            result = coordinates.clone()
            result[:, mask, :] = target[:, mask, :]
            return result

        def validate(coordinates, label):
            events.append(("validate", label, coordinates.clone()))

        projector = UnifiedJointProjector(project, restore, validate)
        coordinates = torch.zeros((1, 3, 3))
        target = torch.full((1, 3, 3), 2.0)
        mask = torch.tensor([True, False, True])

        result = projector.project(
            coordinates,
            constraint_target=target,
            constraint_mask=mask,
            restore=True,
            label="test state",
        )

        self.assertEqual(events[:2], ["project", "restore"])
        self.assertEqual(events[2][0:2], ("validate", "test state"))
        self.assertTrue(torch.all(result[:, mask, :] == 2.0))
        self.assertTrue(torch.all(result[:, ~mask, :] == 10.0))

    def test_can_project_without_restoring_constraints(self) -> None:
        restore_calls = []
        projector = UnifiedJointProjector(
            project_symmetry=lambda coordinates: coordinates + 1.0,
            restore_constraints=lambda coordinates, target, mask: (
                restore_calls.append(True) or coordinates
            ),
            validate_closure=lambda coordinates, label: None,
        )

        result = projector.project(
            torch.zeros((1, 1, 3)),
            constraint_target=torch.zeros((1, 1, 3)),
            constraint_mask=torch.tensor([True]),
            restore=False,
            label="projection only",
        )

        self.assertEqual(restore_calls, [])
        self.assertTrue(torch.all(result == 1.0))


if __name__ == "__main__":
    unittest.main()
