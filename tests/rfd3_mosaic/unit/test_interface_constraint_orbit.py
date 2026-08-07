import unittest

import torch

from rfd3.inference.symmetry.constraint_orbit import (
    ConstraintOrbitLayout,
)
from rfd3.inference.symmetry.interface_constraint_orbit import (
    InterfaceConstraintOrbitLayout,
)


def _features():
    membership = torch.zeros((3, 6), dtype=torch.bool)
    membership[0, [0, 1]] = True
    membership[1, [2, 3]] = True
    membership[2, [4, 5]] = True
    return {
        "motif_constraint_group_atom_indices": torch.tensor(
            [[0, 1], [2, 3], [4, 5]]
        ),
        "motif_constraint_group_atom_mask": torch.ones(
            (3, 2), dtype=torch.bool
        ),
        "motif_constraint_group_orbit_index": torch.tensor([0, 0, 0]),
        "motif_constraint_group_orbit_transform_id": torch.tensor(
            [0, 1, 2]
        ),
        "motif_constraint_orbit_master_group_index": torch.tensor([0]),
        "motif_constraint_orbit_mobility_mode": torch.tensor([1]),
        "motif_constraint_orbit_bounds": torch.tensor([[1.0, 5.0]]),
        "motif_constraint_orbit_subspace": torch.tensor([4]),
        "motif_constraint_orbit_proposal": torch.tensor([2]),
        "motif_constraint_orbit_schedule": torch.tensor(
            [[0.05, 0.70, 0.20, 0.15, 0.75]]
        ),
        "motif_constraint_orbit_objective_ids": (
            ("junction", "assembly_clash"),
        ),
        "motif_constraint_orbit_ids": ("mobile_orbit",),
        "motif_constraint_orbit_component_ids": ("mobile_component",),
        "motif_constraint_group_membership": membership,
        "sym_transform": {
            0: (torch.eye(3), torch.zeros(3)),
            1: (torch.eye(3), torch.zeros(3)),
            2: (torch.eye(3), torch.zeros(3)),
        },
    }


class InterfaceConstraintOrbitLayoutTestCase(unittest.TestCase):
    def test_neutral_api_and_legacy_name_resolve_same_layout(self) -> None:
        self.assertIs(ConstraintOrbitLayout, InterfaceConstraintOrbitLayout)

    def test_resolves_one_mobile_cross_protomer_orbit(self) -> None:
        layout = InterfaceConstraintOrbitLayout.from_features(
            _features(), atom_count=6
        )

        self.assertIsNotNone(layout)
        self.assertEqual(len(layout.groups), 3)
        self.assertEqual(len(layout.orbits), 1)
        self.assertEqual(len(layout.mobile_orbits), 1)
        orbit = layout.orbits[0]
        self.assertEqual(orbit.group_indices, (0, 1, 2))
        self.assertEqual(orbit.transform_ids, (0, 1, 2))
        self.assertEqual(orbit.mobility_mode, "orbit_rigid")
        self.assertEqual(orbit.constraint_orbit_id, "mobile_orbit")
        self.assertEqual(orbit.coupling_group_id, "mobile_component")
        self.assertEqual(orbit.maximum_translation, 1.0)
        self.assertEqual(orbit.maximum_rotation_degrees, 5.0)
        self.assertEqual(orbit.mobility_subspace, "bounded_se3")
        self.assertEqual(orbit.proposal_source, "scaffold_objectives")
        self.assertEqual(
            orbit.objective_ids,
            ("junction", "assembly_clash"),
        )
        self.assertAlmostEqual(orbit.schedule[0], 0.05, places=6)
        self.assertAlmostEqual(orbit.schedule[4], 0.75, places=6)

    def test_rejects_partial_feature_transport(self) -> None:
        features = _features()
        del features["motif_constraint_orbit_bounds"]

        with self.assertRaisesRegex(ValueError, "incomplete"):
            InterfaceConstraintOrbitLayout.from_features(
                features, atom_count=6
            )

    def test_rejects_membership_slot_disagreement(self) -> None:
        features = _features()
        features["motif_constraint_group_membership"][0, 1] = False
        features["motif_constraint_group_membership"][0, 2] = True

        with self.assertRaisesRegex(ValueError, "disagree"):
            InterfaceConstraintOrbitLayout.from_features(
                features, atom_count=6
            )

    def test_rejects_mobile_overlap_with_another_orbit(self) -> None:
        features = _features()
        features["motif_constraint_group_atom_indices"] = torch.tensor(
            [[0, 1], [2, 3], [1, 4]]
        )
        membership = torch.zeros((3, 6), dtype=torch.bool)
        membership[0, [0, 1]] = True
        membership[1, [2, 3]] = True
        membership[2, [1, 4]] = True
        features["motif_constraint_group_membership"] = membership
        features["motif_constraint_group_orbit_index"] = torch.tensor(
            [0, 0, 1]
        )
        features["motif_constraint_orbit_master_group_index"] = (
            torch.tensor([0, 2])
        )
        features["motif_constraint_orbit_mobility_mode"] = torch.tensor(
            [1, 0]
        )
        features["motif_constraint_orbit_bounds"] = torch.tensor(
            [[1.0, 5.0], [0.0, 0.0]]
        )
        features["motif_constraint_orbit_subspace"] = torch.tensor(
            [4, 0]
        )
        features["motif_constraint_orbit_proposal"] = torch.tensor(
            [2, 0]
        )
        features["motif_constraint_orbit_schedule"] = torch.tensor(
            [
                [0.05, 0.70, 0.20, 0.15, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        features["motif_constraint_orbit_objective_ids"] = (
            ("junction", "assembly_clash"),
            (),
        )
        features["motif_constraint_orbit_ids"] = (
            "mobile_orbit",
            "fixed_orbit",
        )
        features["motif_constraint_orbit_component_ids"] = (
            "mobile_component",
            "fixed_component",
        )

        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            InterfaceConstraintOrbitLayout.from_features(
                features, atom_count=6
            )


if __name__ == "__main__":
    unittest.main()
