import unittest

import numpy as np

from rfd3_mosaic.functional_geometry import evaluate_functional_geometry
from rfd3_mosaic.schema.functional_geometry import FunctionalGeometrySpec


def _specification(relations: list[dict]) -> FunctionalGeometrySpec:
    return FunctionalGeometrySpec.model_validate(
        {
            "schema_version": 1,
            "name": "geometry-test",
            "input": "unused.pdb",
            "fragments": [
                {"id": "f1", "selector": "chain A"},
                {"id": "f2", "selector": "chain B"},
                {"id": "f3", "selector": "chain C"},
            ],
            "atoms": [
                {"id": "a", "fragment": "f1", "selector": "a"},
                {"id": "b", "fragment": "f1", "selector": "b"},
                {"id": "c", "fragment": "f2", "selector": "c"},
                {"id": "d", "fragment": "f3", "selector": "d"},
            ],
            "relations": relations,
        }
    )


class FunctionalGeometryEvaluationTestCase(unittest.TestCase):
    def test_distance_angle_and_chirality_pass(self) -> None:
        specification = _specification(
            [
                {
                    "kind": "distance",
                    "id": "distance",
                    "atoms": ["a", "b"],
                    "target": 1.0,
                    "tolerance": 0.01,
                },
                {
                    "kind": "angle",
                    "id": "angle",
                    "atoms": ["a", "b", "c"],
                    "target_deg": 90.0,
                    "tolerance_deg": 0.1,
                },
                {
                    "kind": "chirality",
                    "id": "handedness",
                    "atoms": ["b", "a", "c", "d"],
                    "sign": "positive",
                    "minimum_abs_volume": 0.5,
                },
            ]
        )
        coordinates = {
            "a": (1.0, 0.0, 0.0),
            "b": (0.0, 0.0, 0.0),
            "c": (0.0, 1.0, 0.0),
            "d": (0.0, 0.0, 1.0),
        }

        report = evaluate_functional_geometry(specification, coordinates)

        self.assertTrue(report.passed)
        self.assertEqual(report.maximum_normalized_violation, 0.0)

    def test_coordination_is_one_joint_hyperedge_evaluation(self) -> None:
        specification = _specification(
            [
                {
                    "kind": "coordination",
                    "id": "three_way_site",
                    "center": "a",
                    "ligands": ["b", "c", "d"],
                    "shape": "trigonal_planar",
                    "distance_target": 2.0,
                    "distance_tolerance": 0.05,
                    "angle_tolerance_deg": 1.0,
                }
            ]
        )
        root_three = np.sqrt(3.0)
        coordinates = {
            "a": (0.0, 0.0, 0.0),
            "b": (2.0, 0.0, 0.0),
            "c": (-1.0, root_three, 0.0),
            "d": (-1.0, -root_three, 0.0),
        }

        report = evaluate_functional_geometry(specification, coordinates)

        self.assertTrue(report.passed)
        self.assertEqual(report.evaluated_relations, 1)
        self.assertEqual(report.relations[0].kind, "coordination")
        self.assertEqual(
            len(report.relations[0].observed["pairwise_angles_deg"]),
            3,
        )

    def test_distorted_coordination_fails_with_nonnegative_violation(self) -> None:
        specification = _specification(
            [
                {
                    "kind": "coordination",
                    "id": "three_way_site",
                    "center": "a",
                    "ligands": ["b", "c", "d"],
                    "shape": "trigonal_planar",
                    "distance_target": 2.0,
                    "distance_tolerance": 0.05,
                    "angle_tolerance_deg": 1.0,
                }
            ]
        )
        coordinates = {
            "a": (0.0, 0.0, 0.0),
            "b": (2.0, 0.0, 0.0),
            "c": (0.0, 2.0, 0.0),
            "d": (-2.0, 0.0, 0.0),
        }

        report = evaluate_functional_geometry(specification, coordinates)

        self.assertFalse(report.passed)
        self.assertGreater(report.maximum_normalized_violation, 0.0)

    def test_relative_pose_uses_second_fragment_in_first_frame(self) -> None:
        identity = np.eye(4)
        translated = np.eye(4)
        translated[0, 3] = 3.0
        specification = _specification(
            [
                {
                    "kind": "relative_pose",
                    "id": "pose",
                    "fragments": ["f1", "f2"],
                    "target_transform": translated.tolist(),
                    "translation_tolerance": 0.01,
                    "rotation_tolerance_deg": 0.01,
                }
            ]
        )
        coordinates = {
            "a": (0.0, 0.0, 0.0),
            "b": (1.0, 0.0, 0.0),
            "c": (0.0, 1.0, 0.0),
            "d": (0.0, 0.0, 1.0),
        }

        report = evaluate_functional_geometry(
            specification,
            coordinates,
            fragment_transforms={"f1": identity, "f2": translated},
        )

        self.assertTrue(report.passed)

    def test_missing_atom_coordinate_is_rejected(self) -> None:
        specification = _specification(
            [
                {
                    "kind": "distance",
                    "id": "distance",
                    "atoms": ["a", "b"],
                    "target": 1.0,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "Missing functional atom"):
            evaluate_functional_geometry(
                specification,
                {"a": (0.0, 0.0, 0.0)},
            )


if __name__ == "__main__":
    unittest.main()
