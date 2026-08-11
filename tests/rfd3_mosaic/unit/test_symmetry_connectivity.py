import unittest

from rfd3_mosaic.geometry import build_transform_registry
from rfd3_mosaic.topology.symmetry_connectivity import (
    finite_symmetry_spec,
    generated_transform_ids,
    minimal_group_relations,
)


class SymmetryConnectivityTestCase(unittest.TestCase):
    def test_c_d_and_polyhedral_groups_get_complete_generators(self) -> None:
        for symmetry_id in ("C3", "C7", "D3", "D5", "T", "O", "I"):
            with self.subTest(symmetry=symmetry_id):
                registry = build_transform_registry(
                    finite_symmetry_spec(symmetry_id)
                )
                relations = minimal_group_relations(symmetry_id)

                self.assertLessEqual(len(relations), 3)
                self.assertEqual(
                    generated_transform_ids(symmetry_id, relations),
                    registry.transform_ids,
                )

    def test_cyclic_group_requires_one_relation(self) -> None:
        self.assertEqual(len(minimal_group_relations("C12")), 1)

    def test_non_group_relation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown relations"):
            generated_transform_ids("T", ("T:not-a-transform",))

    def test_generator_budget_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 1"):
            minimal_group_relations("T", maximum_generators=1)


if __name__ == "__main__":
    unittest.main()
