import unittest

from rfd3_mosaic.topology.stabilizer_cosets import (
    stabilizer_coset_hypotheses,
    supported_orbit_sizes,
)


class StabilizerCosetTestCase(unittest.TestCase):
    def test_tetrahedral_three_instance_orbit_uses_order_four_stabilizer(self):
        hypotheses = stabilizer_coset_hypotheses("T", 3)

        self.assertTrue(hypotheses)
        for hypothesis in hypotheses:
            self.assertEqual(hypothesis.group_order, 12)
            self.assertEqual(hypothesis.orbit_size, 3)
            self.assertEqual(hypothesis.stabilizer_order, 4)
            self.assertEqual(len(hypothesis.cosets), 3)
            self.assertEqual(
                {item for coset in hypothesis.cosets for item in coset},
                {item for item, _ in hypothesis.transform_to_coset_representative},
            )

    def test_d3_three_instance_orbit_uses_order_two_stabilizer(self):
        hypotheses = stabilizer_coset_hypotheses("D3", 3)

        self.assertTrue(hypotheses)
        self.assertTrue(
            all(item.stabilizer_order == 2 for item in hypotheses)
        )

    def test_full_orbit_uses_identity_stabilizer(self):
        hypothesis = stabilizer_coset_hypotheses("I", 60)

        self.assertEqual(len(hypothesis), 1)
        self.assertEqual(hypothesis[0].stabilizer_order, 1)
        self.assertEqual(len(hypothesis[0].cosets), 60)

    def test_divisibility_without_subgroup_is_rejected(self):
        # A4 (the rotational tetrahedral group) has no subgroup of order 6,
        # therefore it has no transitive orbit of size 2.
        self.assertEqual(stabilizer_coset_hypotheses("T", 2), ())
        self.assertNotIn(2, supported_orbit_sizes("T"))

    def test_invalid_orbit_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            stabilizer_coset_hypotheses("C3", 0)


if __name__ == "__main__":
    unittest.main()
