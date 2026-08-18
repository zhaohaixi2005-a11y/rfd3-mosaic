import unittest

from rfd3_mosaic.topology.component_incidence import (
    enumerate_binary_interface_incidence_plans,
)


class ComponentIncidenceTestCase(unittest.TestCase):
    def _assert_valency_pair(
        self,
        symmetry: str,
        expected_edges: int,
        valencies: tuple[int, int],
        component_counts: tuple[int, int],
    ) -> None:
        plans = enumerate_binary_interface_incidence_plans(
            symmetry=symmetry,
            interface_id="natural_interface",
            left_participant="A",
            right_participant="B",
        )
        selected = [
            plan for plan in plans
            if (plan.left.valency, plan.right.valency) == valencies
        ]
        self.assertTrue(selected)
        for plan in selected:
            self.assertEqual(plan.physical_interface_count, expected_edges)
            self.assertEqual(len(plan.physical_edges), expected_edges)
            self.assertEqual(
                (
                    plan.left.physical_component_count,
                    plan.right.physical_component_count,
                ),
                component_counts,
            )
            self.assertFalse(plan.executable)
            self.assertEqual(plan.edge_stabilizer_order, 1)
            self.assertEqual(plan.left.interface_degree, plan.left.valency)
            self.assertEqual(plan.right.interface_degree, plan.right.valency)

    def test_tetrahedral_c2_c3_incidence(self) -> None:
        self._assert_valency_pair("T", 12, (2, 3), (6, 4))

    def test_octahedral_c2_c4_incidence(self) -> None:
        self._assert_valency_pair("O", 24, (2, 4), (12, 6))

    def test_icosahedral_c2_c5_incidence(self) -> None:
        self._assert_valency_pair("I", 60, (2, 5), (30, 12))

    def test_tetrahedral_c2_c2_quotient_interface_orbit(self) -> None:
        plans = enumerate_binary_interface_incidence_plans(
            symmetry="T",
            interface_id="natural_interface",
            left_participant="A",
            right_participant="B",
            physical_interface_count=6,
            minimum_valency=2,
            maximum_valency=2,
        )
        selected = [
            plan
            for plan in plans
            if plan.left.physical_component_count == 6
            and plan.right.physical_component_count == 6
            and plan.left.interface_degree == 1
            and plan.right.interface_degree == 1
        ]

        self.assertTrue(selected)
        for plan in selected:
            self.assertEqual(plan.edge_stabilizer_order, 2)
            self.assertEqual(len(plan.physical_edges), 6)
            self.assertEqual(
                {len(actions) for _, _, actions in plan.physical_edge_actions},
                {2},
            )
            self.assertEqual(
                {
                    action
                    for _, _, actions in plan.physical_edge_actions
                    for action in actions
                },
                {
                    action
                    for action, _ in plan.left.action.transform_to_coset_representative
                },
            )

    def test_nonuniform_edge_orbit_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniform transitive edge orbit"):
            enumerate_binary_interface_incidence_plans(
                symmetry="T",
                interface_id="natural_interface",
                left_participant="A",
                right_participant="B",
                physical_interface_count=5,
            )


if __name__ == "__main__":
    unittest.main()
