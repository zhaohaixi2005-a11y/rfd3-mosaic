import unittest

import torch

from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceEdge,
    GraphInterfaceGuidanceConfig,
    GraphInterfaceTopology,
    apply_graph_interface_guidance,
    build_graph_interface_topology,
    graph_interface_energy,
)


class GraphInterfaceGuidanceTestCase(unittest.TestCase):
    def _features(self, *, mode: int = 1, stage: str = "output"):
        return {
            "assembly_interface_left_membership": torch.tensor(
                [[True, False, False, False, False, False]]
            ),
            "assembly_interface_right_membership": torch.tensor(
                [[False, False, False, True, False, False]]
            ),
            "assembly_interface_mode": torch.tensor([mode]),
            "assembly_interface_required": torch.tensor([True]),
            "assembly_interface_minimum_contacts": torch.tensor([2]),
            "assembly_interface_contact_cutoff": torch.tensor([4.5]),
            "assembly_interface_distance_target": torch.tensor(
                [float("nan")]
            ),
            "assembly_interface_distance_tolerance": torch.tensor(
                [float("nan")]
            ),
            "assembly_interface_ids": ("designed_edge@0",),
            "assembly_interface_satisfaction_stages": (stage,),
            "atom_to_token_map": torch.arange(6),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "is_ca": torch.ones(6, dtype=torch.bool),
            "is_virtual": torch.zeros(6, dtype=torch.bool),
        }

    def test_only_output_stage_contact_edges_create_guidance(self) -> None:
        fixed = torch.tensor([True, False, False, True, False, False])

        self.assertIsNone(
            build_graph_interface_topology(
                self._features(mode=0),
                fixed,
            )
        )
        self.assertIsNone(
            build_graph_interface_topology(
                self._features(stage="input"),
                fixed,
            )
        )
        topology = build_graph_interface_topology(self._features(), fixed)
        self.assertIsNotNone(topology)
        self.assertEqual(len(topology.edges), 1)

    def test_output_contact_rejects_same_chain_self_distances(self) -> None:
        features = self._features()
        features["asym_id"] = torch.zeros(6, dtype=torch.long)
        fixed = torch.tensor([True, False, False, True, False, False])

        with self.assertRaisesRegex(ValueError, "distinct output chains"):
            build_graph_interface_topology(features, fixed)

    def test_guidance_moves_only_generated_tokens_towards_contact(self) -> None:
        features = self._features()
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [24.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
                [23.0, 0.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=10.0,
            pairs_per_edge=2,
            maximum_token_step=0.5,
        )
        initial = graph_interface_energy(coordinates, topology, config)
        guided, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=0.425,
            config=config,
        )
        final = graph_interface_energy(guided, topology, config)

        self.assertTrue(diagnostics["applied"])
        self.assertLess(final.attraction, initial.attraction)
        self.assertTrue(torch.equal(guided[:, fixed], coordinates[:, fixed]))
        self.assertLessEqual(diagnostics["maximum_token_step"], 0.500001)

    def test_guidance_repels_a_ca_overlap(self) -> None:
        features = self._features()
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        coordinates = torch.tensor(
            [[
                [-10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            clash_weight=10.0,
            pairs_per_edge=2,
            maximum_token_step=0.5,
        )
        initial = graph_interface_energy(coordinates, topology, config)
        # Keep weight non-zero only to activate the runtime step; attraction
        # is already zero because both selected distances are below target.
        config = GraphInterfaceGuidanceConfig(
            weight=1e-6,
            clash_weight=10.0,
            pairs_per_edge=2,
            maximum_token_step=0.5,
        )
        guided, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=0.425,
            config=config,
        )
        final = graph_interface_energy(guided, topology, config)

        self.assertTrue(diagnostics["applied"])
        self.assertLess(final.clash, initial.clash)

    def test_declared_com_distance_contributes_to_joint_energy(self) -> None:
        features = self._features()
        features["assembly_interface_minimum_contacts"] = torch.tensor([0])
        features["assembly_interface_distance_target"] = torch.tensor([6.0])
        features["assembly_interface_distance_tolerance"] = torch.tensor([0.5])
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [24.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
                [23.0, 0.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(distance_weight=1.0)
        energy = graph_interface_energy(coordinates, topology, config)

        self.assertGreater(float(energy.distance), 0.0)

    def test_auto_interface_size_uses_generated_residue_count(self) -> None:
        features = self._features()
        features["assembly_interface_minimum_contacts"] = torch.tensor([0])
        features["assembly_interface_automatic_quality"] = torch.tensor(
            [True]
        )
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [24.0, 0.0, 0.0],
                [22.0, 0.0, 0.0],
                [23.0, 0.0, 0.0],
            ]]
        )

        energy = graph_interface_energy(
            coordinates,
            topology,
            GraphInterfaceGuidanceConfig(),
        )

        self.assertEqual(energy.target_residues_per_side.tolist(), [2])
        self.assertEqual(
            energy.target_contiguous_residues_per_side.tolist(),
            [2],
        )
        self.assertGreater(float(energy.attraction), 0.0)

    def test_coverage_penalizes_one_sided_point_contact(self) -> None:
        features = self._features()
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        point_contact = torch.tensor(
            [[
                [-10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]]
        )
        broad_contact = point_contact.clone()
        broad_contact[:, 2, 0] = 1.0
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=1.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        point_energy = graph_interface_energy(
            point_contact,
            topology,
            config,
        )
        broad_energy = graph_interface_energy(
            broad_contact,
            topology,
            config,
        )

        self.assertGreater(float(point_energy.coverage), 0.0)
        self.assertEqual(float(broad_energy.coverage), 0.0)
        self.assertEqual(
            point_energy.covered_left_residues.tolist(),
            [1],
        )
        self.assertEqual(
            broad_energy.covered_left_residues.tolist(),
            [2],
        )

    def test_rejects_invalid_token_smoothing_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_smoothing_weight"):
            GraphInterfaceGuidanceConfig(token_smoothing_weight=1.1)
        with self.assertRaisesRegex(ValueError, "token_smoothing_passes"):
            GraphInterfaceGuidanceConfig(token_smoothing_passes=-1)

    def test_continuity_rejects_scattered_contact_coverage(self) -> None:
        features = self._features()
        features.update(
            {
                "assembly_interface_left_membership": torch.tensor(
                    [[True, False, False, False, False] + [False] * 5]
                ),
                "assembly_interface_right_membership": torch.tensor(
                    [[False] * 5 + [True, False, False, False, False]]
                ),
                "assembly_interface_minimum_contacts": torch.tensor([4]),
                "atom_to_token_map": torch.arange(10),
                "asym_id": torch.tensor([0] * 5 + [1] * 5),
                "is_ca": torch.ones(10, dtype=torch.bool),
                "is_virtual": torch.zeros(10, dtype=torch.bool),
            }
        )
        fixed = torch.tensor(
            [True, False, False, False, False]
            + [True, False, False, False, False]
        )
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        scattered = torch.tensor(
            [[
                [-20.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]]
        )
        contiguous = scattered.clone()
        contiguous[:, 2, 0] = 1.0
        contiguous[:, 3, 0] = 100.0
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=1.0,
            continuity_weight=1.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        scattered_energy = graph_interface_energy(
            scattered,
            topology,
            config,
        )
        contiguous_energy = graph_interface_energy(
            contiguous,
            topology,
            config,
        )

        self.assertEqual(float(scattered_energy.coverage), 0.0)
        self.assertGreater(float(scattered_energy.continuity), 0.0)
        self.assertEqual(float(contiguous_energy.continuity), 0.0)

    def test_source_interfaces_are_balanced_independent_of_copy_count(
        self,
    ) -> None:
        def mask(index: int) -> torch.Tensor:
            value = torch.zeros(6, dtype=torch.bool)
            value[index] = True
            return value

        def edge(
            edge_id: str,
            source_id: str,
            left: int,
            right: int,
        ) -> GraphInterfaceEdge:
            return GraphInterfaceEdge(
                edge_id=edge_id,
                source_interface_id=source_id,
                left_generated_ca_mask=mask(left),
                right_generated_ca_mask=mask(right),
                left_generated_token_ids=torch.tensor([left]),
                right_generated_token_ids=torch.tensor([right]),
                requested_contact_count=1,
                requested_residues_per_side=0,
                requested_contiguous_residues_per_side=0,
                automatic_quality=False,
                contact_cutoff=4.5,
                distance_target=None,
                distance_tolerance=None,
            )

        topology = GraphInterfaceTopology(
            edges=(
                edge("alpha@0", "alpha", 0, 1),
                edge("alpha@1", "alpha", 2, 3),
                edge("beta@0", "beta", 4, 5),
            ),
            generated_atom_mask=torch.ones(6, dtype=torch.bool),
        )
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [20.0, 10.0, 0.0],
                [0.0, 20.0, 0.0],
                [7.0, 20.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=1.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            pairs_per_edge=1,
        )

        energy = graph_interface_energy(coordinates, topology, config)

        self.assertAlmostEqual(
            float(energy.total),
            float(energy.per_edge_total[0]) / 2.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
