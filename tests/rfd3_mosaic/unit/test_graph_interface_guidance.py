import unittest
from dataclasses import replace

import torch
from rfd3.inference.symmetry.graph_interface_guidance import (
    GraphInterfaceEdge,
    GraphInterfaceGuidanceConfig,
    GraphInterfacePatchState,
    GraphInterfaceTopology,
    adaptive_graph_interface_phase,
    apply_graph_interface_guidance,
    build_graph_interface_topology,
    build_symmetric_scaffold_interface_topology,
    graph_interface_capacity_preflight,
    graph_interface_energy,
    graph_interface_proposal_acceptable,
    graph_interface_quality_satisfied,
    guidance_window_weight,
    rf_contact_prior_schedule_scale,
    rf_oligomer_contact_prior,
    scheduled_interface_ca_distance,
)


class GraphInterfaceGuidanceTestCase(unittest.TestCase):
    @staticmethod
    def _three_by_three_topology() -> GraphInterfaceTopology:
        left = torch.tensor([True, True, True, False, False, False])
        right = ~left
        edge = GraphInterfaceEdge(
            edge_id="packing@0",
            source_interface_id="packing",
            left_generated_ca_mask=left,
            right_generated_ca_mask=right,
            left_generated_token_ids=torch.tensor([0, 1, 2]),
            right_generated_token_ids=torch.tensor([3, 4, 5]),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=2,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        return GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.ones(6, dtype=torch.bool),
        )

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
            "assembly_interface_distance_target": torch.tensor([0.0]),
            "assembly_interface_distance_tolerance": torch.tensor([0.0]),
            "assembly_interface_has_distance_target": torch.tensor([False]),
            "assembly_interface_ids": ("designed_edge@0",),
            "assembly_interface_satisfaction_stages": (stage,),
            "atom_to_token_map": torch.arange(6),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "is_ca": torch.ones(6, dtype=torch.bool),
            "is_virtual": torch.zeros(6, dtype=torch.bool),
        }

    def test_rf_contact_prior_is_finite_at_switch_midpoint(self) -> None:
        energy = rf_oligomer_contact_prior(
            torch.tensor([[10.0]]),
            r_0=8.0,
            d_0=2.0,
            normalization=1,
        )

        self.assertTrue(torch.isfinite(energy))
        self.assertAlmostEqual(float(energy), -0.5, places=6)

    def test_rf_contact_prior_prefers_contact_and_decays_early(self) -> None:
        close = rf_oligomer_contact_prior(
            torch.full((3, 3), 7.0),
            r_0=8.0,
            d_0=2.0,
            normalization=3,
        )
        far = rf_oligomer_contact_prior(
            torch.full((3, 3), 20.0),
            r_0=8.0,
            d_0=2.0,
            normalization=3,
        )
        config = GraphInterfaceGuidanceConfig(
            contact_prior_guide_scale=2.0,
            contact_prior_decay_power=2.0,
        )

        self.assertLess(float(close), float(far))
        self.assertAlmostEqual(
            rf_contact_prior_schedule_scale(0.0, config), 2.0
        )
        self.assertAlmostEqual(
            rf_contact_prior_schedule_scale(0.5, config), 0.5
        )
        self.assertAlmostEqual(
            rf_contact_prior_schedule_scale(1.0, config), 0.0
        )

    def test_contact_prior_is_a_separate_broad_interface_objective(self) -> None:
        topology = self._three_by_three_topology()
        close = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
                [7.0, 7.6, 0.0],
            ]]
        )
        far = close.clone()
        far[:, 3:, 0] = 30.0
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            contact_prior_weight=0.1,
        )

        close_energy = graph_interface_energy(close, topology, config)
        far_energy = graph_interface_energy(far, topology, config)

        self.assertLess(float(close_energy.total), float(far_energy.total))
        self.assertEqual(close_energy.per_edge_contact_prior.shape, (1,))

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

    def test_c3_scaffold_packing_builds_three_cyclic_neighbour_edges(self) -> None:
        atoms_per_chain = 7
        atom_count = 3 * atoms_per_chain
        features = {
            "symmetry_id": "C3",
            "atom_to_token_map": torch.arange(atom_count),
            "asym_id": torch.repeat_interleave(
                torch.arange(3), atoms_per_chain
            ),
            "is_ca": torch.ones(atom_count, dtype=torch.bool),
            "is_virtual": torch.zeros(atom_count, dtype=torch.bool),
            "token_bonds": torch.zeros(
                (atom_count, atom_count), dtype=torch.bool
            ),
            "residue_index": torch.arange(atoms_per_chain).repeat(3),
        }
        fixed = torch.zeros(atom_count, dtype=torch.bool)
        fixed[::atoms_per_chain] = True

        topology = build_symmetric_scaffold_interface_topology(
            features,
            fixed,
        )

        self.assertEqual(len(topology.edges), 3)
        self.assertEqual(
            {edge.source_interface_id for edge in topology.edges},
            {"automatic_symmetric_scaffold_interface"},
        )
        self.assertTrue(all(edge.automatic_quality for edge in topology.edges))
        self.assertFalse(torch.any(topology.generated_atom_mask & fixed))

    def test_scaffold_packing_rejects_noncyclic_group(self) -> None:
        features = {
            "symmetry_id": "T",
            "atom_to_token_map": torch.arange(4),
            "asym_id": torch.tensor([0, 0, 1, 1]),
            "is_ca": torch.ones(4, dtype=torch.bool),
            "is_virtual": torch.zeros(4, dtype=torch.bool),
        }
        with self.assertRaisesRegex(NotImplementedError, "requires Cn"):
            build_symmetric_scaffold_interface_topology(
                features,
                torch.tensor([True, False, True, False]),
            )

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
        features["assembly_interface_has_distance_target"] = torch.tensor(
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
        with self.assertRaisesRegex(
            ValueError,
            "maximum_tangent_normal_cosine",
        ):
            GraphInterfaceGuidanceConfig(
                maximum_tangent_normal_cosine=1.1
            )
        with self.assertRaisesRegex(ValueError, "backbone_ca_distance"):
            GraphInterfaceGuidanceConfig(backbone_ca_distance=0.0)
        with self.assertRaisesRegex(ValueError, "terminal_weight_floor"):
            GraphInterfaceGuidanceConfig(terminal_weight_floor=1.1)
        with self.assertRaisesRegex(ValueError, "unsatisfied_step_fraction"):
            GraphInterfaceGuidanceConfig(unsatisfied_step_fraction=-0.1)
        with self.assertRaisesRegex(ValueError, "final_polish_steps"):
            GraphInterfaceGuidanceConfig(final_polish_steps=-1)
        with self.assertRaisesRegex(ValueError, "continuity_softness"):
            GraphInterfaceGuidanceConfig(continuity_softness=0.0)
        with self.assertRaisesRegex(ValueError, "patch_rigid_weight"):
            GraphInterfaceGuidanceConfig(patch_rigid_weight=1.1)
        with self.assertRaisesRegex(ValueError, "patch_blend_radius"):
            GraphInterfaceGuidanceConfig(patch_blend_radius=-1)
        with self.assertRaisesRegex(ValueError, "line_search_steps"):
            GraphInterfaceGuidanceConfig(line_search_steps=0)
        with self.assertRaisesRegex(ValueError, "patch_lock_fraction"):
            GraphInterfaceGuidanceConfig(patch_lock_fraction=1.1)
        with self.assertRaisesRegex(ValueError, "capture_ca_distance"):
            GraphInterfaceGuidanceConfig(capture_ca_distance=7.0)

    def test_terminal_guidance_floor_survives_late_diffusion(self) -> None:
        self.assertEqual(
            guidance_window_weight(
                0.01,
                start_fraction=0.05,
                end_fraction=0.80,
                terminal_weight_floor=0.35,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            guidance_window_weight(
                0.90,
                start_fraction=0.05,
                end_fraction=0.80,
                terminal_weight_floor=0.35,
            ),
            0.35,
        )
        self.assertAlmostEqual(
            guidance_window_weight(
                1.0,
                start_fraction=0.05,
                end_fraction=0.80,
                terminal_weight_floor=0.35,
            ),
            0.35,
        )

    def test_default_guidance_keeps_a_strong_late_quality_field(self) -> None:
        config = GraphInterfaceGuidanceConfig()

        self.assertEqual(config.continuity_weight, 1.0)
        self.assertEqual(config.terminal_weight_floor, 0.8)
        self.assertEqual(config.final_polish_steps, 12)
        self.assertAlmostEqual(
            guidance_window_weight(
                1.0,
                start_fraction=config.start_fraction,
                end_fraction=config.end_fraction,
                terminal_weight_floor=config.terminal_weight_floor,
            ),
            0.8,
        )
    def test_orientation_discourages_end_on_contact_patches(self) -> None:
        topology = self._three_by_three_topology()
        end_on = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]]
        )
        side_on = torch.tensor(
            [[
                [0.0, -3.8, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [7.0, -3.8, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=1.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        end_on_energy = graph_interface_energy(end_on, topology, config)
        side_on_energy = graph_interface_energy(side_on, topology, config)

        self.assertGreater(
            float(end_on_energy.orientation),
            float(side_on_energy.orientation),
        )
        self.assertAlmostEqual(float(side_on_energy.orientation), 0.0)

    def test_shape_term_prefers_uniform_contact_depth(self) -> None:
        topology = self._three_by_three_topology()
        uniform = torch.tensor(
            [[
                [0.0, -3.8, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [7.0, -3.8, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
            ]]
        )
        corrugated = uniform.clone()
        corrugated[:, 4, 0] = 10.0
        corrugated[:, 5, 0] = 13.0
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=1.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        uniform_energy = graph_interface_energy(uniform, topology, config)
        corrugated_energy = graph_interface_energy(
            corrugated,
            topology,
            config,
        )

        self.assertAlmostEqual(float(uniform_energy.shape), 0.0)
        self.assertGreater(
            float(corrugated_energy.shape),
            float(uniform_energy.shape),
        )

    def test_backbone_term_rejects_local_ca_collapse(self) -> None:
        topology = self._three_by_three_topology()
        regular = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [8.0, 0.0, 0.0],
                [8.0, 3.8, 0.0],
                [8.0, 7.6, 0.0],
            ]]
        )
        collapsed = regular.clone()
        collapsed[:, 1, 1] = 1.0
        collapsed[:, 2, 1] = 2.0
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=1.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        regular_energy = graph_interface_energy(regular, topology, config)
        collapsed_energy = graph_interface_energy(
            collapsed,
            topology,
            config,
        )

        self.assertAlmostEqual(float(regular_energy.backbone), 0.0)
        self.assertGreater(
            float(collapsed_energy.backbone),
            float(regular_energy.backbone),
        )

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

        self.assertGreater(float(scattered_energy.coverage), 0.0)
        self.assertLess(
            float(contiguous_energy.coverage),
            float(scattered_energy.coverage),
        )
        self.assertGreater(float(scattered_energy.continuity), 0.0)
        self.assertLess(
            float(contiguous_energy.continuity),
            float(scattered_energy.continuity),
        )

    def test_paired_patch_rejects_nonreciprocal_contiguous_contacts(
        self,
    ) -> None:
        left_mask = torch.tensor([True] * 7 + [False] * 7)
        edge = GraphInterfaceEdge(
            edge_id="paired@0",
            source_interface_id="paired",
            left_generated_ca_mask=left_mask,
            right_generated_ca_mask=~left_mask,
            left_generated_token_ids=torch.arange(7),
            right_generated_token_ids=torch.arange(7, 14),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=3,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.ones(14, dtype=torch.bool),
        )

        # Globally, L0-L2 and R0-R2 each have three adjacent residues with a
        # close partner.  Those partners live in mutually incompatible parts
        # of the opposing sequence, however, so there is no reciprocal 3x3
        # interface patch.  Independent left/right nearest-neighbour windows
        # incorrectly treated this arrangement as a complete interface.
        nonreciprocal = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 40.0, 0.0],
                [0.0, 80.0, 0.0],
                [0.0, 120.0, 0.0],
                [0.0, 300.0, 0.0],
                [0.0, 340.0, 0.0],
                [0.0, 160.0, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 120.0, 0.0],
                [7.0, 160.0, 0.0],
                [7.0, 40.0, 0.0],
                [7.0, 400.0, 0.0],
                [7.0, 440.0, 0.0],
                [7.0, 80.0, 0.0],
            ]]
        )
        coherent = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 120.0, 0.0],
                [0.0, 140.0, 0.0],
                [0.0, 160.0, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
                [7.0, 7.6, 0.0],
                [50.0, 100.0, 0.0],
                [50.0, 120.0, 0.0],
                [50.0, 140.0, 0.0],
                [50.0, 160.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=1.0,
            coverage_weight=1.0,
            continuity_weight=1.0,
            orientation_weight=0.25,
            shape_weight=0.5,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            pairs_per_edge=3,
        )

        nonreciprocal_energy = graph_interface_energy(
            nonreciprocal,
            topology,
            config,
        )
        coherent_energy = graph_interface_energy(
            coherent,
            topology,
            config,
        )

        self.assertGreater(
            float(nonreciprocal_energy.attraction),
            float(coherent_energy.attraction),
        )
        self.assertGreater(
            float(nonreciprocal_energy.coverage),
            float(coherent_energy.coverage),
        )
        self.assertGreater(
            float(nonreciprocal_energy.total),
            float(coherent_energy.total),
        )
        self.assertEqual(
            coherent_energy.covered_left_residues.tolist(),
            [3],
        )
        self.assertEqual(
            coherent_energy.covered_right_residues.tolist(),
            [3],
        )
        self.assertLess(
            nonreciprocal_energy.covered_left_residues.item(),
            3,
        )
        self.assertLess(
            nonreciprocal_energy.covered_right_residues.item(),
            3,
        )

        differentiable = nonreciprocal.clone().requires_grad_(True)
        graph_interface_energy(
            differentiable,
            topology,
            config,
        ).total.backward()
        self.assertIsNotNone(differentiable.grad)
        self.assertTrue(torch.isfinite(differentiable.grad).all())
        self.assertGreater(float(differentiable.grad.abs().sum()), 0.0)

    def test_paired_patch_does_not_hide_clashes_outside_selected_window(
        self,
    ) -> None:
        left_mask = torch.tensor([True] * 5 + [False] * 5)
        edge = GraphInterfaceEdge(
            edge_id="clash@0",
            source_interface_id="clash",
            left_generated_ca_mask=left_mask,
            right_generated_ca_mask=~left_mask,
            left_generated_token_ids=torch.arange(5),
            right_generated_token_ids=torch.arange(5, 10),
            requested_contact_count=2,
            requested_residues_per_side=2,
            requested_contiguous_residues_per_side=2,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.ones(10, dtype=torch.bool),
        )
        # The first two residues make the selected broad patch; the final CA
        # pair overlaps.  Clash energy must still inspect the complete edge.
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 140.0, 0.0],
                [20.0, 200.0, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
                [50.0, 100.0, 0.0],
                [50.0, 140.0, 0.0],
                [20.0, 200.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=1.0,
            coverage_weight=1.0,
            continuity_weight=1.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=8.0,
            distance_weight=0.0,
        )

        energy = graph_interface_energy(coordinates, topology, config)

        self.assertEqual(float(energy.minimum_distances.item()), 0.0)
        self.assertGreater(float(energy.clash), 0.0)

    def test_continuity_never_treats_disconnected_short_runs_as_one_patch(
        self,
    ) -> None:
        left_mask = torch.tensor([True] * 6 + [False] * 6)
        right_mask = ~left_mask
        edge = GraphInterfaceEdge(
            edge_id="disconnected@0",
            source_interface_id="disconnected",
            left_generated_ca_mask=left_mask,
            right_generated_ca_mask=right_mask,
            left_generated_token_ids=torch.tensor([0, 1, 3, 4, 6, 7]),
            right_generated_token_ids=torch.tensor(
                [10, 11, 13, 14, 16, 17]
            ),
            requested_contact_count=0,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=3,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.ones(12, dtype=torch.bool),
        )
        left = torch.stack(
            [torch.tensor([0.0, float(index), 0.0]) for index in range(6)]
        )
        right = left + torch.tensor([7.0, 0.0, 0.0])
        coordinates = torch.cat((left, right)).unsqueeze(0)
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=1.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        energy = graph_interface_energy(coordinates, topology, config)

        self.assertGreater(float(energy.continuity), 0.0)
        self.assertEqual(
            energy.target_contiguous_residues_per_side.tolist(),
            [3],
        )

        automatic_topology = GraphInterfaceTopology(
            edges=(
                replace(
                    edge,
                    requested_residues_per_side=0,
                    requested_contiguous_residues_per_side=0,
                    automatic_quality=True,
                ),
            ),
            generated_atom_mask=topology.generated_atom_mask,
        )
        automatic_energy = graph_interface_energy(
            coordinates,
            automatic_topology,
            config,
        )
        self.assertEqual(
            automatic_energy.target_contiguous_residues_per_side.tolist(),
            [2],
        )

    def test_unsatisfied_contract_uses_declared_trust_region(self) -> None:
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
            weight=1e-6,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            maximum_token_step=0.5,
            unsatisfied_step_fraction=0.5,
        )

        _, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=0.9,
            config=config,
        )

        self.assertFalse(diagnostics["quality_targets_satisfied"])
        self.assertGreater(diagnostics["gradient_boost"], 1.0)
        self.assertGreaterEqual(diagnostics["maximum_token_step"], 0.249)
        self.assertLessEqual(diagnostics["maximum_token_step"], 0.500001)

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
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
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

    def test_interface_balance_emphasizes_the_worst_source_interface(
        self,
    ) -> None:
        def mask(index: int) -> torch.Tensor:
            value = torch.zeros(4, dtype=torch.bool)
            value[index] = True
            return value

        def edge(source_id: str, left: int, right: int) -> GraphInterfaceEdge:
            return GraphInterfaceEdge(
                edge_id=f"{source_id}@0",
                source_interface_id=source_id,
                left_generated_ca_mask=mask(left),
                right_generated_ca_mask=mask(right),
                left_generated_token_ids=torch.tensor([left]),
                right_generated_token_ids=torch.tensor([right]),
                requested_contact_count=1,
                requested_residues_per_side=0,
                requested_contiguous_residues_per_side=0,
                automatic_quality=False,
                contact_cutoff=5.5,
                distance_target=None,
                distance_tolerance=None,
            )

        topology = GraphInterfaceTopology(
            edges=(edge("alpha", 0, 1), edge("beta", 2, 3)),
            generated_atom_mask=torch.ones(4, dtype=torch.bool),
        )
        imbalanced = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [8.0, 10.0, 0.0],
            ]]
        )
        balanced = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [14.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [14.0, 10.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=1.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=1.0,
            clash_weight=0.0,
            distance_weight=0.0,
            pairs_per_edge=1,
        )

        imbalanced_energy = graph_interface_energy(
            imbalanced,
            topology,
            config,
        )
        balanced_energy = graph_interface_energy(
            balanced,
            topology,
            config,
        )

        self.assertGreater(
            float(imbalanced_energy.interface_balance),
            float(balanced_energy.interface_balance),
        )

    def test_joint_acceptance_cannot_sacrifice_one_source_interface(
        self,
    ) -> None:
        def mask(index: int) -> torch.Tensor:
            value = torch.zeros(4, dtype=torch.bool)
            value[index] = True
            return value

        def edge(source_id: str, left: int, right: int) -> GraphInterfaceEdge:
            return GraphInterfaceEdge(
                edge_id=f"{source_id}@0",
                source_interface_id=source_id,
                left_generated_ca_mask=mask(left),
                right_generated_ca_mask=mask(right),
                left_generated_token_ids=torch.tensor([left]),
                right_generated_token_ids=torch.tensor([right]),
                requested_contact_count=1,
                requested_residues_per_side=0,
                requested_contiguous_residues_per_side=0,
                automatic_quality=False,
                contact_cutoff=5.5,
                distance_target=None,
                distance_tolerance=None,
            )

        topology = GraphInterfaceTopology(
            edges=(edge("alpha", 0, 1), edge("beta", 2, 3)),
            generated_atom_mask=torch.ones(4, dtype=torch.bool),
        )
        before_coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [8.0, 10.0, 0.0],
            ]]
        )
        after_coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [13.0, 10.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            pairs_per_edge=1,
        )
        before = graph_interface_energy(
            before_coordinates, topology, config
        )
        after = graph_interface_energy(
            after_coordinates, topology, config
        )

        self.assertLess(float(after.total), float(before.total))
        self.assertFalse(
            graph_interface_proposal_acceptable(before, after, config)
        )

    def test_coarse_to_fine_capture_contracts_without_user_targets(
        self,
    ) -> None:
        config = GraphInterfaceGuidanceConfig(
            start_fraction=0.1,
            end_fraction=0.8,
            target_ca_distance=8.0,
            capture_ca_distance=12.0,
        )

        self.assertEqual(scheduled_interface_ca_distance(0.0, config), 12.0)
        self.assertGreater(
            scheduled_interface_ca_distance(0.45, config),
            8.0,
        )
        self.assertEqual(scheduled_interface_ca_distance(0.9, config), 8.0)
        base_topology = self._three_by_three_topology()
        topology = GraphInterfaceTopology(
            edges=(replace(base_topology.edges[0], contact_cutoff=4.5),),
            generated_atom_mask=base_topology.generated_atom_mask,
        )
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [12.0, 0.0, 0.0],
                [12.0, 3.8, 0.0],
                [12.0, 7.6, 0.0],
            ]]
        )
        final_direct = graph_interface_energy(
            coordinates,
            topology,
            config,
        )
        final_scheduled = graph_interface_energy(
            coordinates,
            topology,
            config,
            target_ca_distance_override=scheduled_interface_ca_distance(
                1.0,
                config,
            ),
        )
        self.assertAlmostEqual(
            float(final_direct.total),
            float(final_scheduled.total),
            places=7,
        )

    def test_adaptive_phase_follows_observed_patch_quality(self) -> None:
        topology = self._three_by_three_topology()
        config = GraphInterfaceGuidanceConfig()
        far = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [14.0, 0.0, 0.0],
                [14.0, 3.8, 0.0],
                [14.0, 7.6, 0.0],
            ]]
        )
        narrow = far.clone()
        narrow[0, 3:, 0] = 8.5
        packed = far.clone()
        packed[0, 3:, 0] = 6.0

        self.assertEqual(
            adaptive_graph_interface_phase(
                graph_interface_energy(far, topology, config),
                config,
            ),
            "capture",
        )
        self.assertEqual(
            adaptive_graph_interface_phase(
                graph_interface_energy(narrow, topology, config),
                config,
            ),
            "expand",
        )
        self.assertEqual(
            adaptive_graph_interface_phase(
                graph_interface_energy(packed, topology, config),
                config,
            ),
            "polish",
        )

    def test_explicit_patch_capacity_fails_before_sampling(self) -> None:
        topology = self._three_by_three_topology()
        topology = replace(
            topology,
            edges=(
                replace(
                    topology.edges[0],
                    requested_residues_per_side=4,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "only 3/3"):
            graph_interface_energy(
                torch.zeros((1, 6, 3)),
                topology,
                GraphInterfaceGuidanceConfig(),
            )

    def test_capacity_preflight_rejects_overlapping_patch_demand(self) -> None:
        base = self._three_by_three_topology().edges[0]
        topology = GraphInterfaceTopology(
            edges=(
                replace(
                    base,
                    edge_id="alpha@0",
                    source_interface_id="alpha",
                    requested_residues_per_side=2,
                ),
                replace(
                    base,
                    edge_id="beta@0",
                    source_interface_id="beta",
                    requested_residues_per_side=2,
                ),
            ),
            generated_atom_mask=torch.ones(6, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "over-subscribed"):
            graph_interface_capacity_preflight(topology)

    def test_far_mobile_pose_is_capture_not_capacity_failure(self) -> None:
        topology = self._three_by_three_topology()
        records = graph_interface_capacity_preflight(topology)
        self.assertEqual(len(records), 1)

        # Capacity preflight deliberately has no coordinate input.  A poor
        # relative pose is therefore recoverable scientific search space,
        # not a reason to reject an otherwise movable interface.
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [80.0, 0.0, 0.0],
                [80.0, 3.8, 0.0],
                [80.0, 7.6, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig()
        self.assertEqual(
            adaptive_graph_interface_phase(
                graph_interface_energy(coordinates, topology, config),
                config,
            ),
            "capture",
        )

    def test_patch_rigid_step_preserves_local_backbone_and_decreases_energy(
        self,
    ) -> None:
        topology = self._three_by_three_topology()
        features = {
            "atom_to_token_map": torch.arange(6),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [14.0, 0.0, 0.0],
                [14.0, 3.8, 0.0],
                [14.0, 7.6, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            weight=1.0,
            coverage_weight=1.0,
            continuity_weight=1.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            patch_rigid_weight=1.0,
            patch_blend_radius=0,
            maximum_token_step=0.5,
        )
        before_internal = torch.cdist(coordinates[0, :3], coordinates[0, :3])
        before_energy = graph_interface_energy(coordinates, topology, config)

        guided, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=1.0,
            config=config,
        )
        after_internal = torch.cdist(guided[0, :3], guided[0, :3])
        after_energy = graph_interface_energy(guided, topology, config)

        self.assertTrue(diagnostics["proposal_accepted"])
        self.assertEqual(diagnostics["rigid_patch_count"], 2)
        self.assertLess(
            diagnostics["energy_after"],
            diagnostics["energy_before"],
        )
        self.assertLess(float(after_energy.total), float(before_energy.total))
        self.assertTrue(
            torch.allclose(before_internal, after_internal, atol=1e-5)
        )

    def test_line_search_evaluates_projected_state_and_rolls_back(self) -> None:
        topology = self._three_by_three_topology()
        features = {
            "atom_to_token_map": torch.arange(6),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [14.0, 0.0, 0.0],
                [14.0, 3.8, 0.0],
                [14.0, 7.6, 0.0],
            ]]
        )
        projected_worse = coordinates.clone()
        projected_worse[:, 3:, 0] = 30.0
        config = GraphInterfaceGuidanceConfig(
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )

        guided, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=1.0,
            config=config,
            projector=lambda _candidate: projected_worse,
        )

        self.assertFalse(diagnostics["proposal_accepted"])
        self.assertEqual(diagnostics["line_search_scale"], 0.0)
        self.assertTrue(torch.equal(guided, coordinates))

    def test_locked_patch_cannot_hop_to_an_easier_residue_window(self) -> None:
        left_mask = torch.tensor([True] * 6 + [False] * 6)
        right_mask = ~left_mask
        topology = GraphInterfaceTopology(
            edges=(
                GraphInterfaceEdge(
                    edge_id="stable@0",
                    source_interface_id="stable",
                    left_generated_ca_mask=left_mask,
                    right_generated_ca_mask=right_mask,
                    left_generated_token_ids=torch.arange(6),
                    right_generated_token_ids=torch.arange(6, 12),
                    requested_contact_count=3,
                    requested_residues_per_side=3,
                    requested_contiguous_residues_per_side=3,
                    automatic_quality=False,
                    contact_cutoff=4.5,
                    distance_target=None,
                    distance_tolerance=None,
                ),
            ),
            generated_atom_mask=torch.ones(12, dtype=torch.bool),
        )
        features = {
            "atom_to_token_map": torch.arange(12),
            "asym_id": torch.tensor([0] * 6 + [1] * 6),
            "residue_index": torch.tensor(list(range(6)) * 2),
        }
        initial = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 101.0, 0.0],
                [0.0, 102.0, 0.0],
                [9.0, 0.0, 0.0],
                [9.0, 1.0, 0.0],
                [9.0, 2.0, 0.0],
                [9.0, 200.0, 0.0],
                [9.0, 201.0, 0.0],
                [9.0, 202.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )
        state = GraphInterfacePatchState(assignments={})
        _, first = apply_graph_interface_guidance(
            initial,
            features,
            topology,
            progress=0.25,
            config=config,
            patch_state=state,
        )
        selected = first["patch_assignments"]["stable@0"]
        self.assertEqual(selected["left_token_ids"], [0, 1, 2])
        self.assertEqual(selected["right_token_ids"], [6, 7, 8])

        easier_elsewhere = initial.clone()
        easier_elsewhere[0, 6:9, 1] = torch.tensor([200.0, 201.0, 202.0])
        easier_elsewhere[0, 9:12, 1] = torch.tensor([100.0, 101.0, 102.0])
        state.locked = True
        _, second = apply_graph_interface_guidance(
            easier_elsewhere,
            features,
            topology,
            progress=0.75,
            config=config,
            patch_state=state,
        )

        self.assertTrue(second["patch_locked"])
        self.assertEqual(second["patch_assignments"]["stable@0"], selected)

    def test_uncaptured_patch_remains_adaptive_after_lock_fraction(self) -> None:
        left_mask = torch.tensor([True] * 6 + [False] * 6)
        right_mask = ~left_mask
        topology = GraphInterfaceTopology(
            edges=(
                GraphInterfaceEdge(
                    edge_id="adaptive@0",
                    source_interface_id="adaptive",
                    left_generated_ca_mask=left_mask,
                    right_generated_ca_mask=right_mask,
                    left_generated_token_ids=torch.arange(6),
                    right_generated_token_ids=torch.arange(6, 12),
                    requested_contact_count=3,
                    requested_residues_per_side=3,
                    requested_contiguous_residues_per_side=3,
                    automatic_quality=False,
                    contact_cutoff=4.5,
                    distance_target=None,
                    distance_tolerance=None,
                ),
            ),
            generated_atom_mask=torch.ones(12, dtype=torch.bool),
        )
        features = {
            "atom_to_token_map": torch.arange(12),
            "asym_id": torch.tensor([0] * 6 + [1] * 6),
            "residue_index": torch.tensor(list(range(6)) * 2),
        }
        initial = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 101.0, 0.0],
                [0.0, 102.0, 0.0],
                [9.0, 0.0, 0.0],
                [9.0, 1.0, 0.0],
                [9.0, 2.0, 0.0],
                [9.0, 200.0, 0.0],
                [9.0, 201.0, 0.0],
                [9.0, 202.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )
        state = GraphInterfacePatchState(assignments={})
        _, first = apply_graph_interface_guidance(
            initial,
            features,
            topology,
            progress=0.75,
            config=config,
            patch_state=state,
        )
        self.assertFalse(state.locked)
        self.assertEqual(
            first["patch_assignments"]["adaptive@0"]["left_token_ids"],
            [0, 1, 2],
        )

        moved = initial.clone()
        moved[0, 6:9, 1] = torch.tensor([200.0, 201.0, 202.0])
        moved[0, 9:12, 1] = torch.tensor([100.0, 101.0, 102.0])
        _, second = apply_graph_interface_guidance(
            moved,
            features,
            topology,
            progress=0.75,
            config=config,
            patch_state=state,
        )
        self.assertFalse(state.locked)
        self.assertEqual(
            second["patch_assignments"]["adaptive@0"]["right_token_ids"],
            [9, 10, 11],
        )

    def test_patch_locks_only_after_final_quality_capture(self) -> None:
        topology = self._three_by_three_topology()
        features = {
            "atom_to_token_map": torch.arange(6),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [6.0, 0.0, 0.0],
                [6.0, 1.0, 0.0],
                [6.0, 2.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )
        state = GraphInterfacePatchState(assignments={})
        _, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=0.75,
            config=config,
            patch_state=state,
        )
        self.assertTrue(state.locked)
        self.assertEqual(state.lock_reason, "quality_capture")
        self.assertTrue(diagnostics["patch_capture_satisfied"])

    def test_patch_se3_rotates_all_atoms_without_deforming_patch(self) -> None:
        atom_to_token = torch.arange(6).repeat_interleave(3)
        ca_indices = torch.arange(6) * 3 + 1
        left_ca_mask = torch.zeros(18, dtype=torch.bool)
        right_ca_mask = torch.zeros(18, dtype=torch.bool)
        left_ca_mask[ca_indices[:3]] = True
        right_ca_mask[ca_indices[3:]] = True
        edge = GraphInterfaceEdge(
            edge_id="rotate@0",
            source_interface_id="rotate",
            left_generated_ca_mask=left_ca_mask,
            right_generated_ca_mask=right_ca_mask,
            left_generated_token_ids=torch.tensor([0, 1, 2]),
            right_generated_token_ids=torch.tensor([3, 4, 5]),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=2,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.ones(18, dtype=torch.bool),
            guided_ca_mask=left_ca_mask | right_ca_mask,
        )
        ca_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [9.0, 0.5, 0.0],
                [10.0, 1.0, 0.0],
            ]
        )
        atom_points = torch.stack(
            [
                point + offset
                for point in ca_points
                for offset in (
                    torch.tensor([0.0, -0.5, 0.0]),
                    torch.zeros(3),
                    torch.tensor([0.0, 0.5, 0.0]),
                )
            ]
        ).unsqueeze(0)
        features = {
            "atom_to_token_map": atom_to_token,
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        }
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=1.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
            patch_rigid_weight=1.0,
            patch_blend_radius=0,
            maximum_patch_rotation_degrees=5.0,
        )
        before_distances = torch.cdist(
            atom_points[0, :9],
            atom_points[0, :9],
        )
        before = graph_interface_energy(atom_points, topology, config)

        guided, diagnostics = apply_graph_interface_guidance(
            atom_points,
            features,
            topology,
            progress=1.0,
            config=config,
        )
        after_distances = torch.cdist(guided[0, :9], guided[0, :9])
        after = graph_interface_energy(guided, topology, config)

        self.assertTrue(diagnostics["proposal_accepted"])
        self.assertGreater(
            diagnostics["maximum_patch_rotation_degrees"],
            0.0,
        )
        self.assertLessEqual(
            diagnostics["maximum_token_step"],
            config.maximum_token_step + 1e-6,
        )
        self.assertLess(float(after.orientation), float(before.orientation))
        self.assertTrue(
            torch.allclose(before_distances, after_distances, atol=1e-5)
        )

    def test_energy_decrease_cannot_buy_a_new_global_ca_clash(self) -> None:
        left = torch.tensor([True, True, True, False, False, False, False])
        right = torch.tensor([False, False, False, True, True, True, False])
        edge = GraphInterfaceEdge(
            edge_id="target@0",
            source_interface_id="target",
            left_generated_ca_mask=left,
            right_generated_ca_mask=right,
            left_generated_token_ids=torch.tensor([0, 1, 2]),
            right_generated_token_ids=torch.tensor([3, 4, 5]),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=2,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        exclusions = torch.zeros((6, 7), dtype=torch.bool)
        exclusions[torch.arange(6), torch.arange(6)] = True
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.tensor([True] * 6 + [False]),
            guided_ca_mask=torch.tensor([True] * 6 + [False]),
            safety_ca_mask=torch.ones(7, dtype=torch.bool),
            safety_exclusions=exclusions,
        )
        before_xyz = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [14.0, 0.0, 0.0],
                [14.0, 3.8, 0.0],
                [14.0, 7.6, 0.0],
                [30.0, 3.8, 0.0],
            ]]
        )
        after_xyz = before_xyz.clone()
        after_xyz[:, 3:, 0] = 7.0
        after_xyz[:, 6] = after_xyz[:, 1]
        config = GraphInterfaceGuidanceConfig(
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=0.0,
            distance_weight=0.0,
        )
        before = graph_interface_energy(before_xyz, topology, config)
        after = graph_interface_energy(after_xyz, topology, config)

        self.assertLess(float(after.total), float(before.total))
        self.assertFalse(
            graph_interface_proposal_acceptable(before, after, config)
        )

    def test_global_safety_field_repels_non_target_fixed_obstacle(self) -> None:
        left = torch.tensor([True, True, True, False, False, False, False])
        right = torch.tensor([False, False, False, True, True, True, False])
        edge = GraphInterfaceEdge(
            edge_id="target@0",
            source_interface_id="target",
            left_generated_ca_mask=left,
            right_generated_ca_mask=right,
            left_generated_token_ids=torch.tensor([0, 1, 2]),
            right_generated_token_ids=torch.tensor([3, 4, 5]),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=2,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        exclusions = torch.zeros((6, 7), dtype=torch.bool)
        exclusions[torch.arange(6), torch.arange(6)] = True
        topology = GraphInterfaceTopology(
            edges=(edge,),
            generated_atom_mask=torch.tensor(
                [True, True, True, True, True, True, False]
            ),
            guided_ca_mask=torch.tensor(
                [True, True, True, True, True, True, False]
            ),
            safety_ca_mask=torch.ones(7, dtype=torch.bool),
            safety_exclusions=exclusions,
        )
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [8.0, 0.0, 0.0],
                [8.0, 3.8, 0.0],
                [8.0, 7.6, 0.0],
                [0.2, 3.8, 0.0],
            ]]
        )
        features = {
            "atom_to_token_map": torch.arange(7),
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1, 2]),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2, 0]),
        }
        config = GraphInterfaceGuidanceConfig(
            weight=0.0,
            coverage_weight=0.0,
            continuity_weight=0.0,
            orientation_weight=0.0,
            shape_weight=0.0,
            backbone_weight=0.0,
            interface_balance_weight=0.0,
            patch_exclusivity_weight=0.0,
            clash_weight=10.0,
            distance_weight=0.0,
            patch_rigid_weight=0.0,
            maximum_token_step=0.5,
        )
        before = graph_interface_energy(coordinates, topology, config)

        guided, diagnostics = apply_graph_interface_guidance(
            coordinates,
            features,
            topology,
            progress=1.0,
            config=config,
        )
        after = graph_interface_energy(guided, topology, config)

        self.assertGreater(float(before.global_safety_clash), 0.0)
        self.assertTrue(diagnostics["proposal_accepted"])
        self.assertLess(
            float(after.global_safety_clash),
            float(before.global_safety_clash),
        )
        self.assertTrue(torch.equal(guided[:, 6], coordinates[:, 6]))

    def test_all_guided_peptide_edges_are_in_backbone_contract(
        self,
    ) -> None:
        features = self._features()
        features["residue_index"] = torch.tensor([0, 1, 2, 0, 1, 2])
        fixed = torch.tensor([True, False, False, True, False, False])
        topology = build_graph_interface_topology(features, fixed)
        self.assertIsNotNone(topology)
        self.assertEqual(topology.junction_ca_pairs.shape, (4, 2))
        self.assertEqual(
            {
                tuple(pair)
                for pair in topology.junction_ca_pairs.tolist()
            },
            {(0, 1), (1, 2), (3, 4), (4, 5)},
        )
        regular = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [23.8, 0.0, 0.0],
                [27.6, 0.0, 0.0],
            ]]
        )
        stretched = regular.clone()
        stretched[:, 1, 0] = 10.0
        config = GraphInterfaceGuidanceConfig()

        regular_energy = graph_interface_energy(regular, topology, config)
        stretched_energy = graph_interface_energy(stretched, topology, config)

        self.assertAlmostEqual(float(regular_energy.junction), 0.0)
        self.assertGreater(
            float(stretched_energy.junction),
            float(regular_energy.junction),
        )

    def test_chain_order_protects_compiled_junctions_with_numbering_gaps(
        self,
    ) -> None:
        features = self._features()
        # Compiled fixed and generated contigs can retain disjoint source
        # numbering even though their CA tokens are consecutive in one chain.
        features["residue_index"] = torch.tensor(
            [12, 101, 102, 12, 201, 202]
        )
        fixed = torch.tensor([True, False, False, True, False, False])

        topology = build_graph_interface_topology(features, fixed)

        self.assertIsNotNone(topology)
        self.assertEqual(
            {
                tuple(pair)
                for pair in topology.junction_ca_pairs.tolist()
            },
            {(0, 1), (1, 2), (3, 4), (4, 5)},
        )

    def test_physical_edges_cannot_reuse_the_same_generated_patch(
        self,
    ) -> None:
        def mask(indices: tuple[int, ...]) -> torch.Tensor:
            value = torch.zeros(9, dtype=torch.bool)
            value[list(indices)] = True
            return value

        alpha = GraphInterfaceEdge(
            edge_id="alpha@0",
            source_interface_id="alpha",
            left_generated_ca_mask=mask((0, 1, 2)),
            right_generated_ca_mask=mask((3, 4, 5)),
            left_generated_token_ids=torch.tensor([0, 1, 2]),
            right_generated_token_ids=torch.tensor([3, 4, 5]),
            requested_contact_count=3,
            requested_residues_per_side=3,
            requested_contiguous_residues_per_side=3,
            automatic_quality=False,
            contact_cutoff=5.5,
            distance_target=None,
            distance_tolerance=None,
        )
        beta = replace(
            alpha,
            edge_id="beta@0",
            source_interface_id="beta",
            right_generated_ca_mask=mask((6, 7, 8)),
            right_generated_token_ids=torch.tensor([6, 7, 8]),
        )
        coordinates = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [6.0, 0.0, 0.0],
                [6.0, 3.8, 0.0],
                [6.0, 7.6, 0.0],
                [-6.0, 0.0, 0.0],
                [-6.0, 3.8, 0.0],
                [-6.0, 7.6, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig()
        distinct = graph_interface_energy(
            coordinates,
            GraphInterfaceTopology(
                edges=(alpha, beta),
                generated_atom_mask=torch.ones(9, dtype=torch.bool),
            ),
            config,
        )
        same_type = graph_interface_energy(
            coordinates,
            GraphInterfaceTopology(
                edges=(alpha, replace(beta, source_interface_id="alpha")),
                generated_atom_mask=torch.ones(9, dtype=torch.bool),
            ),
            config,
        )

        self.assertGreater(float(distinct.patch_exclusivity), 0.0)
        self.assertGreater(float(same_type.patch_exclusivity), 0.0)
        self.assertAlmostEqual(
            float(distinct.patch_exclusivity),
            float(same_type.patch_exclusivity),
            places=6,
        )

    def test_terminal_quality_rejects_end_on_or_damaged_patches(self) -> None:
        topology = self._three_by_three_topology()
        packed = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 3.8, 0.0],
                [0.0, 7.6, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 3.8, 0.0],
                [7.0, 7.6, 0.0],
            ]]
        )
        end_on = torch.tensor(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]]
        )
        config = GraphInterfaceGuidanceConfig(
            maximum_shape_loss=100.0,
            maximum_backbone_loss=100.0,
        )

        self.assertTrue(
            graph_interface_quality_satisfied(
                graph_interface_energy(packed, topology, config),
                config=config,
            )
        )
        self.assertFalse(
            graph_interface_quality_satisfied(
                graph_interface_energy(end_on, topology, config),
                config=config,
            )
        )


if __name__ == "__main__":
    unittest.main()
