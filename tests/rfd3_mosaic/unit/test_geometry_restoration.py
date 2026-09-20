import unittest

import torch
from rfd3.inference.symmetry.geometry_restoration import restore_generated_geometry
from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig, build_scaffold_core_topology,
)


class GeometryRestorationTests(unittest.TestCase):
    def topology(self, fixed):
        return build_scaffold_core_topology({
            "atom_to_token_map": torch.arange(4),
            "asym_id": torch.tensor([0, 0, 1, 1]),
            "residue_index": torch.tensor([0, 1, 0, 1]),
            "is_ca": torch.ones(4, dtype=torch.bool),
            "is_protein": torch.ones(4, dtype=torch.bool),
        }, fixed)

    def test_restores_separated_broken_chains_without_moving_fixed_anchors(self):
        fixed = torch.tensor([True, False, True, False])
        x = torch.tensor([[[0., 0, 0], [6., 0, 0], [0., 10, 0], [6., 10, 0]]])
        result, info = restore_generated_geometry(
            x, self.topology(fixed), ScaffoldCoreGuidanceConfig(), projector=lambda v: v,
        )
        self.assertTrue(info["within_tolerance"])
        self.assertTrue(torch.equal(result[:, fixed], x[:, fixed]))
        self.assertLess(info["final"]["continuity"]["maximum_violation"], 1e-6)

    def test_valid_structure_is_unchanged(self):
        x = torch.tensor([[[0., 0, 0], [3.8, 0, 0], [0., 10, 0], [3.8, 10, 0]]])
        result, info = restore_generated_geometry(
            x, self.topology(torch.zeros(4, dtype=torch.bool)),
            ScaffoldCoreGuidanceConfig(), projector=lambda v: v,
        )
        self.assertTrue(torch.equal(result, x))
        self.assertFalse(info["applied"])

    def test_projection_cannot_introduce_new_clashes_to_fix_a_bond(self):
        fixed = torch.tensor([True, False, True, False])
        x = torch.tensor([[[0., 0, 0], [6., 0, 0], [0., 10, 0], [6., 10, 0]]])
        def bad_projector(v):
            v = v.clone()
            v[:, 3] = v[:, 1]
            return v
        result, info = restore_generated_geometry(
            x, self.topology(fixed), ScaffoldCoreGuidanceConfig(), projector=bad_projector,
        )
        self.assertTrue(torch.equal(result, x))
        self.assertFalse(info["within_tolerance"])
        self.assertFalse(info["applied"])

    def test_near_crossing_can_separate_without_changing_chain_lengths(self):
        x = torch.tensor([[[-1.9, 0, 0], [1.9, 0, 0], [0., -1.9, .2], [0., 1.9, .2]]])
        result, info = restore_generated_geometry(
            x, self.topology(torch.zeros(4, dtype=torch.bool)),
            ScaffoldCoreGuidanceConfig(), projector=lambda v: v,
        )
        self.assertTrue(info["applied"])
        self.assertLess(info["final"]["cross_chain_segment_overlap"]["maximum_violation"],
                        info["initial"]["cross_chain_segment_overlap"]["maximum_violation"])
        self.assertEqual(info["final"]["continuity"]["violated_constraints"], 0)
