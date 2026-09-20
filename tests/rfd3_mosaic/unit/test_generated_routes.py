import math
import unittest

import torch

from rfd3.inference.symmetry.generated_routes import (
    apply_generated_route_guidance,
    generated_route_deficits,
)
from rfd3.inference.symmetry.scaffold_core_guidance import (
    ScaffoldCoreGuidanceConfig,
    build_scaffold_core_topology,
    scaffold_geometry_deficits,
)


def topology(chains, tokens=3, residue_indices=None):
    fixed = torch.tensor(([True] + [False] * (tokens - 2) + [True]) * chains)
    features = {
        "atom_to_token_map": torch.arange(chains * tokens),
        "asym_id": torch.arange(chains).repeat_interleave(tokens),
        "residue_index": torch.arange(tokens).repeat(chains) if residue_indices is None else residue_indices,
        "is_ca": torch.ones(chains * tokens, dtype=torch.bool),
        "is_protein": torch.ones(chains * tokens, dtype=torch.bool),
    }
    return build_scaffold_core_topology(features, fixed), fixed


class GeneratedRouteTests(unittest.TestCase):
    def test_invalid_coordinates_and_multi_design_batches_fail_explicitly(self):
        topo, _ = topology(2)
        config = ScaffoldCoreGuidanceConfig(routing_ownership_weight=1.)
        for coordinates in (torch.zeros(2, 6, 3), torch.full((1, 6, 3), float("nan"))):
            with self.assertRaises(ValueError):
                apply_generated_route_guidance(
                    coordinates, topo, progress=.5, config=config, projector=lambda v: v,
                )

    def test_c3_centre_has_outward_gradient_and_rotation_equivariance(self):
        topo, _ = topology(3)
        rotations = []
        master = torch.tensor([[12., -5., 0.], [0., 0., 0.], [12., 5., 0.]], dtype=torch.float64)
        for k in range(3):
            angle = k * 2 * math.pi / 3
            rotations.append(torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                                           [math.sin(angle), math.cos(angle), 0.],
                                           [0., 0., 1.]], dtype=torch.float64))
        x = torch.cat([master @ rotation.T for rotation in rotations]).requires_grad_(True)
        deficits = generated_route_deficits(x, topo)
        self.assertGreater(float(deficits.detach().square().sum()), 0.)
        gradient = torch.autograd.grad(deficits.square().sum(), x)[0]
        self.assertLess(float(gradient[1, 0]), 0.)
        self.assertAlmostEqual(float(gradient[1, 1]), 0., places=8)
        for k, rotation in enumerate(rotations):
            torch.testing.assert_close(gradient[3*k+1], gradient[1] @ rotation.T)

    def test_cyclic_and_dihedral_separated_routes_pass(self):
        from rfd3_mosaic.geometry import build_cyclic_registry, build_dihedral_registry
        master = torch.tensor([[30., -3.8, 12.], [30., 0., 12.], [30., 3.8, 12.]], dtype=torch.float64)
        for builder in (build_cyclic_registry, build_dihedral_registry):
            for order in (2, 3, 4, 6):
                registry = builder(order)
                topo, _ = topology(registry.order)
                copies = []
                for name in registry.transform_ids:
                    transform = torch.tensor(registry.transform(name), dtype=master.dtype)
                    copies.append(master @ transform[:3, :3].T + transform[:3, 3])
                with self.subTest(group=registry.group_name):
                    self.assertEqual(float(generated_route_deficits(torch.cat(copies), topo).max()), 0.)

    def test_no_clash_tie_is_corrected_even_at_final_progress_without_moving_seed(self):
        topo, fixed = topology(2)
        x = torch.tensor([[[-3., 0., 0.], [0., 5., 0.], [3., 0., 0.],
                           [-3., 10., 0.], [0., 5., 8.], [3., 10., 0.]]])
        config = ScaffoldCoreGuidanceConfig(routing_ownership_weight=1.)
        initial = scaffold_geometry_deficits(x, topo, config)
        self.assertEqual(float(initial["ca_overlap"].sum()), 0.)
        self.assertGreater(float(initial["route_ownership"].sum()), 0.)
        result, diagnostic = apply_generated_route_guidance(
            x, topo, progress=1., config=config, projector=lambda v: v,
        )
        self.assertTrue(diagnostic["applied"])
        self.assertLess(diagnostic["final"]["squared_excess_angstrom2"],
                        diagnostic["initial"]["squared_excess_angstrom2"])
        self.assertTrue(torch.equal(result[:, fixed], x[:, fixed]))
        final = scaffold_geometry_deficits(result, topo, config)
        for name in ("ca_overlap", "continuity", "cross_chain_segment_overlap"):
            self.assertTrue(torch.all(final[name] <= initial[name] + 1e-6))

    def test_missing_residues_do_not_create_a_fictitious_anchored_route(self):
        topo, _ = topology(2, 4, torch.tensor([0, 1, 4, 5] * 2))
        self.assertEqual(len(topo.generated_runs), 0)

    def test_coincident_reference_is_nonzero_and_cannot_silently_pass(self):
        topo, _ = topology(2)
        x = torch.tensor([[-3., 0., 0.], [0., 5., 0.], [3., 0., 0.],
                          [-3., 0., 0.], [0., -5., 0.], [3., 0., 0.]])
        residual = generated_route_deficits(x, topo)
        self.assertGreater(float(residual.min()), 0.)

    def test_mobile_reference_moves_with_seed_rigid_transform(self):
        topo, _ = topology(2)
        x = torch.tensor([[-3., 0., 0.], [0., 5., 0.], [3., 0., 0.],
                          [-3., 10., 0.], [0., 5., 8.], [3., 10., 0.]], dtype=torch.float64)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=x.dtype)
        moved = x @ rotation.T + torch.tensor([5., 7., 12.], dtype=x.dtype)
        torch.testing.assert_close(generated_route_deficits(x, topo), generated_route_deficits(moved, topo))

    def test_disabled_guidance_preserves_coordinates(self):
        topo, _ = topology(2)
        x = torch.randn(1, 6, 3)
        result, diagnostic = apply_generated_route_guidance(
            x, topo, progress=.5, config=ScaffoldCoreGuidanceConfig(), projector=lambda v: v,
        )
        self.assertTrue(torch.equal(x, result))
        self.assertFalse(diagnostic["applied"])
