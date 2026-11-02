"""Inactive guidance keeps diagnostics without constructing a backward graph."""
import unittest
from unittest.mock import patch
import torch
from rfd3.inference.symmetry import scaffold_core_guidance as core


class InactiveCoreAutogradTest(unittest.TestCase):
    def test_inactive_window_preserves_energy_without_graph(self):
        f = {
            'atom_to_token_map': torch.arange(6),
            'asym_id': torch.zeros(6, dtype=torch.long),
            'residue_index': torch.arange(6),
            'is_ca': torch.ones(6, dtype=torch.bool),
            'is_protein': torch.ones(6, dtype=torch.bool),
        }
        fixed = torch.tensor([True, False, False, False, False, True])
        topology = core.build_scaffold_core_topology(f, fixed)
        x = torch.stack((torch.arange(6) * 3.8, torch.zeros(6), torch.zeros(6)), -1)[None]
        config = core.ScaffoldCoreGuidanceConfig(intra_chain_weight=1.)
        expected = core.scaffold_core_energy(x[0], topology, config).detached_dict()
        energy = core.scaffold_core_energy
        observations = []

        def inspect(*args, **kwargs):
            result = energy(*args, **kwargs)
            observations.append((torch.is_grad_enabled(), result.total.requires_grad))
            return result

        for progress in (0., config.start_fraction, config.end_fraction, 1.):
            with self.subTest(progress=progress), patch.object(core, 'scaffold_core_energy', inspect):
                result, diagnostics = core.apply_scaffold_core_guidance(
                    x, topology, progress=progress, config=config,
                    record_rejected_trial_energies=False,
                )
            self.assertTrue(torch.equal(result, x))
            self.assertEqual(diagnostics['initial'], expected)
            self.assertEqual(diagnostics['final'], expected)
            self.assertEqual(diagnostics['reason'], 'inactive_window')
            self.assertEqual(observations[-1], (False, False))
